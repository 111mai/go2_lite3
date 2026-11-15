#!/usr/bin/env python3
"""Export and validate the Lite3 MoE-CTS student policy without Isaac Gym."""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch import Tensor
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS

# Import the existing exporter file directly: importing legged_gym.utils first
# runs its package initializer, which imports Isaac Gym and requires torch to
# be imported afterward. Exporting this CPU policy does not need Isaac Gym.
_exporter_spec = importlib.util.spec_from_file_location(
    "lite3_existing_exporter", REPO_ROOT / "legged_gym/utils/exporter.py"
)
if _exporter_spec is None or _exporter_spec.loader is None:
    raise ImportError("Could not load legged_gym/utils/exporter.py")
_exporter_module = importlib.util.module_from_spec(_exporter_spec)
_exporter_spec.loader.exec_module(_exporter_module)
export_policy_as_jit = _exporter_module.export_policy_as_jit


DEFAULT_RUN = REPO_ROOT / "logs/lite3_moe_cts/Oct14_16-32-10_main_moe_cts_4096_150k"
DEFAULT_CHECKPOINT = DEFAULT_RUN / "model_150000.pt"
DEFAULT_CONFIG = DEFAULT_RUN / "config.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "exports/lite3_moe_cts/student_150000.pt"


class ActionOnlyStudent(nn.Module):
    """Remove the exporter's MoE diagnostics from the deployment interface."""

    def __init__(self, exported_policy):
        super().__init__()
        self.exported_policy = exported_policy

    def forward(self, obs: Tensor) -> Tensor:
        result = self.exported_policy(obs)
        return result[0]

    @torch.jit.export
    def reset(self) -> None:
        self.exported_policy.reset()


def load_policy(config_path: Path, checkpoint_path: Path):
    with config_path.open("r", encoding="utf-8") as stream:
        saved_config = yaml.safe_load(stream)

    train_cfg = saved_config["train_cfg"]
    env_cfg = saved_config["env_cfg"]["env"]
    runner_cfg = train_cfg["runner"]
    if runner_cfg["policy_class_name"] != "ActorCriticMoECTS":
        raise ValueError("Run config does not select ActorCriticMoECTS")
    if runner_cfg["algorithm_class_name"] != "MoECTS":
        raise ValueError("Run config does not select MoECTS")

    num_obs = int(env_cfg["num_observations"])
    num_critic_obs = int(env_cfg["num_privileged_obs"])
    num_actions = int(env_cfg["num_actions"])
    history_length = int(train_cfg["history_length"])
    if (num_obs, num_actions, history_length) != (45, 12, 5):
        raise ValueError(
            "Unexpected Lite3 policy dimensions: "
            f"obs={num_obs}, actions={num_actions}, history={history_length}"
        )

    policy = ActorCriticMoECTS(
        num_obs,
        num_critic_obs,
        num_actions,
        1,
        history_length,
        **train_cfg["policy"],
    ).cpu()

    # OnPolicyRunnerCTS.load restores model_state_dict before touching optimizer state.
    # Reproduce that model restore directly so no VecEnv / Isaac Gym is constructed.
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    iteration = int(checkpoint["iter"])
    if iteration != 150000:
        raise ValueError(f"Expected checkpoint iteration 150000, found {iteration}")
    missing, unexpected = policy.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
        )
    policy.eval()
    return policy, saved_config, iteration


def finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def verify_export(
    policy: ActorCriticMoECTS,
    raw_export_path: Path,
    final_export_path: Path,
    num_obs: int,
    num_actions: int,
    expert_num: int,
    latent_dim: int,
):
    raw = torch.jit.load(str(raw_export_path), map_location="cpu").eval()

    # The project's exporter returns (actions, (expert_weights, latent)). Confirm
    # the first tuple element is the action tensor before stripping diagnostics.
    probe = torch.linspace(-0.25, 0.25, num_obs, dtype=torch.float32).reshape(1, -1)
    policy.history.zero_()
    with torch.no_grad():
        reference_probe = policy.act_inference(probe)
    raw.reset()
    with torch.no_grad():
        raw_result = raw(probe)
    if not isinstance(raw_result, tuple) or len(raw_result) != 2:
        raise AssertionError("Existing JIT exporter did not return (actions, auxiliary)")
    raw_actions, auxiliary = raw_result
    if not isinstance(auxiliary, tuple) or len(auxiliary) != 2:
        raise AssertionError("MoE auxiliary result is not (expert_weights, latent)")
    weights, latent = auxiliary
    if tuple(raw_actions.shape) != (1, num_actions):
        raise AssertionError(f"Raw JIT action shape is {tuple(raw_actions.shape)}")
    if tuple(weights.shape) != (1, expert_num):
        raise AssertionError(f"MoE weights shape is {tuple(weights.shape)}")
    if tuple(latent.shape) != (1, latent_dim):
        raise AssertionError(f"MoE latent shape is {tuple(latent.shape)}")
    if not torch.allclose(raw_actions, reference_probe, rtol=1e-6, atol=1e-7):
        raise AssertionError("Raw JIT tuple element 0 does not match deterministic actions")
    raw.reset()
    policy.history.zero_()

    scripted = torch.jit.script(ActionOnlyStudent(raw).eval())
    torch.jit.save(scripted, str(final_export_path))
    del scripted

    # Exercise loading from disk, rather than validating only the in-memory module.
    exported = torch.jit.load(str(final_export_path), map_location="cpu").eval()
    exported.reset()
    sequence_len = 11
    generator = torch.Generator(device="cpu").manual_seed(20261102)
    sequence = [
        torch.randn((1, num_obs), generator=generator, dtype=torch.float32) * 0.25
        for _ in range(sequence_len)
    ]
    expected_history = torch.zeros((1, 5, num_obs), dtype=torch.float32)
    reference_outputs = []
    exported_outputs = []
    max_abs_error = 0.0
    history_matches_reference = True
    history_matches_expected = True

    policy.history.zero_()
    with torch.no_grad():
        for obs in sequence:
            expected_history = torch.cat(
                (expected_history[:, 1:, :], obs.unsqueeze(1)), dim=1
            )
            reference_action = policy.act_inference(obs)
            exported_action = exported(obs)
            if tuple(exported_action.shape) != (1, num_actions):
                raise AssertionError(
                    f"Action-only JIT output has shape {tuple(exported_action.shape)}"
                )
            if not finite(exported_action):
                raise AssertionError("Action-only JIT output contains NaN or Inf")
            step_error = float((reference_action - exported_action).abs().max().item())
            max_abs_error = max(max_abs_error, step_error)
            history_matches_reference &= torch.equal(policy.history, expected_history)
            history_matches_expected &= torch.equal(
                exported.exported_policy.history, expected_history
            )
            reference_outputs.append(reference_action.clone())
            exported_outputs.append(exported_action.clone())

    if not history_matches_reference:
        raise AssertionError("PyTorch reference history did not roll over as expected")
    if not history_matches_expected:
        raise AssertionError("TorchScript history did not roll over across 5 frames")
    if max_abs_error > 1e-6:
        raise AssertionError(f"TorchScript mismatch: max abs error {max_abs_error}")

    # reset() must clear all cached frames, and replay must match a fresh run.
    exported.reset()
    policy.history.zero_()
    cleared_history = exported.exported_policy.history
    reset_cleared_history = bool(torch.count_nonzero(cleared_history).item() == 0)
    if not reset_cleared_history:
        raise AssertionError("TorchScript reset() did not clear history")
    replay_outputs = []
    with torch.no_grad():
        for obs in sequence:
            replay_outputs.append(exported(obs).clone())
    replay_matches = all(
        torch.equal(first, replay)
        for first, replay in zip(exported_outputs, replay_outputs)
    )
    if not replay_matches:
        raise AssertionError("Outputs after reset did not match the fresh history replay")

    # A reset followed by the same initial observation is deterministic.
    exported.reset()
    with torch.no_grad():
        first = exported(probe)
    exported.reset()
    with torch.no_grad():
        second = exported(probe)
    if not torch.equal(first, second):
        raise AssertionError("Repeated deterministic inference produced different actions")

    return {
        "torchscript_load": True,
        "input_shape": [1, num_obs],
        "action_shape": [1, num_actions],
        "finite_actions": True,
        "deterministic_actions": True,
        "history_sequence_length": sequence_len,
        "five_frame_history_rollover": True,
        "reset_clears_history": reset_cleared_history,
        "reset_replay_matches": replay_matches,
        "max_abs_error_vs_pytorch_student": max_abs_error,
        "raw_export_return": "(actions, (expert_weights, latent))",
        "raw_export_action_shape": list(raw_actions.shape),
        "raw_export_expert_weights_shape": list(weights.shape),
        "raw_export_latent_shape": list(latent.shape),
        "deployment_output": "actions only; auxiliary MoE tensors removed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing export and report"
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    report_path = args.output.with_suffix(".validation.json")
    if not args.overwrite and (args.output.exists() or report_path.exists()):
        raise FileExistsError(
            f"Export or report already exists; pass --overwrite to replace: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    policy, saved_config, iteration = load_policy(args.config, args.checkpoint)
    train_cfg = saved_config["train_cfg"]
    env_cfg = saved_config["env_cfg"]["env"]
    policy_cfg = train_cfg["policy"]
    num_obs = int(env_cfg["num_observations"])
    num_actions = int(env_cfg["num_actions"])

    with tempfile.TemporaryDirectory(
        prefix=".lite3_moe_cts_export_", dir=str(args.output.parent)
    ) as temporary_dir:
        temporary = Path(temporary_dir)
        raw_dir = temporary / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "student_with_aux.pt"
        candidate_path = temporary / args.output.name

        export_policy_as_jit(policy, str(raw_dir), filename=raw_path.name)
        results = verify_export(
            policy,
            raw_path,
            candidate_path,
            num_obs,
            num_actions,
            int(policy_cfg["expert_num"]),
            int(policy_cfg["latent_dim"]),
        )
        results.update(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_iteration": iteration,
                "run_config": str(args.config.resolve()),
                "architecture": {
                    "policy": "ActorCriticMoECTS",
                    "observation_dim": num_obs,
                    "critic_observation_dim": int(env_cfg["num_privileged_obs"]),
                    "history_length": int(train_cfg["history_length"]),
                    "expert_count": int(policy_cfg["expert_num"]),
                    "latent_dim": int(policy_cfg["latent_dim"]),
                    "action_dim": num_actions,
                },
                "device": "cpu",
                "exporter": "legged_gym.utils.exporter.export_policy_as_jit",
            }
        )
        report_candidate = temporary / report_path.name
        with report_candidate.open("w", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")

        os.replace(str(candidate_path), str(args.output))
        os.replace(str(report_candidate), str(report_path))

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("export:", args.output.resolve())
    print("report:", report_path.resolve())


if __name__ == "__main__":
    main()
