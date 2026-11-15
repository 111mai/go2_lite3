# Lite3 SuperMimic: first-stage contracts

This isolated package is an initial framework, **not an implemented world-model
trainer**. It is not registered with existing Lite3/Go2 tasks and does not
modify the PPO, MoE-CTS, checkpoint, or deployment code.

The actor-facing interface follows the current project: observations `[B, 45]`,
12 action outputs, and five observations in chronological order `[B, 5, 45]`.
This is a policy observation contract, not a complete physics state. It does not
declare the units or physical semantics of a future world-model state.

`assemble_next_state` refuses to use returned reset data as a successor for
terminated environments. A future collector must actually capture pre-reset
physical state and provide a corresponding availability mask. It must also
record timeouts separately from other terminations for value bootstrapping;
both are episode boundaries for `valid_window_mask`. The present helpers do
not capture snapshots or perform any collection, and callers must apply the
validity mask to all losses and samples. Source provenance cannot be checked by
shape validation alone.

Pending: simulator adapter; handling randomized action delays and pushes;
replay storage and rare-event sampling; multistep world model; differentiable
imagination and task rewards; actor and TD(lambda) critic; evaluation and
export. None are replaced by a PPO update or claimed as complete.

CPU checks, from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' /mnt/A/hust_hbc/mai/envs/lite3gym/bin/python \
  -m unittest legged_gym.algorithms.supermimic_lite3.test_contracts -v
```
