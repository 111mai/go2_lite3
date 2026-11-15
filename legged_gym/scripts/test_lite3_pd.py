#!/usr/bin/env python3
"""在平地上用零动作检查 Lite3 的 PD 站立控制，不运行 PPO。"""

# Isaac Gym 必须先于 PyTorch 导入。
import isaacgym
from isaacgym import gymtorch
import torch

import math
from types import MethodType

import legged_gym.envs  # 导入时注册 go2/lite3 等任务
from legged_gym.envs.lite3.lite3_config import Lite3Cfg
from legged_gym.utils import get_args, task_registry


class Lite3PDTestCfg(Lite3Cfg):
    """仅供本脚本使用的临时配置，不改变注册的 Lite3Cfg。"""

    class env(Lite3Cfg.env):
        num_envs = 1
        test = False

    class terrain(Lite3Cfg.terrain):
        # 平地仍保留高度采样点；plane 模式返回 187 个零高度，
        # 因此继续匹配继承的 263 维特权观测。
        mesh_type = "plane"
        curriculum = False
        measure_heights = True

    class init_state(Lite3Cfg.init_state):
        turn_over = False

    class domain_rand(Lite3Cfg.domain_rand):
        # 关闭会改变机器人物理属性、PD 参数或外力的随机化。
        robot_properties_update = None
        randomize_friction = False
        randomize_base_mass = False
        randomize_link_mass = False
        randomize_base_com = False
        randomize_restitution = False
        randomize_pd_gains = False
        randomize_motor_zero_offset = False
        randomize_motor_strength = False
        push_robots = False
        randomize_action_delay = False

    class noise(Lite3Cfg.noise):
        add_noise = False


def _tensor_values(tensor, digits=4):
    return [round(float(value), digits) for value in tensor.detach().cpu().flatten().tolist()]


def _nonfinite_fields(fields):
    bad = []
    for name, value in fields.items():
        if value is not None and not bool(torch.isfinite(value).all().item()):
            bad.append(name)
    return bad


def main():
    args = get_args()
    if args.task != "lite3":
        raise ValueError(f"本脚本只运行 lite3 任务，当前 task={args.task!r}")

    test_cfg = Lite3PDTestCfg()
    # 传入自定义环境配置时，make_env 不会像 get_cfgs() 那样自动补上 seed。
    # 复用 lite3 任务注册的 PPO seed，保持与正式任务一致。
    test_cfg.seed = task_registry.train_cfgs["lite3"].seed
    env = None
    reset_events = []
    nonfinite_seen = set()
    limit_violation_steps = 0
    limit_violation_names = set()
    torso_heights = []
    roll_values = []
    pitch_values = []
    max_abs_torque_seen = 0.0
    completed_steps = 0

    try:
        env, cfg = task_registry.make_env(
            name="lite3", args=args, env_cfg=test_cfg
        )

        if env.num_envs != 1:
            raise RuntimeError(f"期望 1 个环境，实际创建 {env.num_envs} 个")
        if env.num_actions != 12 or cfg.env.num_actions != 12:
            raise RuntimeError(
                f"期望 12 维 Action，实际 env={env.num_actions}, "
                f"cfg={cfg.env.num_actions}"
            )
        if not str(env.device).startswith("cuda"):
            raise RuntimeError(
                f"期望 GPU tensor pipeline，实际环境 device={env.device!r}"
            )
        if cfg.env.num_observations != 45 or env.obs_buf.shape[1] != 45:
            raise RuntimeError(
                f"Actor Observation 维度不匹配：cfg={cfg.env.num_observations}, "
                f"buffer={env.obs_buf.shape[1]}"
            )
        height_count = (
            len(cfg.terrain.measured_points_x)
            * len(cfg.terrain.measured_points_y)
        )
        expected_privileged = (
            cfg.env.num_observations + 3 + 4
            + cfg.env.num_actions + cfg.env.num_actions + height_count
        )
        if not cfg.terrain.measure_heights:
            raise RuntimeError("平地测试必须保留高度测量以匹配特权观测")
        if cfg.env.num_privileged_obs != expected_privileged:
            raise RuntimeError(
                "Privileged Observation 配置不匹配："
                f"cfg={cfg.env.num_privileged_obs}, expected={expected_privileged}"
            )
        if env.privileged_obs_buf is None or env.privileged_obs_buf.shape[1] != expected_privileged:
            actual = None if env.privileged_obs_buf is None else env.privileged_obs_buf.shape[1]
            raise RuntimeError(
                f"Privileged Observation buffer 维度不匹配：actual={actual}, "
                f"expected={expected_privileged}"
            )

        print("Temporary terrain:", cfg.terrain.mesh_type)
        print("Terrain curriculum:", cfg.terrain.curriculum)
        print("Height measurements:", cfg.terrain.measure_heights)
        print("Height sample count:", height_count)
        print("Actor/critic observation dimensions:",
              env.obs_buf.shape[1], env.privileged_obs_buf.shape[1])
        print("Environment/control dt:", env.dt, "s")
        print("Control decimation:", cfg.control.decimation)
        print("Randomization switches:", {
            name: getattr(cfg.domain_rand, name)
            for name in (
                "randomize_friction", "randomize_base_mass",
                "randomize_link_mass", "randomize_base_com",
                "randomize_restitution", "randomize_pd_gains",
                "randomize_motor_zero_offset", "randomize_motor_strength",
                "push_robots", "randomize_action_delay",
            )
        })

        # 基类创建 Actor 时把初始位置设为环境原点；这里显式应用配置中的
        # Lite3 初始根位置，避免机器人从地面内开始测试。测试固定为直立、零根速度。
        actor_ids = torch.arange(
            env.num_envs, dtype=torch.int32, device=env.device
        )
        with torch.no_grad():
            init_pos = torch.as_tensor(
                cfg.init_state.pos, dtype=torch.float32, device=env.device
            )
            env.root_states[:, 0:3] = env.env_origins + init_pos
            env.root_states[:, 3:7] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=env.device
            )
            env.root_states[:, 7:13] = 0.0
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        print(
            "Initialized root position:",
            _tensor_values(env.root_states[0, 0:3]),
            "m; linear/angular velocity:",
            _tensor_values(env.root_states[0, 7:13]),
        )

        # 新建 Actor 的关节状态不一定等于 Lite3 的默认站姿；在第一个
        # 零动作控制步前，将关节状态设为配置默认角度和零速度。
        with torch.no_grad():
            env.dof_pos[:] = env.default_dof_pos.expand(env.num_envs, -1)
            env.dof_vel.zero_()
            env.last_dof_vel.zero_()
            env.actions.zero_()
            env.last_actions.zero_()
            env.last_last_actions.zero_()
        env.gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.dof_state),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )
        env.gym.refresh_dof_state_tensor(env.sim)

        dof_props = env.gym.get_actor_dof_properties(
            env.envs[0], env.actor_handles[0]
        )
        lower = torch.tensor(
            dof_props["lower"].copy(), dtype=torch.float32, device=env.device
        )
        upper = torch.tensor(
            dof_props["upper"].copy(), dtype=torch.float32, device=env.device
        )
        dof_names = list(env.dof_names)
        body_names = list(env.gym.get_asset_rigid_body_names(env.robot_asset))
        torso_index = body_names.index("TORSO")
        termination_indices = [int(i) for i in env.termination_contact_indices.detach().cpu().tolist()]
        termination_bodies = [(i, body_names[i]) for i in termination_indices]
        rigid_body_states = env.rigid_body_states.view(
            env.num_envs, env.num_bodies, 13
        )

        def state_snapshot(env_id=0):
            q = env.dof_pos[env_id]
            outside = ((q < lower) | (q > upper)).nonzero(as_tuple=False).flatten()
            outside_names = [dof_names[int(i)] for i in outside.detach().cpu().tolist()]
            return {
                "torso_z": float(rigid_body_states[env_id, torso_index, 2].item()),
                "roll": float(env.rpy[env_id, 0].item()),
                "pitch": float(env.rpy[env_id, 1].item()),
                "q": _tensor_values(q, 4),
                "out_of_range": outside_names,
                "max_abs_torque": float(env.torques[env_id].abs().max().item()),
                "episode_step": int(env.episode_length_buf[env_id].item()),
            }

        # reset_idx 在终止检测后、状态重置前被调用；包装该实例方法，
        # 只记录事件，不改变 Go2Robot 的重置行为。
        original_reset_idx = env.reset_idx

        def capture_reset(self, env_ids):
            ids = env_ids.detach().cpu().tolist()
            for env_id in ids:
                timeout = bool(self.time_out_buf[env_id].item())
                contacts = []
                for body_index, body_name in termination_bodies:
                    magnitude = float(
                        torch.linalg.vector_norm(
                            self.contact_forces[env_id, body_index]
                        ).item()
                    )
                    if magnitude > 1.0:
                        contacts.append(f"{body_name}={magnitude:.3f}N")

                reasons = []
                if timeout:
                    reasons.append("episode_timeout")
                if contacts:
                    reasons.append("termination_contact:" + ",".join(contacts))
                if not reasons:
                    reasons.append("reset_flag_without_known_cause")

                event = state_snapshot(env_id)
                bad_before_reset = _nonfinite_fields({
                    "root_states": self.root_states[env_id],
                    "dof_pos": self.dof_pos[env_id],
                    "dof_vel": self.dof_vel[env_id],
                    "rpy": self.rpy[env_id],
                    "contact_forces": self.contact_forces[env_id],
                    "torques": self.torques[env_id],
                })
                nonfinite_seen.update(bad_before_reset)
                event.update({
                    "time": float(self.gym.get_sim_time(self.sim)),
                    "env_id": int(env_id),
                    "reason": "; ".join(reasons),
                    "nonfinite_fields": bad_before_reset,
                })
                reset_events.append(event)
                print(
                    "[RESET BEFORE STATE RESET] "
                    f"t={event['time']:.3f}s env={event['env_id']} "
                    f"reason={event['reason']} episode_step={event['episode_step']} "
                    f"TORSO_z={event['torso_z']:.4f}m "
                    f"roll={event['roll']:.4f}rad pitch={event['pitch']:.4f}rad "
                    f"out_of_range={event['out_of_range']} "
                    f"max_abs_torque={event['max_abs_torque']:.3f}Nm "
                    f"NaN/Inf={bad_before_reset or 'no'} q={event['q']}",
                    flush=True,
                )
            return original_reset_idx(env_ids)

        env.reset_idx = MethodType(capture_reset, env)

        test_duration = 3.0
        report_interval = 0.2
        control_dt = float(env.dt)
        total_steps = int(math.ceil(test_duration / control_dt))
        actions = torch.zeros(
            (env.num_envs, env.num_actions), dtype=torch.float32, device=env.device
        )
        print(
            f"Starting zero-action PD test: target={test_duration:.2f}s, "
            f"control_dt={control_dt:.4f}s, steps={total_steps}",
            flush=True,
        )

        next_report_time = report_interval
        events_at_last_report = 0
        last_snapshot = state_snapshot()
        for step in range(total_steps):
            previous_event_count = len(reset_events)
            obs, privileged_obs, rewards, dones, _ = env.step(actions)
            completed_steps = step + 1
            sim_time = float(env.gym.get_sim_time(env.sim))
            new_events = reset_events[previous_event_count:]
            reset_this_step = bool(torch.as_tensor(dones).any().item()) or bool(new_events)

            finite_tensors = {
                "root_states": env.root_states,
                "dof_pos": env.dof_pos,
                "dof_vel": env.dof_vel,
                "rpy": env.rpy,
                "contact_forces": env.contact_forces,
                "torques": env.torques,
                "actor_obs": obs,
                "privileged_obs": privileged_obs,
                "rewards": rewards,
            }
            bad_now = _nonfinite_fields(finite_tensors)
            nonfinite_seen.update(bad_now)

            # 若本步触发了自动重置，周期日志使用重置前快照，
            # 避免把复位后的正常姿态误当作故障前状态。
            snapshot = new_events[-1] if new_events else state_snapshot()
            last_snapshot = snapshot
            if snapshot["out_of_range"]:
                limit_violation_steps += 1
                limit_violation_names.update(snapshot["out_of_range"])
            if math.isfinite(snapshot["torso_z"]):
                torso_heights.append(snapshot["torso_z"])
            if math.isfinite(snapshot["roll"]):
                roll_values.append(snapshot["roll"])
            if math.isfinite(snapshot["pitch"]):
                pitch_values.append(snapshot["pitch"])
            if math.isfinite(snapshot["max_abs_torque"]):
                max_abs_torque_seen = max(
                    max_abs_torque_seen, snapshot["max_abs_torque"]
                )

            should_report = sim_time + 1e-8 >= next_report_time or bool(new_events)
            if bad_now:
                should_report = True
            if step == total_steps - 1:
                should_report = True
            if should_report:
                interval_events = reset_events[events_at_last_report:]
                interval_causes = [event["reason"] for event in interval_events]
                phase = "pre_reset_snapshot" if new_events else "current_state"
                print(
                    f"[t={sim_time:.2f}s] state={phase} "
                    f"TORSO_z={snapshot['torso_z']:.4f}m "
                    f"roll={snapshot['roll']:.4f}rad pitch={snapshot['pitch']:.4f}rad "
                    f"out_of_range={snapshot['out_of_range']} "
                    f"q={snapshot['q']} "
                    f"max_abs_PD_torque={snapshot['max_abs_torque']:.3f}Nm "
                    f"reset_this_step={reset_this_step} "
                    f"resets_since_last_report={len(interval_events)} "
                    f"reset_causes={interval_causes or 'none'} "
                    f"NaN/Inf={'YES:' + ','.join(sorted(nonfinite_seen)) if nonfinite_seen else 'no'}",
                    flush=True,
                )
                events_at_last_report = len(reset_events)
                while next_report_time <= sim_time + 1e-8:
                    next_report_time += report_interval

            if bad_now:
                print(
                    f"Stopping early at t={sim_time:.3f}s because non-finite "
                    f"values appeared in: {bad_now}",
                    flush=True,
                )
                break
            if new_events:
                print(
                    "Stopping after the first automatic reset was captured; "
                    "the base reset routine randomizes the restarted base velocity.",
                    flush=True,
                )
                break

        print("\n=== Zero-action PD test summary ===")
        print(f"Simulated time: {float(env.gym.get_sim_time(env.sim)):.3f}s")
        print(f"Control steps completed: {completed_steps}/{total_steps}")
        print(f"Automatic resets: {len(reset_events)}")
        print(f"Physical joint-limit violation steps: {limit_violation_steps}")
        print(f"Joints observed outside physical limits: {sorted(limit_violation_names) or 'none'}")
        print(f"NaN/Inf observed: {sorted(nonfinite_seen) or 'no'}")
        if torso_heights:
            print(
                f"TORSO height range: {min(torso_heights):.4f}.."
                f"{max(torso_heights):.4f}m"
            )
        if roll_values and pitch_values:
            print(
                f"Maximum absolute roll/pitch: "
                f"{max(abs(x) for x in roll_values):.4f}/"
                f"{max(abs(x) for x in pitch_values):.4f}rad"
            )
        print(f"Maximum absolute PD torque: {max_abs_torque_seen:.3f}Nm")
        print("Final/pre-reset state snapshot:", last_snapshot)
        if reset_events:
            print("Reset events (pre-reset snapshots):")
            for event in reset_events:
                print(event)
        if not reset_events and not nonfinite_seen and limit_violation_steps == 0:
            print(
                "Result: no reset, NaN/Inf, or physical joint-limit violation "
                "was detected. Use the reported TORSO height and attitude trace "
                "to judge upright stability; the base environment does not reset "
                "solely on a roll/pitch threshold."
            )
        else:
            print("Result: see reset events and abnormal-state counts above.")
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
