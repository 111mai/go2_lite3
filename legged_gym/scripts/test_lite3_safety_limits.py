#!/usr/bin/env python3
"""CPU 单元测试：Lite3 的 PD 目标和最终力矩保护，不创建 Isaac Gym 仿真。"""

import isaacgym  # noqa: F401  # 必须先于 torch / 项目模块导入
import torch

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.go2.go2_config import GO2Cfg
from legged_gym.envs.lite3.lite3_config import Lite3Cfg


def make_test_robot(enable_safety):
    """构造仅含 PD 计算所需张量的 LeggedRobot 测试对象。"""
    urdf_path = (
        Path(LEGGED_GYM_ROOT_DIR)
        / "../Lite3_rl_training/legged_gym/resources/robots/lite3/urdf/Lite3.urdf"
    ).resolve()
    urdf = ET.parse(urdf_path).getroot()
    limits_by_name = {}
    for joint in urdf.findall("joint"):
        limit = joint.find("limit")
        if joint.get("type") != "fixed" and limit is not None:
            physical_lower = float(limit.attrib["lower"])
            physical_upper = float(limit.attrib["upper"])
            midpoint = (physical_lower + physical_upper) / 2.0
            half_range = (physical_upper - physical_lower) / 2.0
            soft_factor = Lite3Cfg.rewards.soft_dof_pos_limit
            limits_by_name[joint.get("name")] = (
                midpoint - half_range * soft_factor,
                midpoint + half_range * soft_factor,
                float(limit.attrib["effort"]),
            )

    names = list(Lite3Cfg.init_state.default_joint_angles)
    defaults = torch.tensor(
        [Lite3Cfg.init_state.default_joint_angles[name] for name in names],
        dtype=torch.float32,
    ).view(1, -1)
    soft_limits = torch.tensor(
        [limits_by_name[name][:2] for name in names], dtype=torch.float32
    )
    efforts = torch.tensor(
        [limits_by_name[name][2] for name in names], dtype=torch.float32
    )

    robot = object.__new__(LeggedRobot)
    robot.cfg = SimpleNamespace(
        control=SimpleNamespace(
            control_type="P",
            action_scale=Lite3Cfg.control.action_scale,
            enable_lite3_safety_limits=enable_safety,
        ),
        domain_rand=SimpleNamespace(randomize_motor_strength=True),
    )
    robot.p_gains = torch.full((len(names),), 20.0)
    robot.d_gains = torch.full((len(names),), Lite3Cfg.control.damping["joint"])
    robot.p_gains_multiplier = torch.ones((1, len(names)))
    robot.d_gains_multiplier = torch.ones((1, len(names)))
    robot.default_dof_pos = defaults
    robot.dof_pos = defaults.clone()
    robot.dof_vel = torch.zeros_like(defaults)
    robot.last_dof_vel = torch.zeros_like(defaults)
    robot.motor_zero_offsets = torch.zeros_like(defaults)
    robot.dof_pos_limits = soft_limits
    robot.torque_limits = efforts
    robot.motor_strengths = torch.full_like(defaults, 1.2)
    robot.torques = torch.zeros_like(defaults)
    return robot, names, soft_limits, efforts


class Lite3SafetyLimitTests(unittest.TestCase):
    def test_safety_switch_is_specific_to_lite3(self):
        self.assertTrue(Lite3Cfg.control.enable_lite3_safety_limits)
        self.assertFalse(hasattr(GO2Cfg.control, "enable_lite3_safety_limits"))

    def test_oversized_actions_are_clipped_to_lite3_soft_joint_targets(self):
        robot, _, soft_limits, _ = make_test_robot(enable_safety=True)
        huge_positive = torch.full_like(robot.dof_pos, 100.0)
        huge_negative = torch.full_like(robot.dof_pos, -100.0)

        positive_torque = LeggedRobot._compute_torques(robot, huge_positive)
        negative_torque = LeggedRobot._compute_torques(robot, huge_negative)

        expected_positive = torch.clamp(
            20.0 * (soft_limits[:, 1] - robot.default_dof_pos[0]),
            min=-robot.torque_limits,
            max=robot.torque_limits,
        )
        expected_negative = torch.clamp(
            20.0 * (soft_limits[:, 0] - robot.default_dof_pos[0]),
            min=-robot.torque_limits,
            max=robot.torque_limits,
        )
        torch.testing.assert_close(positive_torque[0], expected_positive)
        torch.testing.assert_close(negative_torque[0], expected_negative)

    def test_motor_strength_1_2_is_clipped_against_urdf_effort(self):
        robot, _, _, efforts = make_test_robot(enable_safety=True)
        robot.torques[:] = efforts

        robot._apply_motor_strength_and_lite3_limit()

        torch.testing.assert_close(robot.torques[0], efforts)
        self.assertTrue(torch.all(torch.abs(robot.torques[0]) <= efforts).item())

        # 开关关闭时保留原有 Go2 行为：乘以 1.2 后不在基类额外限幅。
        legacy_robot, _, _, _ = make_test_robot(enable_safety=False)
        legacy_robot.torques[:] = efforts
        legacy_robot._apply_motor_strength_and_lite3_limit()
        torch.testing.assert_close(legacy_robot.torques[0], efforts * 1.2)

    def test_zero_action_pd_matches_legacy_when_target_and_torque_are_in_range(self):
        safe_robot, _, soft_limits, efforts = make_test_robot(enable_safety=True)
        legacy_robot, _, _, _ = make_test_robot(enable_safety=False)
        offset = torch.linspace(-0.02, 0.02, safe_robot.dof_pos.shape[1]).view(1, -1)
        velocity = torch.linspace(-0.1, 0.1, safe_robot.dof_pos.shape[1]).view(1, -1)
        safe_robot.dof_pos += offset
        legacy_robot.dof_pos += offset
        safe_robot.dof_vel[:] = velocity
        legacy_robot.dof_vel[:] = velocity
        zero_action = torch.zeros_like(safe_robot.dof_pos)
        nominal_target = safe_robot.default_dof_pos
        self.assertTrue(torch.all(nominal_target >= soft_limits[:, 0]).item())
        self.assertTrue(torch.all(nominal_target <= soft_limits[:, 1]).item())

        safe_torque = LeggedRobot._compute_torques(safe_robot, zero_action)
        legacy_torque = LeggedRobot._compute_torques(legacy_robot, zero_action)
        self.assertTrue(torch.all(torch.abs(safe_torque[0]) * 1.2 < efforts).item())
        safe_robot.torques[:] = safe_torque
        legacy_robot.torques[:] = legacy_torque
        safe_robot._apply_motor_strength_and_lite3_limit()
        legacy_robot._apply_motor_strength_and_lite3_limit()

        torch.testing.assert_close(safe_robot.torques, legacy_robot.torques)


if __name__ == "__main__":
    unittest.main(verbosity=2)
