"""Lite3 的最小 Go2 配置子类。

这里仅覆盖 Lite3 机器人模型和硬件相关参数。没有重新定义的项目，
继续从 GO2Cfg 继承，包括 45 维 Actor Observation、263 维 Critic
Observation、地形与高度测量、课程学习、奖励、Dynamic Sigma 和
Domain Randomization。sim.dt 也暂时沿用 Go2，后续还要核验 Lite3
实际需要的控制周期。
"""

# 导入 Go2 的环境配置和 PPO 配置，Lite3 在这两者之上按需继承。
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO, GO2CfgMoECTS


class Lite3Cfg(GO2Cfg):
    """Lite3 环境配置：只写与 Go2 不同的部分，其余沿用父类。"""

    class init_state(GO2Cfg.init_state):
        # 继承 Go2 初始状态配置；未覆盖的初始旋转、线速度和角速度仍沿用 Go2。
        # 根部初始位置 [x, y, z]，单位米；z=0.32 是仿真摆放高度，和奖励目标高度不同。
        pos = [0.0, 0.0, 0.32]

        # 动作值为 0 时，各关节的默认目标角度，单位弧度；数值来自官方 Lite3RoughCfg。
        # 字典按关节名称匹配 URDF，不会与父类字典自动合并，因此要列出 Lite3 的全部 12 个活动关节。
        default_joint_angles = {
            # 左前腿：HipX（髋外展）、HipY（髋俯仰）、Knee（膝）。
            "FL_HipX_joint": 0.0,
            "FL_HipY_joint": -1.0,
            "FL_Knee_joint": 1.8,
            # 右前腿。
            "FR_HipX_joint": 0.0,
            "FR_HipY_joint": -1.0,
            "FR_Knee_joint": 1.8,
            # 左后腿。
            "HL_HipX_joint": 0.0,
            "HL_HipY_joint": -1.0,
            "HL_Knee_joint": 1.8,
            # 右后腿。
            "HR_HipX_joint": 0.0,
            "HR_HipY_joint": -1.0,
            "HR_Knee_joint": 1.8,
        }

    class control(GO2Cfg.control):
        # 继承 Go2 的控制模式、刚度 Kp=20、动作缩放 action_scale=0.25 和 decimation=4。
        # action_scale 将策略输出变成关节目标角度增量；decimation 表示每隔多少个仿真步更新一次策略动作。
        # 只把阻尼 Kd 改为 Lite3RoughCfg 的 0.7，单位 N·m·s/rad；"joint" 会匹配关节名称。
        stiffness = {"joint": 40.0}  # Kp，原来继承的是20.0
        damping = {"joint": 0.7}
        # 仅 Lite3 开启：PD 目标按软关节范围限幅，电机强度缩放后再次按 URDF 力矩限幅。
        # Go2 配置没有此开关，公共基类中会按关闭处理，保留原 Go2 行为。
        enable_lite3_safety_limits = True
        # sim.dt 属于单独的 sim 配置，不由上面的 control 类设置；当前继承 Go2 的 0.005 秒。
        # 与 decimation=4 相乘，策略动作周期为 0.020 秒，后续需核对 Lite3 的控制周期。

    class asset(GO2Cfg.asset):
        # URDF 文件路径。运行时 LEGGED_GYM_ROOT_DIR 会替换为 go2_lite3 工程根目录；
        # ../Lite3_rl_training 指同级的官方 Lite3 工程。当前配置依赖这两个工程保持同级。
        # URDF 中 ../meshes 的网格路径相对 URDF 所在目录解析。
        file = (
            "{LEGGED_GYM_ROOT_DIR}/../Lite3_rl_training/legged_gym/"
            "resources/robots/lite3/urdf/Lite3.urdf"
        )

        # Isaac Gym 中创建的机器人实例名称，不是关节或身体名称。
        name = "Lite3"

        # 以下字符串用于从 URDF 刚体名称中查找足端、碰撞惩罚部位和触发回合终止的部位。
        # 查找方式是大小写敏感的子串匹配，因此要与 Lite3 URDF 的大写命名相符。
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK"]
        terminate_after_contacts_on = ["TORSO"]

        # 来自官方 Lite3RoughCfg；Isaac Gym 该标志设为 1 表示禁用机器人自身碰撞。
        self_collisions = 1

    class rewards(GO2Cfg.rewards):
        # 奖励计算中希望机身达到的高度，单位米；来自官方 Lite3RoughCfg。
        # 这是目标高度，不是 init_state.pos 的初始根部高度；奖励函数本身仍沿用 Go2 实现。
        base_height_target = 0.36
        class scales(GO2Cfg.rewards.scales):
            tracking_lin_vel = 2.0###
            tracking_ang_vel = 1.0######
            stand_still = -0.05  # 低平移指令时，约束关节靠近默认角
    class terrain(GO2Cfg.terrain):
    # 最后一项 0.50 表示平地区域的总比例；
    # 下一步在 terrain.py 中再把它分成纯平地和 rough 平地。
        terrain_proportions = [
            0.0,   # 1. 波浪地形 wave：0%
            0.0,   # 2. 平滑坡 slope：0%
            0.0,   # 3. 粗糙坡 rough_slope：0%
            0.25,  # 4. 上楼梯 stairs_up：25%
            0.25,  # 5. 下楼梯 stairs_down：25%
            0.0,   # 6. 障碍物 obstacles：0%
            0.0,   # 7. 梅花桩 stepping_stones：0%
            0.0,   # 8. 间隙 gap：0%
            0.50,  # 9. 平地区域 flat：50%
        ]
    class commands(GO2Cfg.commands):
        # 速度课程：覆盖父类，防止后期超过你要求的上限。
        command_range_curriculum = [
            {
                "iter": 20000,
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.0, 1.0],
                "heading": [-1.57, 1.57],
            },
            {
                "iter": 30000,
                "lin_vel_x": [-1.167, 1.167],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.0, 1.0],
                "heading": [-1.57, 1.57],
            },
            {
                "iter": 40000,
                "lin_vel_x": [-1.333, 1.333],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.0, 1.0],
                "heading": [-1.57, 1.57],
            },
            {
                "iter": 50000,
                "lin_vel_x": [-1.5, 1.5],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.0, 1.0],
                "heading": [-1.57, 1.57],
            },
        ]
        # 复制父类列表，避免直接修改 GO2Cfg 的列表。
        terrain_max_command_ranges = GO2Cfg.commands.terrain_max_command_ranges.copy()

        # 上楼梯和下楼梯：不下发偏航角速度指令。
        terrain_max_command_ranges[3] = {
            "lin_vel_x": [-1.0, 1.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [0.0, 0.0],
            "heading": [-1.57, 1.57],
        }
        terrain_max_command_ranges[4] = {
            "lin_vel_x": [-1.0, 1.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [0.0, 0.0],
            "heading": [-1.57, 1.57],
        }

        # 纯平地和 rough 平地共用 ID 8。
        terrain_max_command_ranges[8] = {
            "lin_vel_x": [-1.5, 1.5],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-1.0, 1.0],
            "heading": [-1.57, 1.57],
        }

class Lite3CfgPPO(GO2CfgPPO):
    """Lite3 的 PPO 配置；算法、网络结构和训练参数均沿用 Go2。"""

    class runner(GO2CfgPPO.runner):
        # 单独使用 Lite3 日志根目录，避免与 Go2 的 logs/go2_ppo 混用。
        experiment_name = "lite3_ppo"
        # 冒烟训练从随机初始化开始，不从任何已有 Checkpoint 恢复。
        resume = False


class Lite3CfgMoECTS(GO2CfgMoECTS):
    """Lite3 的 MoE-CTS 配置；算法、网络和教师-学生机制沿用 Go2。"""
    seed = 1  # 之前使用的是 0
    class runner(GO2CfgMoECTS.runner):
        # 使用独立日志目录，不与 Go2 或 Lite3 普通 PPO 混用。
        experiment_name = "lite3_moe_cts"
        # 从头训练，不从 PPO 基线或其他实验的 Checkpoint 恢复。
        resume = False
