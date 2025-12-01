"""
Helios Leg (LW-360 Gen2V1) biped wheeled-leg robot configuration.

This file defines the ArticulationCfg for IsaacLab/Isaac Sim to instantiate
the "Helios Leg / LW-360 Gen2V1" biped wheeled-leg robot.

Robot structure:
- Left and right legs, each with 3 joints:
  - thigh_joint: thigh joint (revolute)
  - calf_joint: calf joint (revolute)
  - foot_joint: wheel joint (continuous, unlimited rotation)
- Total 6 DOF (4 leg position control + 2 wheel velocity control)

Joint limits (from URDF):
- right_thigh_joint: [0.7, 2.7] rad, axis: (0, -1, 0)
- right_calf_joint: [-2.1, 0.9] rad, axis: (0, 1, 0)
- left_thigh_joint: [-2.7, -0.7] rad, axis: (0, 1, 0)
- left_calf_joint: [-0.9, 2.1] rad, axis: (0, -1, 0)
- foot_joint: continuous (wheel)

Usage: Reference HELIOS_LEG_CFG to spawn the robot in environment code.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


# URDF file path
HELIOS_LEG_URDF_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/helios_leg/urdf/LW-360 Gen2V1.urdf"


# ArticulationCfg: describes how to instantiate and drive the robot in simulation
HELIOS_LEG_CFG = ArticulationCfg(
    # spawn: specify URDF and conversion parameters
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,  # False means robot has floating base
        merge_fixed_joints=True,  # merge fixed joints to improve simulation performance
        replace_cylinders_with_capsules=False,
        asset_path=HELIOS_LEG_URDF_PATH,
        activate_contact_sensors=True,  # enable contact sensors for wheel ground detection
        # rigid body properties: damping/velocity limits, affects simulation stability
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # articulation root solver parameters
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # joint_drive: specify drive type and PD gains during conversion
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),

    # init_state: initial pose and joint states
    # Note: set reasonable initial angles within joint limits
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),  # initial position in world coordinates (x, y, z)
        # joint_pos: set initial angles for each joint (standing pose within limits)
        joint_pos={
            # right leg: thigh [0.7, 2.7], calf [-2.1, 0.9]
            "right_thigh_joint": 1.7,   # thigh initial angle
            "right_calf_joint": -1.2,   # calf initial angle
            # left leg: thigh [-2.7, -0.7], calf [-0.9, 2.1]
            "left_thigh_joint": -1.7,   # thigh initial angle
            "left_calf_joint": 1.2,     # calf initial angle
            # wheel initial position
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},  # all joint velocities initialized to 0
    ),

    # soft joint position limit factor
    soft_joint_pos_limit_factor=0.9,

    # actuators: define which joints are controlled by which actuator groups
    actuators={
        # thigh: thigh joints use DC motor model (position control)
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=40.0,       # from URDF effort limit
            saturation_effort=40.0,
            velocity_limit=17.0,     # from URDF velocity limit
            stiffness=80.0,          # PD control stiffness
            damping=4.0,             # PD control damping
            friction=0.0,
        ),
        # calf: calf joints use DC motor model (position control)
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=40.0,
            saturation_effort=40.0,
            velocity_limit=17.0,
            stiffness=80.0,
            damping=4.0,
            friction=0.0,
        ),
        # wheel: wheel joints use implicit actuator (velocity control)
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=40.0,
            velocity_limit_sim=17.0,
            stiffness=0.0,           # velocity control mode, stiffness=0
            damping=2.0,             # damping controls velocity tracking
            friction=0.0,
        ),
    },
)
