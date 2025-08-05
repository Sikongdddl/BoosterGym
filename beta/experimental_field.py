import os
from isaacgym import gymapi, gymutil

def load_robot_asset_and_create_actor(gym, sim, env):
    asset_cfg = {
        "file": "/home/ubuntu/jrWork/booster_gym/resources/T1/T1_locomotion.urdf",
        "mujoco_file": "/home/ubuntu/jrWork/booster_gym/resources/T1/T1_locomotion.xml",
        "name": "T1",
        "base_name": "Trunk",
        "foot_names": ["left_foot_link", "right_foot_link"],
        "disable_gravity": False,
        "default_dof_drive_mode": 3,  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        "collapse_fixed_joints": True,  # bug: if collapse_fixed_joints = false, foot doesn't collide with ground
        "fix_base_link": False,  # fix the base of the robot
        "self_collisions": 0,  # 1 to disable, 0 to enable...bitwise filter
        "replace_cylinder_with_capsule": False,  # replace collision cylinders with capsules, leads to faster/more stable simulation
        "flip_visual_attachments": False,  # Some .obj meshes must be flipped from y-up to z-up
        "density": 0.001,
        "angular_damping": 0.0,
        "linear_damping": 0.0,
        "max_angular_velocity": 1000.0,
        "max_linear_velocity": 1000.0,
        "armature": 0.0,
        "thickness": 0.01,
        "feet_edge_pos": [[ 0.1215,  0.05, -0.03],
                          [ 0.1215, -0.05, -0.03],
                          [-0.1015,  0.05, -0.03],
                          [-0.1015, -0.05, -0.03]] # x,y,z [m]
    }

    asset_root = os.path.dirname(asset_cfg["file"])
    asset_file = os.path.basename(asset_cfg["file"])

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
    asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
    asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
    asset_options.fix_base_link = asset_cfg["fix_base_link"]
    asset_options.density = asset_cfg["density"]
    asset_options.angular_damping = asset_cfg["angular_damping"]
    asset_options.linear_damping = asset_cfg["linear_damping"]
    asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
    asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
    asset_options.armature = asset_cfg["armature"]
    asset_options.thickness = asset_cfg["thickness"]
    asset_options.disable_gravity = asset_cfg["disable_gravity"]

    # 加载机器人资产
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # 设置机器人初始位置，稍微抬高避免穿地面
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(1.5, 0, 1)

    # 创建机器人actor
    actor_handle = gym.create_actor(env, robot_asset, start_pose, asset_cfg["name"], 0, asset_cfg["self_collisions"], 0)

    return actor_handle

if __name__ == "__main__":
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments()

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.gravity = gymapi.Vec3(0.0,0.0,-9.81)

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    # 扩大环境范围，保证球门在视野内
    env_lower = gymapi.Vec3(-12, -8, 0)
    env_upper = gymapi.Vec3(12, 8, 5)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # 添加地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # 加载机器人
    robot_actor = load_robot_asset_and_create_actor(gym, sim, env)
    # 创建viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    # 设置摄像机，俯瞰场地，从稍后方和上方看向场地中心
    cam_pos = gymapi.Vec3(0, 15, 20)
    cam_target = gymapi.Vec3(0, 0, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # 模拟渲染循环
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)