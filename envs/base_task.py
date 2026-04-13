import sys
import numpy as np
from isaacgym import gymapi, gymutil
import torch
from utils.terrain import Terrain


class BaseTask:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.terrain = Terrain(self.gym, self.sim, self.device, self.cfg["terrain"])

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.set_viewer()
        self.policy_camera = None
        if self.viewer is not None:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "A")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "D")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "W")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "S")

    def _configure_lighting(self):
        viewer_cfg = self.cfg.get("viewer", {})
        light_cfg = viewer_cfg.get("policy_light")
        if not light_cfg:
            return
        color = light_cfg.get("color", [0.7, 0.7, 0.7])
        ambient = light_cfg.get("ambient", [0.35, 0.35, 0.35])
        direction = light_cfg.get("direction", [0.3, 0.2, -1.0])
        light_index = int(light_cfg.get("index", 0))
        self.gym.set_light_parameters(
            self.sim,
            light_index,
            gymapi.Vec3(float(color[0]), float(color[1]), float(color[2])),
            gymapi.Vec3(float(ambient[0]), float(ambient[1]), float(ambient[2])),
            gymapi.Vec3(float(direction[0]), float(direction[1]), float(direction[2])),
        )

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        sim_cfg = self.cfg["sim"]
        sim_device = self.cfg["basic"]["sim_device"]
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(sim_device)

        # env device is GPU only if sim is on GPU, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda":
            self.device = sim_device
        else:
            self.device = "cpu"

        # graphics device for rendering, -1 for no rendering
        self.headless = self.cfg["basic"]["headless"]
        self.graphics_device_id = self.sim_device_id
        viewer_cfg = self.cfg.get("viewer", {})
        need_graphics = bool(viewer_cfg.get("record_video", False)) or bool(
            viewer_cfg.get("capture_for_policy", False)
        )
        if self.headless and not need_graphics:
            self.graphics_device_id = -1

        self.sim_params = gymapi.SimParams()

        # assign general sim parameters
        self.sim_params.dt = sim_cfg["dt"]
        self.sim_params.num_client_threads = sim_cfg.get("num_client_threads", 0)
        self.sim_params.use_gpu_pipeline = sim_device_type == "cuda"
        self.sim_params.substeps = sim_cfg.get("substeps", 2)

        # assign up-axis
        if sim_cfg["up_axis"] == "z":
            self.up_axis_idx = 2
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
        elif sim_cfg["up_axis"] == "y":
            self.up_axis_idx = 1
            self.sim_params.up_axis = gymapi.UP_AXIS_Y
        else:
            raise ValueError(f"Invalid physics up-axis: {sim_cfg['up_axis']}")

        # assign gravity
        self.sim_params.gravity = gymapi.Vec3(*sim_cfg["gravity"])

        # configure physics parameters
        if sim_cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
            # set the parameters
            if "physx" in sim_cfg:
                for opt in sim_cfg["physx"].keys():
                    if opt == "contact_collection":
                        setattr(self.sim_params.physx, opt, gymapi.ContactCollection(sim_cfg["physx"][opt]))
                    else:
                        setattr(self.sim_params.physx, opt, sim_cfg["physx"][opt])
                setattr(self.sim_params.physx, "use_gpu", sim_device_type == "cuda")
        elif sim_cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
            # set the parameters
            if "flex" in sim_cfg:
                for opt in sim_cfg["flex"].keys():
                    setattr(self.sim_params.flex, opt, sim_cfg["flex"][opt])
        else:
            raise ValueError(f"Invalid physics engine backend: {sim_cfg['physics_engine']}")

        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._configure_lighting()

    def set_viewer(self):
        self.viewer = None
        self.camera = None
        if not self.headless:
            # if running with a viewer, set up keyboard shortcuts and camera
            self.enable_viewer_sync = True
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            position = self.cfg["viewer"]["pos"]
            lookat = self.cfg["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(position[0], position[1], position[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def render(self):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

        if self.cfg["viewer"]["record_video"]:
            if self.viewer is None:
                if self.device != "cpu":
                    self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
            if self.camera is None:
                camera_props = gymapi.CameraProperties()
                camera_props.width = 1280
                camera_props.height = 720
                camera_props.use_collision_geometry = False
                self.camera = self.gym.create_camera_sensor(self.envs[self.cfg["viewer"]["record_env_idx"]], camera_props)
                self.camera_frames = []
            cam_pos = gymapi.Vec3(
                *(x + y for x, y in zip(self.root_states[self.cfg["viewer"]["record_env_idx"], 0:3].tolist(), self.cfg["viewer"]["pos"]))
            )
            cam_target = gymapi.Vec3(*self.root_states[self.cfg["viewer"]["record_env_idx"], 0:3].tolist())
            self.gym.set_camera_location(self.camera, self.envs[self.cfg["viewer"]["record_env_idx"]], cam_pos, cam_target)
            self.gym.render_all_camera_sensors(self.sim)
            img = self.gym.get_camera_image(self.sim, self.envs[self.cfg["viewer"]["record_env_idx"]], self.camera, gymapi.IMAGE_COLOR)
            self.camera_frames.append(img.reshape(img.shape[0], -1, 4))

    def capture_policy_frame(
        self,
        root_states: torch.Tensor,
        *,
        env_idx: int = 0,
        follow_actor_index: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> np.ndarray:
        """
        Headless-safe RGB capture for vision / high-level policy input.
        Requires viewer.capture_for_policy: true (or record_video) so graphics_device_id stays enabled.

        Args:
            root_states: Actor root state tensor, shape (num_actors, 13).
            env_idx: Which IsaacGym env handle to use.
            follow_actor_index: Row in root_states used as camera look-at (usually first robot).
            width, height: Camera resolution.

        Returns:
            uint8 array (H, W, 3) RGB.
        """
        if self.graphics_device_id < 0:
            raise RuntimeError(
                "capture_policy_frame needs graphics: set viewer.capture_for_policy: true "
                "(or record_video: true) in YAML when running headless."
            )
        if not hasattr(self, "envs") or not self.envs:
            raise RuntimeError("capture_policy_frame called before envs were created.")

        viewer_cfg = self.cfg.get("viewer", {})
        policy_camera_mode = str(viewer_cfg.get("policy_camera_mode", "follow_actor")).lower()
        cam_offset = viewer_cfg.get("pos", [3.0, -3.0, 2.0])
        if len(cam_offset) < 3:
            raise ValueError("viewer.pos must have length >= 3")

        if self.policy_camera is None:
            cam_props = gymapi.CameraProperties()
            cam_props.width = int(width)
            cam_props.height = int(height)
            cam_props.use_collision_geometry = False
            self.policy_camera = self.gym.create_camera_sensor(self.envs[env_idx], cam_props)

        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)

        if policy_camera_mode == "topdown_center":
            center = viewer_cfg.get("policy_camera_center", [0.0, 0.0, 0.0])
            if len(center) < 3:
                raise ValueError("viewer.policy_camera_center must have length >= 3")
            height_offset = float(viewer_cfg.get("policy_camera_height", 12.0))
            tilt_offset = viewer_cfg.get("policy_camera_tilt_offset", [0.0, -0.5, 0.0])
            if len(tilt_offset) < 3:
                raise ValueError("viewer.policy_camera_tilt_offset must have length >= 3")
            cx = float(center[0])
            cy = float(center[1])
            cz = float(center[2])
            cam_pos = gymapi.Vec3(
                cx + float(tilt_offset[0]),
                cy + float(tilt_offset[1]),
                cz + height_offset + float(tilt_offset[2]),
            )
            cam_target = gymapi.Vec3(cx, cy, cz)
        else:
            base = root_states[follow_actor_index, 0:3]
            bx = float(base[0].item())
            by = float(base[1].item())
            bz = float(base[2].item())
            cam_pos = gymapi.Vec3(bx + float(cam_offset[0]), by + float(cam_offset[1]), bz + float(cam_offset[2]))
            cam_target = gymapi.Vec3(bx, by, bz)

        self.gym.set_camera_location(self.policy_camera, self.envs[env_idx], cam_pos, cam_target)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(
            self.sim, self.envs[env_idx], self.policy_camera, gymapi.IMAGE_COLOR
        )
        arr = np.asarray(img, dtype=np.uint8)
        h, w = int(height), int(width)
        if arr.ndim == 1:
            arr = arr.reshape(h, w, -1)
        elif arr.ndim == 2:
            arr = arr.reshape(h, w, -1)
        if arr.shape[-1] >= 3:
            return np.ascontiguousarray(arr[..., :3])
        raise RuntimeError(f"Unexpected camera image shape: {arr.shape}")
