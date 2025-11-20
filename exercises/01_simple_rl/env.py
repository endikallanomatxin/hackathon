from pathlib import Path
import colorsys
import xml.etree.ElementTree as ET
import genesis as gs
import torch

class Environment:
    def __init__(
            self,
            device: torch.device,
            batch_size,
            max_steps,
            show_viewer=False,
            record=False
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.record = record
        self.debug_env_count = min(4, batch_size)
        self.env_colors = self._generate_pair_colors(self.debug_env_count)

        repo_root = Path(__file__).resolve().parents[1]
        robot_path = repo_root / "assets" / "SO101" / "so101_new_calib.xml"

        gs.init(backend=gs.gs_backend.gpu, performance_mode=True)

        self.scene = gs.Scene(
            sim_options = gs.options.SimOptions(
                dt = 1/60,
            ),
            vis_options = gs.options.VisOptions(
                show_world_frame = True,
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = False,
                plane_reflection = True,
                ambient_light    = (0.1, 0.1, 0.1),
                n_rendered_envs  = min(4, batch_size),
            ),
            renderer=gs.renderers.Rasterizer(),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS = False,
            ),
            show_viewer = show_viewer,
        )

        if self.record:
            # a = np.ceil(np.sqrt(batch_size)/2)
            # The first env of the env grid is at (-a, -a)
            self.cam = self.scene.add_camera(
                res    = (1920, 1080),
                pos    = (2, 2, 1.5),
                lookat = (0, 0, 0.5),
                fov    = 40,
                GUI    = False,
            )

        self.plane = self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=str(robot_path))
        )

        # Construimos los entornos en batch
        self.scene.build(n_envs=batch_size)
        # option env_spacing=(2, 2) to create a grid

        self.jnt_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow_flex',
            'wrist_flex',
            'wrist_roll',
            'gripper',
        ]
        self.dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in self.jnt_names]
        joint_limits = self._load_joint_limits(robot_path)
        lower = [joint_limits[name][0] for name in self.jnt_names]
        upper = [joint_limits[name][1] for name in self.jnt_names]
        self.joint_lower = torch.tensor(lower, device=self.device, dtype=torch.float32)
        self.joint_upper = torch.tensor(upper, device=self.device, dtype=torch.float32)
        self.gripper_link_name = 'gripper_tip'
        self.forearm_link_name = 'lower_arm'

        # Para este ejemplo se asume que todos los entornos avanzan de forma sincronizada.
        self.current_step = 0

        # Observations include joint positions/velocities plus gripper pose/velocity and target
        self.obs_dim = 2*len(self.dofs_idx) + 3 + 3 + 3
        self.act_dim = len(self.dofs_idx)

        if self.record:
            self.cam.start_recording()

    def reset(self):
        # Reiniciamos la posición del robot en todos los entornos
        ctrl_pos = torch.zeros(len(self.dofs_idx))
        self.robot.control_dofs_position(
            torch.tile(ctrl_pos, (self.batch_size, 1)),
            self.dofs_idx
        )

        # Reposicionamos el target de forma aleatoria para cada entorno
        # Se genera un tensor de forma [batch_size, 3] en un rango ([0.1, 0.3], [-0.1, -0.1], [0.0, 0.2])
        new_target_pos = torch.tensor([0.2, -0.15, 0.0]) + torch.rand(self.batch_size, 3) * torch.tensor([0.2, 0.30, 0.30])
        self.target_pos = new_target_pos.to(self.device)
        self.current_step = 0

        # Dejar que la simulación se estabilice
        for _ in range(10):
            self.scene.step()

        if self.record:
            self.cam.start_recording()

        gripper_pos = torch.as_tensor(
            self.robot.get_link(self.gripper_link_name).get_pos(),
            device=self.device,
            dtype=torch.float32,
        )
        self._update_debug_markers(gripper_pos)

        # Devuelve las observaciones iniciales
        return self.get_obs()

    def get_obs(self):
        dof_ang = torch.as_tensor(
            self.robot.get_dofs_position(self.dofs_idx),
            device=self.device,
            dtype=torch.float32,
        )  # Shape [batch_size, n_dofs]
        dof_vel = torch.as_tensor(
            self.robot.get_dofs_velocity(self.dofs_idx),
            device=self.device,
            dtype=torch.float32,
        )  # Shape [batch_size, n_dofs]
        gripper_pos = torch.as_tensor(
            self.robot.get_link(self.gripper_link_name).get_pos(),
            device=self.device,
            dtype=torch.float32,
        )  # Shape [batch_size, 3]
        gripper_vel = torch.as_tensor(
            self.robot.get_link(self.gripper_link_name).get_vel(),
            device=self.device,
            dtype=torch.float32,
        )  # Shape [batch_size, 3]
        target_pos = torch.as_tensor(
            self.target_pos,
            device=self.device,
            dtype=torch.float32,
        )  # Shape [batch_size, 3]

        obs = torch.cat([dof_ang, dof_vel, gripper_pos, gripper_vel, target_pos], dim=1)
        # Genesis can occasionally return NaNs if the simulation for one of the
        # batched environments becomes unstable, so clamp the observations to
        # a safe numeric range before handing them to the policy.
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e4, neginf=-1e4)
        return obs

    def compute_reward(self):
        reward_dict = {}

        # MAIN PENALTY: distance between gripper and target
        # La recompensa es la distancia negativa entre el gripper y el target, para cada entorno
        gripper_pos = torch.as_tensor(self.robot.get_link(self.gripper_link_name).get_pos(), device=self.device)
        target_pos = torch.as_tensor(self.target_pos, device=self.device)
        distance = target_pos - gripper_pos
        distance_magnitude = torch.linalg.vector_norm(distance, dim=-1)
        reward_dict['distance'] = torch.mean(distance_magnitude).clone().detach().cpu()
        reward_dict['distance_reward'] = -distance_magnitude

        # REWARD: Moving towards the target
        # Cosine similarity between the gripper-target vector and the gripper velocity
        gripper_velocity = torch.as_tensor(self.robot.get_link(self.gripper_link_name).get_vel(), device=self.device)
        gripper_velocity_mag = torch.linalg.vector_norm(gripper_velocity, dim=-1)
        direction_similarity = torch.nn.functional.cosine_similarity(
            gripper_velocity,
            distance,
            dim=-1,
            eps=1e-6,
        )
        reward_dict['direction_similarity'] = torch.mean(direction_similarity).clone().detach().cpu()
        reward_dict['direction_similarity_reward'] = (
                6.0
                * direction_similarity
                * (gripper_velocity_mag.clone().detach())**2  # Make it important only if moving
                * (distance_magnitude.clone().detach())**2 # Make it important only if far away
        )

        # PENALTY: contact_force_sum
        # Obtenemos la información de contactos de la simulación.
        contact_info = self.robot.get_contacts()
        # contact_info es un diccionario con claves: 'geom_a', 'geom_b', 'link_a', 'link_b',
        # 'position', 'force_a', 'force_b', 'valid_mask'
        # que tiene dimensiones [n_envs, n_contacts, ...]
        # Convertimos la fuerza a tensor. Se asume que tiene shape [n_envs, n_contacts, 3]
        force_a = torch.as_tensor(contact_info['force_a'], dtype=torch.float32, device=self.device)
        # Calculamos la magnitud de la fuerza para cada contacto.
        force_magnitudes = torch.linalg.vector_norm(force_a, dim=-1)  # [n_envs, n_contacts]
        # Sumamos las fuerzas de contacto en cada entorno y luego promediamos.
        contact_force_sum = force_magnitudes.sum(dim=1)  # [n_envs]
        reward_dict['contact_force_sum'] = torch.mean(contact_force_sum).clone().detach().cpu()
        reward_dict['contact_force_sum_reward'] = -0.1*contact_force_sum

        # PENALTY: links_contact_force
        links_contact_force = torch.as_tensor(self.robot.get_links_net_contact_force(), device=self.device)  # Returns a tensor of shape [batch_size, n_links, 3]
        # Compute per-link magnitudes before averaging so opposing forces do not cancel out.
        link_force_magnitudes = torch.linalg.vector_norm(links_contact_force, dim=-1)  # [batch_size, n_links]
        links_contact_force = torch.mean(link_force_magnitudes, dim=1)  # [batch_size]
        reward_dict['links_force_sum'] = torch.mean(links_contact_force).clone().detach().cpu()
        reward_dict['links_force_sum_reward'] = -4*links_contact_force

        # PENALTY: gripper_velocity
        reward_dict['gripper_velocity'] = torch.mean(gripper_velocity_mag).clone().detach().cpu()
        reward_dict['gripper_velocity_reward'] = -0.08*gripper_velocity_mag

        # PENALTY: gripper_angular_velocity
        gripper_angular_velocity = torch.as_tensor(self.robot.get_link(self.gripper_link_name).get_ang(), device=self.device)
        gripper_angular_velocity = torch.linalg.vector_norm(gripper_angular_velocity, dim=-1)
        reward_dict['gripper_angular_velocity'] = torch.mean(gripper_angular_velocity).clone().detach().cpu()
        reward_dict['gripper_angular_velocity_reward'] = -0.010*gripper_angular_velocity

        # PENALTY: joint_velocity_sq
        joint_velocity = torch.as_tensor(self.robot.get_dofs_velocity(self.dofs_idx), device=self.device)
        joint_velocity = torch.linalg.vector_norm(joint_velocity, dim=-1)
        joint_velocity_sq = joint_velocity**2
        reward_dict['joint_velocity_sq'] = torch.mean(joint_velocity_sq).clone().detach().cpu()
        reward_dict['joint_velocity_sq_reward'] = -0.0016*joint_velocity_sq

        # REWARD: forearm_height  (to have it always approach the target from above)
        forearm_pos = torch.as_tensor(
            self.robot.get_link(self.forearm_link_name).get_pos(),
            device=self.device
        )
        # Get the height of the forearm link
        forearm_height = forearm_pos[:, 2]
        reward_dict['forearm_height'] = torch.mean(forearm_height).clone().detach().cpu()
        reward_dict['forearm_height_reward'] = 0.02*forearm_height

        # COMBINED REWARD
        reward = torch.zeros(self.batch_size, device=self.device)
        for key in reward_dict:
            if 'reward' in key:
                reward += reward_dict[key].clone()
                reward_dict[key] = reward_dict[key].mean().clone().detach().cpu().item()

        self._update_debug_markers(gripper_pos)
        return reward, reward_dict

    def step(self, actions, record=False):
        """
        actions: tensor de forma [batch_size, act_dim]
        Aplica las acciones a cada entorno, avanza la simulación y devuelve (obs, reward, done, info)
        """
        # Se asume que control_dofs_position admite batch actions directamente
        angles = actions_to_angles(actions, self.joint_lower, self.joint_upper)
        self.robot.control_dofs_position(angles, self.dofs_idx)
        self.scene.step()
        self.current_step += 1

        obs = self.get_obs()
        reward, reward_dict = self.compute_reward()

        if self.record and record:
            self.cam.render()

        return obs, reward, reward_dict

    def save_video(self, filename):
        if not self.record:
            gs.logger.warning("Video recording requested but record=False for this environment")
            return
        self.cam.stop_recording(save_to_filename=filename, fps=30)

    def _generate_pair_colors(self, count):
        if count <= 0:
            return []
        colors = []
        golden_ratio = 0.61803398875  # Spread hues evenly
        for idx in range(count):
            hue = (idx * golden_ratio) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
            colors.append(rgb)
        return colors

    def _update_debug_markers(self, gripper_pos: torch.Tensor):
        if self.scene is None or not hasattr(self, 'target_pos'):
            return
        debug_envs = min(self.debug_env_count, self.batch_size)
        if debug_envs <= 0:
            return
        self.scene.clear_debug_objects()
        for idx in range(debug_envs):
            color = tuple(self.env_colors[idx % len(self.env_colors)])
            target = self.target_pos[idx].detach().cpu().tolist()
            self.scene.draw_debug_sphere(target, radius=0.01, color=color)
            grip = gripper_pos[idx].detach().cpu().tolist()
            # Slightly larger radius so it looks attached to the robot
            self.scene.draw_debug_sphere(grip, radius=0.015, color=color)

    def _load_joint_limits(self, robot_path: Path):
        tree = ET.parse(robot_path)
        root = tree.getroot()
        limits = {}
        for joint in root.findall(".//joint"):
            name = joint.attrib.get('name')
            rng = joint.attrib.get('range')
            if not name or not rng:
                continue
            lo, hi = map(float, rng.split())
            limits[name] = (lo, hi)
        missing = [name for name in self.jnt_names if name not in limits]
        if missing:
            raise ValueError(f"Missing joint limits for: {missing}")
        return limits


def actions_to_angles(actions: torch.Tensor, joint_lower: torch.Tensor, joint_upper: torch.Tensor):
    """
    Map normalized policy outputs in [-1, 1] to the actual joint limits so each
    actuator can use its full physical range.
    """
    normalized = torch.clamp(actions, -1.0, 1.0).to(dtype=torch.float32)
    lower = joint_lower.to(device=normalized.device, dtype=normalized.dtype)
    upper = joint_upper.to(device=normalized.device, dtype=normalized.dtype)
    return lower + 0.5 * (normalized + 1.0) * (upper - lower)
