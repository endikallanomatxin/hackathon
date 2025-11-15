from pathlib import Path
import genesis as gs
import torch

class Environment:
    def __init__(self,
                 device: torch.device,
                 batch_size=4,
                 max_steps=100,
                 record=False):
        self.device = device
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.record = record

        repo_root = Path(__file__).resolve().parents[2]
        robot_path = repo_root / "assets" / "SO101" / "so101_new_calib.xml"

        gs.init(backend=gs.gs_backend.gpu)

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
            show_viewer = False,
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
        self.gripper_link_name = 'gripper'
        self.forearm_link_name = 'lower_arm'

        # Para este ejemplo se asume que todos los entornos avanzan de forma sincronizada.
        self.current_step = 0

        self.obs_dim = 2*len(self.dofs_idx) + 2*3
        self.act_dim = 2*len(self.dofs_idx)

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
        # Se genera un tensor de forma [batch_size, 3] en un rango ([0.2, 0.8], [0.2, 0.8], [0.2, 0.8])
        new_target_pos = torch.rand(self.batch_size, 3) * torch.tensor([0.6, 0.6, 0.6]) + torch.tensor([0.2, 0.2, 0.2])
        self.target_pos = new_target_pos.to(self.device)
        # Draw the target of the first 10 environments
        self.scene.clear_debug_objects()
        for i in range(min(10, self.batch_size)):
            self.scene.draw_debug_sphere(new_target_pos[i], radius=0.05, color=(1, 0, 0))

        self.current_step = 0

        self.previous_actions = torch.zeros(self.batch_size, self.act_dim, device=self.device)

        # Dejar que la simulación se estabilice
        for _ in range(10):
            self.scene.step()

        if self.record:
            self.cam.start_recording()

        # Devuelve las observaciones iniciales
        return self.get_obs()

    def get_obs(self):
        dof_ang = torch.as_tensor(
            self.robot.get_dofs_position(self.dofs_idx),
            device=self.device,
            dtype=torch.float32,
        )  # Shape [batch_size, n_dofs]
        dof_cos = torch.cos(dof_ang)
        dof_sin = torch.sin(dof_ang)
        gripper_pos = torch.as_tensor(
            self.robot.get_link(self.gripper_link_name).get_pos(),
            device=self.device,
            dtype=torch.float32,
        )  # Shape [batch_size, 3]
        target_pos = torch.as_tensor(
            self.target_pos,
            device=self.device,
            dtype=torch.float32,
        )  # Shape [batch_size, 3]

        obs = torch.cat([dof_cos, dof_sin, gripper_pos, target_pos], dim=1)  # Shape [batch_size, obs_dim=23]
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
        reward_dict['direction_similarity_reward'] = 1*direction_similarity*(gripper_velocity_mag.clone().detach())**2  # Make it important only if moving

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
        reward_dict['contact_force_sum_reward'] = -0.0002*contact_force_sum

        # PENALTY: links_contact_force
        links_contact_force = torch.as_tensor(self.robot.get_links_net_contact_force(), device=self.device)  # Returns a tensor of shape [batch_size, n_links, 3]
        # Average for each link
        links_contact_force = torch.mean(links_contact_force, dim=1)  # [batch_size, 3]
        # Average for each environment
        links_contact_force = torch.linalg.vector_norm(links_contact_force, dim=1)  # [batch_size]
        reward_dict['links_force_sum'] = torch.mean(links_contact_force).clone().detach().cpu()
        reward_dict['links_force_sum_reward'] = -0.003*links_contact_force

        # PENALTY: gripper_velocity
        reward_dict['gripper_velocity'] = torch.mean(gripper_velocity_mag).clone().detach().cpu()
        reward_dict['gripper_velocity_reward'] = -0.04*gripper_velocity_mag

        # PENALTY: gripper_angular_velocity
        gripper_angular_velocity = torch.as_tensor(self.robot.get_link(self.gripper_link_name).get_ang(), device=self.device)
        gripper_angular_velocity = torch.linalg.vector_norm(gripper_angular_velocity, dim=-1)
        reward_dict['gripper_angular_velocity'] = torch.mean(gripper_angular_velocity).clone().detach().cpu()
        reward_dict['gripper_angular_velocity_reward'] = -0.04*gripper_angular_velocity

        # PENALTY: joint_velocity_sq
        joint_velocity = torch.as_tensor(self.robot.get_dofs_velocity(self.dofs_idx), device=self.device)
        joint_velocity = torch.linalg.vector_norm(joint_velocity, dim=-1)
        joint_velocity_sq = joint_velocity**2
        reward_dict['joint_velocity_sq'] = torch.mean(joint_velocity_sq).clone().detach().cpu()
        reward_dict['joint_velocity_sq_reward'] = -0.0009*joint_velocity_sq

        # REWARD: forearm_height  (to have it always approach the target from above)
        forearm_pos = torch.as_tensor(
            self.robot.get_link(self.forearm_link_name).get_pos(),
            device=self.device
        )
        # Get the height of the forearm link
        forearm_height = forearm_pos[:, 2]
        reward_dict['forearm_height'] = torch.mean(forearm_height).clone().detach().cpu()
        reward_dict['forearm_height_reward'] = 0.4*forearm_height

        # PENALTY: low gripper_height (penalizar que esté muy abajo)
        gripper_height = gripper_pos[:, 2]
        reward_dict['gripper_height'] = torch.mean(gripper_height).clone().detach().cpu()
        gripper_height_denom = gripper_height + 0.1
        # Replace near-zero denominators but keep the physical sign to avoid NaNs.
        sign = torch.sign(gripper_height_denom)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        safe_denom = torch.where(
            torch.abs(gripper_height_denom) < 1e-6,
            sign * 1e-6,
            gripper_height_denom,
        )
        reward_dict['gripper_height_reward'] = -0.1 / safe_denom

        # COMBINED REWARD
        reward = torch.zeros(self.batch_size, device=self.device)
        for key in reward_dict:
            if 'reward' in key:
                reward += reward_dict[key].clone()
                reward_dict[key] = reward_dict[key].mean().clone().detach().cpu().item()

        return reward, reward_dict

    def step(self, actions, record=False):
        """
        actions: tensor de forma [batch_size, act_dim]
        Aplica las acciones a cada entorno, avanza la simulación y devuelve (obs, reward, done, info)
        """
        # Se asume que control_dofs_position admite batch actions directamente
        angles = actions_to_angles(actions)
        self.robot.control_dofs_position(angles, self.dofs_idx)
        self.scene.step()
        self.current_step += 1

        obs = self.get_obs()
        reward, reward_dict = self.compute_reward()

        # Compare actions with last actions
        if torch.allclose(actions, self.previous_actions):
            self.previous_actions = actions

        if self.record and record:
            self.cam.render()

        return obs, reward, reward_dict

    def save_video(self, filename):
        if not self.record:
            gs.logger.warning("Video recording requested but record=False for this environment")
            return
        self.cam.stop_recording(save_to_filename=filename, fps=30)

def actions_to_angles(actions: torch.Tensor):
    """
    actions contains the trigonometric functions of the angles
    this function converts them to angles between -pi and pi
    """
    n_dofs = actions.shape[-1] // 2
    cos_components = actions[..., :n_dofs]
    sin_components = actions[..., n_dofs:]
    return torch.atan2(sin_components, cos_components)
