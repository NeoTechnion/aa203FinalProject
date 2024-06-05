import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask

from isaacgym import gymutil, gymtorch, gymapi


class Drone(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.epoch_count = 0

        # Observations:
        self.cfg["env"]["numObservations"] = 13  # Only drone states

        # Actions:
        self.cfg["env"]["numActions"] = 4

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dofs_per_env = 6
        
        # Drone has 5 bodies: 1 root, 4 rotors and the marker
        bodies_per_env = 6

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        print("Shape of root_tensor:", self.root_tensor.shape)  # Debug print to determine the shape

        # Adjust the shape here based on the actual shape of root_tensor
        tensor_shape = self.root_tensor.shape[0]
        num_actors_per_env = tensor_shape // self.num_envs

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, num_actors_per_env, 13)

        self.root_states = vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[:, 0:3]
        self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_root_positions[:, 2] = 1
        self.root_quats = self.root_states[:, 3:7]
        self.root_linvels = self.root_states[:, 7:10]
        self.root_angvels = self.root_states[:, 10:13]

        self.marker_states = vec_root_tensor[:, 1, :]
        self.marker_positions = self.marker_states[:, 0:3]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_marker_states = self.marker_states.clone()

        self.thrust_lower_limit = 0
        self.thrust_upper_limit = 2000
        self.thrust_velocity_scale = 2000
        self.thrust_lateral_component = 0.2

        # control tensors
        self.thrusts = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))

        if self.viewer:
            cam_pos = gymapi.Vec3(2.25, 2.25, 3.0)
            cam_target = gymapi.Vec3(3.5, 4.0, 1.9)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # need rigid body states for visualizing thrusts
            self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            print("Shape of rb_state_tensor:", self.rb_state_tensor.shape)  # Debug print to determine the shape
            # Adjust the shape here based on the actual shape of rb_state_tensor
            rb_tensor_shape = self.rb_state_tensor.shape[0]
            num_bodies_per_env = rb_tensor_shape // self.num_envs

            self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(self.num_envs, num_bodies_per_env, 13)
            self.rb_positions = self.rb_states[..., 0:3]
            self.rb_quats = self.rb_states[..., 3:7]

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z

        # Mars gravity
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/Drone.urdf"
        plate_file = "urdf/plate.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 100.0
        asset_options.max_linear_velocity = 100.0
        asset_options.slices_per_cylinder = 40
        drone_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        # Fix the plate to the environment
        asset_options.fix_base_link = True
        plate_asset = self.gym.load_asset(self.sim, asset_root, plate_file, asset_options)

        default_pose = gymapi.Transform()
        default_pose.p.z = 1.0

        self.envs = []
        self.actor_handles = []
        self.plate_handles = []
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            drone_handle = self.gym.create_actor(env, drone_asset, default_pose, "drone", i, 1, 1)

            # Set a random small tilt angle for the target plate
            tilt_angle = np.random.uniform(-0.1, 0.1)  # Small tilt angle in radians
            tilt_direction = np.random.uniform(0, 2 * np.pi)  # Random direction in radians

            quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(np.cos(tilt_direction), np.sin(tilt_direction), 0), tilt_angle)
            default_pose.r = quat

            plate_handle = self.gym.create_actor(env, plate_asset, default_pose, "plate", i, 1, 1)
            self.gym.set_rigid_body_color(env, plate_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
            self.body_drone_props = self.gym.get_actor_rigid_body_properties(env, drone_handle)
            self.actor_handles.append(drone_handle)
            self.plate_handles.append(plate_handle)
            self.envs.append(env)

        self.drone_mass = 0
        for prop in self.body_drone_props:
            self.drone_mass += prop.mass
        print("Total drone mass: ", self.drone_mass)

        if self.debug_viz:
            self.rotor_env_offsets = torch.zeros((self.num_envs, 2, 3), device=self.device)
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        self.target_root_positions[env_ids, 0:2] = (torch.zeros(num_sets, 2, device=self.device) * 10) - 5
        self.target_root_positions[env_ids, 2] = torch.zeros(num_sets, device=self.device) + 1
        self.marker_positions[env_ids] = self.target_root_positions[env_ids]
        self.marker_positions[env_ids, 2] += 0.0
        actor_indices = self.all_actor_indices[env_ids, 1].flatten()
        return actor_indices

    def reset_idx(self, env_ids, reset_plates=False):
        num_resets = len(env_ids)
        actor_indices = self.all_actor_indices[env_ids, 0].flatten()

        # Reset the drone states
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids, 0] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 1] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 2] += torch_rand_float(-0.2, 1.5, (num_resets, 1), self.device).flatten()

        # Optionally reset the plate states at the beginning of an epoch
        if reset_plates:
            plate_quats = torch.zeros((num_resets, 4), dtype=torch.float32, device=self.device)
            for i in range(num_resets):
                tilt_angle = np.random.uniform(-0.1, 0.1)  # Small tilt angle in radians
                tilt_direction = np.random.uniform(0, 2 * np.pi)  # Random direction in radians
                quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(np.cos(tilt_direction), np.sin(tilt_direction), 0), tilt_angle)
                plate_quats[i] = torch.tensor([quat.w, quat.x, quat.y, quat.z], dtype=torch.float32, device=self.device)
            
            self.marker_states[env_ids, 3:7] = plate_quats

        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_indices), len(actor_indices))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        return torch.unique(actor_indices)

    def pre_physics_step(self, _actions):
        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        target_actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(set_target_ids) > 0:
            target_actor_indices = self.set_targets(set_target_ids)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(reset_env_ids) > 0:
            reset_plates = (self.progress_buf[reset_env_ids] == 0).all().item()
            actor_indices = self.reset_idx(reset_env_ids, reset_plates=reset_plates)

        reset_indices = torch.unique(torch.cat([target_actor_indices, actor_indices]))
        if len(reset_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(reset_indices), len(reset_indices))

        actions = _actions.to(self.device)
        thrust_prop_0 = torch.clamp(actions[:, 0] * self.thrust_velocity_scale, self.thrust_lower_limit, self.thrust_upper_limit)
        thrust_prop_1 = torch.clamp(actions[:, 1] * self.thrust_velocity_scale, self.thrust_lower_limit, self.thrust_upper_limit)
        thrust_prop_2 = torch.clamp(actions[:, 2] * self.thrust_velocity_scale, self.thrust_lower_limit, self.thrust_upper_limit)
        thrust_prop_3 = torch.clamp(actions[:, 3] * self.thrust_velocity_scale, self.thrust_lower_limit, self.thrust_upper_limit)

        force_constant = 0.25 * self.drone_mass * 9.82 * torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device_id, requires_grad=False)

        self.forces[:, 1, 2] = self.dt * thrust_prop_0
        self.forces[:, 2, 2] = self.dt * thrust_prop_1
        self.forces[:, 3, 2] = self.dt * thrust_prop_2
        self.forces[:, 4, 2] = self.dt * thrust_prop_3

        self.thrusts[reset_env_ids] = 0.0
        self.forces[reset_env_ids] = 0.0

        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1

        # Check if all environments have completed an episode to reset plates
        if (self.progress_buf >= self.max_episode_length).all():
            self.epoch_count += 1
            reset_plates = True
        else:
            reset_plates = False

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

        if self.viewer and self.debug_viz:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            rotor_indices = torch.LongTensor([2, 4, 6, 8])
            quats = self.rb_quats[:, rotor_indices]
            dirs = -quat_axis(quats.view(self.num_envs * 4, 4), 2).view(self.num_envs, 4, 3)
            starts = self.rb_positions[:, rotor_indices] + self.rotor_env_offsets
            ends = starts + 0.1 * self.thrusts.view(self.num_envs, 4, 1) * dirs

            verts = torch.stack([starts, ends], dim=2).cpu().numpy()
            colors = np.zeros((self.num_envs * 4, 3), dtype=np.float32)
            colors[..., 0] = 1.0
            self.gym.clear_lines(self.viewer)
            self.gym.add_lines(self.viewer, None, self.num_envs * 4, verts, colors)

    def compute_observations(self):
        self.obs_buf[..., 0:3] = (self.target_root_positions - self.root_positions) / 3
        self.obs_buf[..., 3:7] = self.root_quats
        self.obs_buf[..., 7:10] = self.root_linvels / 2
        self.obs_buf[..., 10:13] = self.root_angvels / math.pi
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_drone_reward(
            self.root_positions,
            self.target_root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_drone_reward(root_positions, target_root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length):
    target_dist = torch.sqrt(torch.square(target_root_positions - root_positions).sum(-1))
    pos_reward = 3.0 / (1.0 + target_dist * target_dist)

    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)

    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 8.0, ones, die)
    die = torch.where(root_positions[..., 2] < 0.5, ones, die)
    die = torch.where(ups[..., 2] < 0, ones, die)

    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset

"""
=> saving checkpoint 'runs/Drone_02-18-23-50/nn/Drone.pth'
"""