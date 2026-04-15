import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces

class Env3d(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(self, render_mode=None, xml_path='four_legged_bot.xml'):
        super().__init__()

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self.max_steps = 20000 

        self.last_x_pos = 0.0
        self.last_action = np.zeros(self.model.nu)

        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.torso_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
        
        leg_joint_names = [
            "front_left_hip_joint", "front_left_knee_joint",
            "front_right_hip_joint", "front_right_knee_joint",
            "back_left_hip_joint", "back_left_knee_joint",
            "back_right_hip_joint", "back_right_knee_joint"
        ]

        leg_end_geom_names = [
            "front_left_knee_geom", "front_right_knee_geom",
            "back_left_knee_geom", "back_right_knee_geom"
        ]
        self.leg_end_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in leg_end_geom_names]

        self.leg_joint_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in leg_joint_names
        ]
        self.leg_qvel_indices = [self.model.jnt_dofadr[i] for i in self.leg_joint_indices]

        n_qpos = self.model.nq
        n_qvel = self.model.nv
        obs_high = np.inf * np.ones(n_qpos + n_qvel)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count=0
        mujoco.mj_resetData(self.model, self.data)
        
        self.last_x_pos = self.data.geom_xpos[self.torso_geom_id][0]
        self.last_action = np.zeros(self.model.nu)

        observation = self._get_obs()
        if self.render_mode == "human":
            self._render_frame()
        return observation, {}

    def step(self, action):
        self.step_count+=1
        
        smoothing_factor = 0.9
        smoothed_action = smoothing_factor * self.last_action + (1 - smoothing_factor) * np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = smoothed_action
        self.last_action = smoothed_action

        mujoco.mj_step(self.model, self.data)
        observation = self._get_obs()
        torso_pos = self.data.geom_xpos[self.torso_geom_id]
        torso_linvel = self.data.cvel[self.torso_id][3:]
        torso_angvel = self.data.cvel[self.torso_id][:3]

        # --- REBALANCED REWARD FUNCTION ---
        
        # 1. PRIMARY OBJECTIVE: Forward Progress
        current_x_pos = torso_pos[0]
        # Clip delta_x to prevent a single massive lunge from being too rewarding
        delta_x = np.clip(current_x_pos - self.last_x_pos, -0.01, 0.01)
        self.last_x_pos = current_x_pos
        forward_reward = 1500 * delta_x 
        # forward_reward = 0
        
        # 2. SECONDARY OBJECTIVES: Small bonuses for good form
        height_reward = 0.1 * np.exp(-50 * (torso_pos[2] - 0.17)**2)
        torso_rot_matrix = self.data.xmat[self.torso_id].reshape(3, 3)
        torso_up_vector = torso_rot_matrix[:, 2]
        upright_reward = 0.1 * (np.dot(torso_up_vector, [0, 0, 1]))**2
        torso_forward_vector = torso_rot_matrix[:, 0]
        level_torso_reward = 0.1 * np.exp(-25 * (torso_forward_vector[2])**2)

        legs_in_contact = np.zeros(len(self.leg_end_geom_ids), dtype=bool)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            for leg_idx, leg_geom_id in enumerate(self.leg_end_geom_ids):
                if (contact.geom1 == leg_geom_id and contact.geom2 == self.ground_id) or \
                   (contact.geom2 == leg_geom_id and contact.geom1 == self.ground_id):
                    legs_in_contact[leg_idx] = True
                    break
        
        leg_end_heights = self.data.geom_xpos[self.leg_end_geom_ids][:, 2]
        lifted_leg_heights = leg_end_heights[~legs_in_contact]
        
        leg_lift_reward = 0.0
        if lifted_leg_heights.size > 0:
            leg_lift_reward = 0.1 * np.max(lifted_leg_heights)
        
        # 3. PENALTIES: Discourage undesirable behavior
        roll_pitch_penalty = 0.3 * (torso_angvel[0]**2 + torso_angvel[1]**2)
        control_penalty = 0.01 * np.sum(np.square(self.data.ctrl))
        vertical_velocity_penalty = 0.2 * abs(torso_linvel[2])
        sideways_velocity_penalty = 0.5 * abs(torso_linvel[1])
        joint_velocity_penalty = 0.01 * np.sum(np.square(self.data.qvel[self.leg_qvel_indices]))
        
        num_legs_in_contact = np.sum(legs_in_contact)
        fil_penalty = 0.0
        if num_legs_in_contact > 3:
            fil_penalty = 0.067

        stillness_penalty = 0.12 * np.exp(-1000 * torso_linvel[0]**2)

        # 4. CATASTROPHIC PENALTY: End the episode on critical failure
        collision_penalty_value = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == self.torso_geom_id and contact.geom2 == self.ground_id) or \
               (contact.geom2 == self.torso_geom_id and contact.geom1 == self.ground_id):
                collision_penalty_value = 1.0
                break

        # Combine all terms
        reward = (
            forward_reward +
            leg_lift_reward +
            height_reward +
            upright_reward +
            level_torso_reward -
            roll_pitch_penalty -
            vertical_velocity_penalty -
            sideways_velocity_penalty -
            joint_velocity_penalty -
            control_penalty -
            stillness_penalty -
            fil_penalty -
            (100 * collision_penalty_value)
        )

        # penalties = {
        #     "roll_pitch_penalty": roll_pitch_penalty,
        #     "vertical_velocity_penalty": vertical_velocity_penalty,
        #     "sideways_velocity_penalty": sideways_velocity_penalty,
        #     "joint_velocity_penalty": joint_velocity_penalty,
        #     "control_penalty": control_penalty,
        #     "stillness_penalty": stillness_penalty,
        #     "fil_penalty": fil_penalty
        # }

        # max_name = max(penalties, key=penalties.get)  # name of the largest
        # max_value = penalties[max_name]               # value of the largest

        # if(self.step_count%5==0):
        #     print(f"{sum(penalties.values())}  {reward + sum(penalties.values()) - forward_reward}")
        
        terminated = False
        if collision_penalty_value > 0 or torso_pos[2] < 0.1:
            terminated = True
        
        truncated = self.step_count > self.max_steps
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def _render_frame(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self.viewer.lock():
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
                self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
