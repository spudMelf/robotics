import gymnasium as gym
import robot_structure as rs
import numpy as np
import mujoco

class QuadrupedEnv(gym.Env):

    def __init__(self, model_xml):
        super().__init__()
        # Load our MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)

        # PD Controller Constants
        self.kp = np.array([5.0, 5.0, 5.0])
        self.kd = np.array([0.15, 0.15, 0.15])

        # Reward function constants
        self.w_angles = 0.25
        self.k_angles = 10

        self.w_tilt = 0.55
        self.k_tilt = 5

        self.k_v = 1
        self.k_w = 1
        self.w_still = 0.1

        self.k_joint = 1
        self.w_joint = 0.1

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(34,),
            dtype=np.float32
        )

        self.action_scale = 0.25

        self.joint_names = np.array(rs.ALL_JOINTS).flatten()
        self.motor_names = np.array(rs.ALL_MOTORS).flatten()

        self.standing_angles = np.array([
            rs.STANDING_ANGLES[name]
            for name in self.joint_names
        ])

    def reset(self, seed=None, options=None):
        # Reset robot pose and velocities
        super().reset(seed=seed)
        key_id = self.model.key("standing").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        for name in self.joint_names:
            joint_id = self.model.joint(name).id
            addr = self.model.jnt_qposadr[joint_id]

            self.data.qpos[addr] += self.np_random.uniform(
                -0.05, 0.05
            )
        mujoco.mj_forward(self.model, self.data)

        # Return initial observation and info
        return self._get_obs(), {}

    def step(self, action):
        # Apply motor commands using PD control
        # Step MuJoCo physics
        # Measure foot position and chassis state
        # Calculate reward
        # Check for termination

        # Apply motor command and step physics
        current_angles = np.array([
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]]
            for name in self.joint_names
        ])
        # theta_desired = theta_standing + scale * action
        desired_angles = self.standing_angles + self.action_scale * action

        for _ in range(20):
            for i in range(len(self.joint_names)):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names[i])

                # get current angle position
                qpos_address = self.model.jnt_qposadr[joint_id]
                angle = self.data.qpos[qpos_address]

                # get current angular velocity
                qvel_address = self.model.jnt_dofadr[joint_id]
                vel = self.data.qvel[qvel_address]

                # our desired velocity is 0, because we are trying to reach that point and stop
                torque = (self.kp[i % 3] * (desired_angles[i] - angle)) - (self.kd[i % 3] * vel)

                self.data.actuator(self.motor_names[i]).ctrl[0] = torque
            mujoco.mj_step(self.model, self.data)

        # Measure foot position and chassis state
        FL_pos = self.data.site("front_left_foot").xpos.copy()
        FR_pos = self.data.site("front_right_foot").xpos.copy()
        RL_pos = self.data.site("rear_left_foot").xpos.copy()
        RR_pos = self.data.site("rear_right_foot").xpos.copy()

        chassis_pos = self.data.body("chassis").xpos.copy()
        chassis_orientation = self.data.body("chassis").xquat.copy()

        chassis_id = self.model.body("chassis").id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,chassis_id, velocity, 0)

        chassis_angular_velocity = velocity[:3]
        chassis_linear_velocity = velocity[3:]

        # calculate reward
        # angles
        joint_angles = np.array([
            self.data.qpos[
                self.model.jnt_qposadr[self.model.joint(name).id]
            ]
            for name in self.joint_names
        ])
        joints_rmse = np.sqrt(np.mean(
            (joint_angles - self.standing_angles)**2
        ))
        reward_pose = np.exp(-self.k_angles * joints_rmse * joints_rmse)

        # pose
        w, x, y, z = chassis_orientation
        tilt_error = x**2 + y**2
        reward_upright = np.exp(-self.k_tilt * tilt_error)

        # velocities
        v_x, v_y, v_z = chassis_linear_velocity
        w_x, w_y, w_z = chassis_angular_velocity
        v_error = np.sqrt((v_x * v_x) + (v_y * v_y) + (v_z * v_z))
        w_error = np.sqrt((w_x * w_x) + (w_y * w_y) + (w_z * w_z))
        reward_still = np.exp(-(self.k_v * v_error + self.k_w * w_error))

        # Joint stillness: penalize RMS velocity across all 12 actuated joints.
        joint_velocities = np.array([
            self.data.qvel[
                self.model.jnt_dofadr[self.model.joint(name).id]
            ]
            for name in self.joint_names
        ])
        joint_velocity_rms = np.sqrt(np.mean(joint_velocities**2))
        reward_joint_still = np.exp(
            -self.k_joint * joint_velocity_rms**2
        )

        reward = (
            self.w_angles * reward_pose
            + self.w_tilt * reward_upright
            + self.w_still * reward_still
            + self.w_joint * reward_joint_still
        )

        # temporary
        floor_z = self.data.geom("floor").xpos[2]
        chassis_height = chassis_pos[2] - floor_z

        # End episode if robot falls or tilts excessively
        terminated = bool(
            chassis_height < 0.08
            or tilt_error > np.sin(np.deg2rad(22.5))**2
        )

        truncated = False
        info = {
            "joint_rmse": joints_rmse,
            "tilt_error": tilt_error,
            "pose_reward": reward_pose,
            "upright_reward": reward_upright,
            "linear_speed": v_error,
            "angular_speed": w_error,
            "still_reward": reward_still,
            "joint_velocity_rms": joint_velocity_rms,
            "joint_still_reward": reward_joint_still,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs = []
        for joint in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            qpos_address = self.model.jnt_qposadr[joint_id]
            joint_angle = self.data.qpos[qpos_address]
            obs.append(joint_angle)

        for joint in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            qvel_address = self.model.jnt_dofadr[joint_id]
            joint_vel = self.data.qvel[qvel_address]
            obs.append(joint_vel)

        obs = np.array(obs)
        obs = np.concatenate((obs, self.data.body("chassis").xquat))
        obs = np.concatenate((obs, self.data.joint("chassis_free").qvel))

        return np.array(obs, dtype=np.float32)
