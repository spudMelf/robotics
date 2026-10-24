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

        pass

    def reset(self, seed=None, options=None):
        # Reset robot pose and velocities
        super().reset(seed=seed)
        key_id = self.model.key("standing").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
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
        desired_angles = (self.action_scale * action) + current_angles

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


        pass

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




env = QuadrupedEnv("dog.xml")
env.reset()
obs = env._get_obs()

print("Observation:", obs)
print("Shape:", obs.shape)
print("Valid:", env.observation_space.contains(obs))