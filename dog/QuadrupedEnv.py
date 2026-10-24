import os
import sys
from dataclasses import dataclass, field
import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer

# chopped import
try:
    import robot_structure as rs
except ImportError:
    from dog import robot_structure as rs


@dataclass
class QuadrupedConfig:
    """standardized config and hyperparams for QuadrupedEnv"""

    # sim, control vars
    substeps: int = 20
    torque_limit: float = 1.0
    action_scale: float = 0.25
    kp: np.ndarray = field(
        default_factory=lambda: np.array(
            [5.0, 5.0, 5.0], 
            dtype=np.float64)
    )
    kd: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.15, 0.15, 0.15], 
            dtype=np.float64)
    )

    # task params
    target_vel: float = 0.5
    max_episode_steps: int = 1000
    gravity: float = 9.81

    # termination thresholds
    min_clearance: float = 0.08
    min_upright_z: float = 0.5
    fall_penalty: float = 5.0

    # reward weighting
    rew_vel_scale: float = 4.0
    rew_alive: float = 0.5
    pen_upright: float = 2.0
    pen_lateral_vel: float = 0.3
    pen_yaw_vel: float = 0.1
    pen_elec_loss: float = 0.002
    pen_mech_work: float = 0.002
    pen_action_smooth: float = 0.05


class QuadrupedEnv(gym.Env):
    """
    gymnasium env for 12-dim of freedom (dof) quad-leg robot dog,
    MuJoCo simulated.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        model_xml=None,
        render_mode=None,
        config: QuadrupedConfig = None,
    ):
        super().__init__()
        self.cfg = config if config is not None else QuadrupedConfig()

        # model pathing
        if model_xml is None:
            model_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dog.xml")
        elif not os.path.isabs(model_xml) and not os.path.exists(model_xml):
            model_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_xml)
        self.model_xml = model_xml

        self.render_mode = render_mode
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(self.model_xml)
        self.data = mujoco.MjData(self.model)

        # ids and addresses for chassis and floor
        self.joint_names = np.array(rs.ALL_JOINTS).flatten()
        self.motor_names = np.array(rs.ALL_MOTORS).flatten()
        self.num_actions = len(self.motor_names)

        self.joint_qpos_addrs = [
            self.model.jnt_qposadr[self.model.joint(name).id] for name in self.joint_names
        ]
        self.joint_dof_addrs = [
            self.model.jnt_dofadr[self.model.joint(name).id] for name in self.joint_names
        ]
        self.motor_actuator_ids = [
            self.model.actuator(name).id for name in self.motor_names
        ]

        self.chassis_body_id = self.model.body("chassis").id
        self.chassis_free_id = self.model.joint("chassis_free").id
        self.chassis_qpos_addr = self.model.jnt_qposadr[self.chassis_free_id]
        self.chassis_qvel_addr = self.model.jnt_dofadr[self.chassis_free_id]
        self.chassis_geom_id = self.model.geom("chassis_geom").id
        self.floor_geom_id = self.model.geom("floor").id

        # contact points for ground
        self.foot_sites = [
            "front_left_foot",
            "front_right_foot",
            "rear_left_foot",
            "rear_right_foot",
        ]

        # simulation run counters and tracking
        self.step_count = 0
        self.total_energy = 0.0
        self.initial_x = 0.0
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)

        # physics props
        self.robot_mass = float(np.sum(self.model.body_mass)) #TODO: refine within dog.xml closer to our real-world components
        self.floor_z = float(self.model.geom("floor").pos[2])
        self.nominal_angles = np.array(
            [self.model.key("standing").qpos[addr] for addr in self.joint_qpos_addrs],
            dtype=np.float64,
        )

        # bounding boxes around the desired angles 
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(34,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        """resets sim to standing pose"""

        # reset stats
        super().reset(seed=seed)
        self.step_count = 0
        self.total_energy = 0.0

        # reset robot to standing keyframe
        key_id = self.model.key("standing").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # forward kinematics to apply update to body
        mujoco.mj_forward(self.model, self.data)

        # reset initial pos
        self.initial_x = float(self.data.body("chassis").xpos[0])
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        """
        per control step:
        1. translates action offsets into desired joint angles around nominal pose.
        2. advances physics using PD torque control.
        3. computes reward and energy/efficiency metrics.
        4. checks termination+truncation conditions.
        """
        self.step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        desired_angles = self.nominal_angles + (self.cfg.action_scale * action)

        elec_loss = 0.0
        mech_work = 0.0

        # physics loop substep times
        for _ in range(self.cfg.substeps):
            for i in range(self.num_actions):
                q = self.data.qpos[self.joint_qpos_addrs[i]]
                dq = self.data.qvel[self.joint_dof_addrs[i]]

                # pd control law: tau = Kp*(q* - q) - Kd*dq
                gain_idx = i % 3
                torque = (self.cfg.kp[gain_idx] * (desired_angles[i] - q)) - (
                    self.cfg.kd[gain_idx] * dq
                )
                torque = float(
                    np.clip(torque, -self.cfg.torque_limit, self.cfg.torque_limit)
                )

                self.data.ctrl[self.motor_actuator_ids[i]] = torque

                elec_loss += (torque ** 2) * self.model.opt.timestep
                mech_work += abs(torque * dq) * self.model.opt.timestep

            mujoco.mj_step(self.model, self.data)

        self.total_energy += mech_work

        # update kinematics and chassis state
        chassis_xpos = self.data.body("chassis").xpos
        forward_vel = float(self.data.qvel[self.chassis_qvel_addr + 0])
        lateral_vel = float(self.data.qvel[self.chassis_qvel_addr + 1])
        yaw_vel = float(self.data.qvel[self.chassis_qvel_addr + 5])

        # clearance above ground
        height = float(chassis_xpos[2]) - self.floor_z
        # chassis vertical orientation from rotation matrix
        up_z = float(self.data.body("chassis").xmat[8])

        # cost of transport (CoT) telemetry
        distance_traveled = max(1e-4, float(chassis_xpos[0]) - self.initial_x)
        cost_of_transport = self.total_energy / (
            self.robot_mass * self.cfg.gravity * distance_traveled
        )

        # reward calc logic
        reward_forward = float(
            np.exp(-self.cfg.rew_vel_scale * ((forward_vel - self.cfg.target_vel) ** 2))
        )
        reward_alive = self.cfg.rew_alive
        penalty_upright = -self.cfg.pen_upright * max(0.0, 1.0 - up_z)
        penalty_lateral = (
            -self.cfg.pen_lateral_vel * abs(lateral_vel)
            - self.cfg.pen_yaw_vel * abs(yaw_vel)
        )
        penalty_energy = (
            -self.cfg.pen_elec_loss * elec_loss
            - self.cfg.pen_mech_work * mech_work
        )
        penalty_smooth = -self.cfg.pen_action_smooth * float(
            np.sum((action - self.prev_action) ** 2)
        )
        reward = (
            reward_forward
            + reward_alive
            + penalty_upright
            + penalty_lateral
            + penalty_energy
            + penalty_smooth
        )

        # update last action
        self.prev_action = action.copy()

        # termination checks
        fallen = (height < self.cfg.min_clearance) or (up_z < self.cfg.min_upright_z)
        chassis_collision = self._check_chassis_collision()
        terminated = bool(fallen or chassis_collision)
        if terminated:
            reward -= self.cfg.fall_penalty
        truncated = bool(self.step_count >= self.cfg.max_episode_steps)

        # telemetry
        info = {
            "forward_vel": forward_vel,
            "lateral_vel": lateral_vel,
            "height": height,
            "up_z": up_z,
            "cost_of_transport": float(cost_of_transport),
            "step_energy": float(mech_work),
            "total_energy": float(self.total_energy),
            "step_count": self.step_count,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        """ builds observation vector """
        obs = []

        for addr in self.joint_qpos_addrs:
            obs.append(self.data.qpos[addr])
        for addr in self.joint_dof_addrs:
            obs.append(self.data.qvel[addr])

        obs = np.array(obs, dtype=np.float32)

        # orientation
        chassis_quat = np.array(self.data.body("chassis").xquat, dtype=np.float32)
        # velocity
        chassis_vel = np.array(
            self.data.qvel[self.chassis_qvel_addr : self.chassis_qvel_addr + 6],
            dtype=np.float32,
        )

        return np.concatenate([obs, chassis_quat, chassis_vel], dtype=np.float32)

    def _check_chassis_collision(self):
        """ is robot main chassis contacting floor?"""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == self.chassis_geom_id and contact.geom2 == self.floor_geom_id) or \
               (contact.geom2 == self.chassis_geom_id and contact.geom1 == self.floor_geom_id):
                return True
        return False

    def render(self):
        """ renders in the "human" MuJoCo viewer."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        """ cleans up render loop """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    print("Initializing QuadrupedEnv")
    env = QuadrupedEnv()
    obs, _ = env.reset()
    print("Observation shape:", obs.shape)
    print("Observation space:", env.observation_space.contains(obs))

    print("\Test run: 10 steps, 0 actions")
    total_reward = 0.0
    for step_i in range(10):
        action = np.zeros(12, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"Step {step_i + 1:2d} | Height: {info['height']:.3f}m | "
            f"Fwd Vel: {info['forward_vel']:+.3f}m/s | Up Z: {info['up_z']:.3f} | "
            f"Step Energy: {info['step_energy']:.4f}J | Reward: {reward:+.3f}"
        )
    print(f"\nDone. Total reward: {total_reward:.3f}")
    env.close()