
import os
import time
import numpy as np
import mujoco.viewer
from stable_baselines3 import PPO
from QuadrupedEnv import QuadrupedEnv

script_dir = os.path.dirname(os.path.abspath(__file__))
model_xml = os.path.join(script_dir, "dog.xml")
model_path = os.path.join(script_dir, "standing_policy_v3")

# Load environment and trained policy
env = QuadrupedEnv(model_xml)
model = PPO.load(model_path)

obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    for step in range(500):
        if not viewer.is_running():
            break

        # Let the trained policy choose the joint adjustments
        action, _ = model.predict(obs, deterministic=True)

        # Apply action through PD controller and advance physics
        obs, reward, terminated, truncated, info = env.step(action)

        # Update visualization
        viewer.sync()

        # Pace simulation approximately in real time
        time.sleep(20 * env.model.opt.timestep)

        if step % 25 == 0:
            print(
                f"Time: {env.data.time:.2f}s | "
                f"Reward: {reward:.4f} | "
                f"Height: {env.data.body('chassis').xpos[2]:.4f}m | "
                f"Tilt error: {info['tilt_error']:.5f}"
            )

        if terminated or truncated:
            print("Episode ended.")
            break

env.close()
