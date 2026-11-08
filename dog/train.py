import gymnasium as gym
from stable_baselines3 import PPO
from QuadrupedEnv import QuadrupedEnv

env = gym.wrappers.TimeLimit(
    QuadrupedEnv("dog.xml"),
    max_episode_steps=250
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42
)

model.learn(total_timesteps=50_000)

model.save("standing_policy_v3")

env.close()