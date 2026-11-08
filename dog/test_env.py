from QuadrupedEnv import QuadrupedEnv
from stable_baselines3.common.env_checker import check_env

env = QuadrupedEnv("dog.xml")
check_env(env)
env.close()

print("Environment validation passed!")