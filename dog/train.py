import os
import argparse
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from QuadrupedEnv import QuadrupedEnv


# allow for different devices to be used for torch (depending on hardware present)
def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    elif device_arg == "mps":
        if not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return "cpu"
        return "mps"
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return "cuda"
    return "cpu"


# factory for making environments that are used in training
def make_env(model_xml: str, max_episode_steps: int = 250):
    """Factory returning an env initializer for SubprocVecEnv."""
    def _init():
        env = QuadrupedEnv(model_xml)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train Quadruped RL standing policy")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "mps", "cpu", "cuda"],
        # GPU (MPS/CUDA) can be faster for large networks but PPO's small MLP
        # with many parallel envs is usually CPU-bound on data collection.
        # Switch to 'mps' or 'cuda' to benchmark on your hardware.
        help="Device to train on (default: 'cpu')",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1,000,000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel SubprocVecEnv workers (default: 8)",
    )
    args = parser.parse_args()

    selected_device = resolve_device(args.device)
    print("--- Quadruped Policy Training ---")
    if selected_device == "mps":
        print("Hardware Device: MPS (Apple Silicon M-series GPU via Metal)")
    else:
        print(f"Hardware Device: {selected_device.upper()}")
    print(f"Total Timesteps:        {args.timesteps:,}")
    print(f"Parallel Environments:  {args.n_envs}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_xml = os.path.join(script_dir, "dog.xml")

    # Spin up N independent MuJoCo processes for parallel rollout collection.
    # Each worker owns its own model/data so there are no thread-safety concerns.
    env = SubprocVecEnv([
        make_env(model_xml, max_episode_steps=250)
        for _ in range(args.n_envs)
    ])
    env = VecMonitor(env)   # tracks episode rewards/lengths across all workers

    model = PPO(
        "MlpPolicy",
        env,
        device=selected_device,
        verbose=1,
        seed=42,
    )

    model.learn(total_timesteps=args.timesteps)

    save_path = os.path.join(script_dir, "standing_policy_v3")
    model.save(save_path)
    print(f"Model saved successfully to {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
