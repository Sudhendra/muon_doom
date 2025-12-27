"""
Training script for ViZDoom Basic with PufferLib PPO.

Supports both Adam and Muon optimizers via config.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import numpy as np
import pufferlib
import pufferlib.vector
import pufferlib.models
from pufferlib import pufferl

# Import our environment wrapper
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from muon_doom.envs.puffer_wrapper import make_puffer_env


def make_policy(env, hidden_size=512):
    """
    Create a CNN policy for ViZDoom.

    Uses PufferLib's Convolutional model (NatureCNN-like) which is appropriate
    for pixel-based observations.
    """
    policy = pufferlib.models.Convolutional(
        env,
        framestack=4,
        flat_size=3136,  # For 84x84 input after conv layers
        hidden_size=hidden_size,
    )

    # Add forward_eval alias for PuffeRL compatibility
    if not hasattr(policy, "forward_eval"):
        policy.forward_eval = policy.forward

    return policy


def train(config):
    """Main training loop."""

    print(f"\n{'=' * 60}")
    print(f"Training ViZDoom Basic with {config['optimizer'].upper()} optimizer")
    print(f"{'=' * 60}\n")

    # Set random seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Create vectorized environments
    env_kwargs = {}
    if config["render"]:
        config["num_envs"] = 1
        config["num_workers"] = 1
        config["backend"] = "Serial"
        env_kwargs["render_mode"] = "human"
        print("Rendering enabled: Forcing num_envs=1, num_workers=1, backend=Serial")

    print(f"Creating {config['num_envs']} environments...")
    vecenv = pufferlib.vector.make(
        make_puffer_env,
        backend=config["backend"],
        num_envs=config["num_envs"],
        num_workers=config["num_workers"],
        env_kwargs=env_kwargs,
    )
    print(f"✓ Environments created")
    print(f"  Observation space: {vecenv.observation_space}")
    print(f"  Action space: {vecenv.action_space}")

    # Create policy
    print(f"\nCreating policy...")
    policy = make_policy(vecenv.driver_env, hidden_size=config["hidden_size"])
    policy = policy.to(config["device"])
    print(f"✓ Policy created")
    print(f"  Model parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create trainer config (with all PufferLib required keys)
    trainer_config = {
        "env": "vizdoom_basic",
        "seed": config["seed"],
        "device": config["device"],
        "torch_deterministic": True,
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
        "total_timesteps": config["total_timesteps"],
        "batch_size": config["batch_size"],
        "bptt_horizon": config["bptt_horizon"],
        "minibatch_size": config["minibatch_size"],
        "max_minibatch_size": 32768,
        "update_epochs": config["update_epochs"],
        "gamma": config["gamma"],
        "gae_lambda": config["gae_lambda"],
        "clip_coef": config["clip_coef"],
        "vf_coef": config["vf_coef"],
        "vf_clip_coef": 0.2,
        "ent_coef": config["ent_coef"],
        "max_grad_norm": config["max_grad_norm"],
        "anneal_lr": config["anneal_lr"],
        "min_lr_ratio": 0.0,
        "data_dir": config["data_dir"],
        "checkpoint_interval": config["checkpoint_interval"],
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "cpu_offload": False,
        "compile": False,
        "compile_mode": "default",
        "use_rnn": False,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
        "prio_alpha": 0.6,
        "prio_beta0": 0.4,
        "precision": "float32",
        "amp": False,
    }

    # Create PuffeRL trainer
    print(f"\nInitializing PuffeRL trainer...")
    trainer = pufferl.PuffeRL(trainer_config, vecenv, policy)
    print(f"✓ Trainer initialized")

    # Training loop
    print(f"\nStarting training for {config['total_timesteps']:,} timesteps...")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    try:
        while trainer.global_step < config["total_timesteps"]:
            # Collect rollouts
            trainer.evaluate()

            # Train on collected data
            logs = trainer.train()

            # Print progress
            if logs is not None:
                elapsed = time.time() - start_time
                sps = trainer.global_step / elapsed if elapsed > 0 else 0

                # Get key metrics
                ep_return = logs.get("environment/episode_return", 0)
                ep_length = logs.get("environment/episode_length", 0)

                print(
                    f"Step: {trainer.global_step:,}/{config['total_timesteps']:,} "
                    f"| SPS: {sps:.0f} "
                    f"| Return: {ep_return:.2f} "
                    f"| Length: {ep_length:.1f}"
                )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Save final model
    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Total time: {time.time() - start_time:.1f}s")

    model_path = trainer.close()
    print(f"Model saved to: {model_path}")
    print(f"{'=' * 60}\n")

    vecenv.close()

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train ViZDoom Basic with PPO")

    # Environment args
    parser.add_argument(
        "--num-envs", type=int, default=8, help="Number of parallel environments"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of worker processes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="Multiprocessing",
        choices=["Serial", "Multiprocessing"],
        help="Vectorization backend",
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")

    # Training args
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "muon"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps to train for",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for training"
    )
    parser.add_argument(
        "--bptt-horizon", type=int, default=16, help="BPTT horizon (sequence length)"
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=512, help="Minibatch size"
    )
    parser.add_argument(
        "--update-epochs", type=int, default=4, help="Number of epochs per update"
    )

    # PPO hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument(
        "--clip-coef", type=float, default=0.2, help="PPO clip coefficient"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value function coefficient"
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Max gradient norm for clipping",
    )
    parser.add_argument("--anneal-lr", action="store_true", help="Anneal learning rate")

    # Model args
    parser.add_argument(
        "--hidden-size", type=int, default=512, help="Hidden size for policy network"
    )

    # Misc args
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="experiments",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Checkpoint every N updates",
    )

    args = parser.parse_args()
    config = vars(args)

    # Create data directory
    os.makedirs(config["data_dir"], exist_ok=True)

    # Train
    train(config)


if __name__ == "__main__":
    main()
