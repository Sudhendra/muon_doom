"""
PufferLib integration for ViZDoom Basic environment.

Creates a PufferLib-compatible environment creator function.
"""

import pufferlib
import pufferlib.emulation
from muon_doom.envs.vizdoom_basic import make_vizdoom_basic_env


def make_puffer_env(**env_kwargs):
    """
    Create a PufferLib-wrapped ViZDoom Basic environment.

    This is the environment creator function that PufferLib expects.
    It returns a Gymnasium env that PufferLib will then wrap and vectorize.

    Args:
        **env_kwargs: Keyword arguments passed to the environment

    Returns:
        Gymnasium environment wrapped for PufferLib
    """
    # PufferLib may pass 'buf' and 'seed' for vectorization - filter them out
    # as our env doesn't use them (seed is handled by reset())
    env_kwargs_filtered = {
        k: v for k, v in env_kwargs.items() if k not in ["buf", "seed"]
    }

    # Create the preprocessed ViZDoom environment
    env = make_vizdoom_basic_env(**env_kwargs_filtered)

    # Wrap with PufferLib emulation for compatibility
    # This handles observation/action space standardization
    env = pufferlib.emulation.GymnasiumPufferEnv(env=env)

    return env


if __name__ == "__main__":
    # Test PufferLib integration
    print("Testing PufferLib integration...")

    # Create single env
    env = make_puffer_env()
    print(f"✓ Single env created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset successful: obs shape = {obs.shape}")

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful: reward = {reward}")

    env.close()

    # Test vectorization
    print("\nTesting vectorization...")
    import pufferlib.vector

    vecenv = pufferlib.vector.make(make_puffer_env, backend="Serial", num_envs=4)
    print(f"✓ Vectorized env created with {vecenv.num_envs} envs")
    print(f"  Action space: {vecenv.action_space}")

    vecenv.async_reset(seed=42)
    obs, reward, done, trunc, info, env_id, mask = vecenv.recv()
    print(f"✓ Vectorized reset successful: obs shape = {obs.shape}")

    import numpy as np

    # PufferLib expects actions shaped (num_envs, ) for Discrete spaces
    actions = np.array(
        [vecenv.single_action_space.sample() for _ in range(vecenv.num_envs)]
    )
    print(f"  Actions shape: {actions.shape}, dtype: {actions.dtype}")
    vecenv.send(actions)
    obs, reward, done, trunc, info, env_id, mask = vecenv.recv()
    print(f"✓ Vectorized step successful")

    vecenv.close()
    print("\n✓ All PufferLib integration tests passed!")
