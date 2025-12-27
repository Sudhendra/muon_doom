"""
ViZDoom Basic environment wrapper with preprocessing for PufferLib.

Implements:
- Grayscale conversion
- 84x84 resize
- Framestack=4
- Dict observation extraction
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque
import cv2


class GrayscaleObservationWrapper(gym.ObservationWrapper):
    """Convert RGB observations to grayscale."""

    def __init__(self, env):
        super().__init__(env)

        # Get the original shape from the 'screen' key
        old_shape = self.observation_space["screen"].shape

        # New shape will be (height, width, 1) for grayscale
        new_shape = (old_shape[0], old_shape[1], 1)

        # Update the observation space for 'screen' key
        new_obs_space = spaces.Dict(
            {"screen": spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)}
        )

        # Copy other keys if they exist
        for key in self.observation_space.spaces:
            if key != "screen":
                new_obs_space.spaces[key] = self.observation_space[key]

        self.observation_space = new_obs_space

    def observation(self, observation):
        """Convert screen to grayscale."""
        screen = observation["screen"]
        # Convert RGB to grayscale using cv2 weights
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        observation["screen"] = np.expand_dims(gray, axis=-1)
        return observation


class ResizeObservationWrapper(gym.ObservationWrapper):
    """Resize observations to target shape."""

    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape

        # Get the original channels
        old_shape = self.observation_space["screen"].shape
        channels = old_shape[-1] if len(old_shape) == 3 else 1

        # Update observation space
        new_obs_space = spaces.Dict(
            {
                "screen": spaces.Box(
                    low=0, high=255, shape=(*self.shape, channels), dtype=np.uint8
                )
            }
        )

        # Copy other keys if they exist
        for key in self.observation_space.spaces:
            if key != "screen":
                new_obs_space.spaces[key] = self.observation_space[key]

        self.observation_space = new_obs_space

    def observation(self, observation):
        """Resize screen observation."""
        screen = observation["screen"]
        resized = cv2.resize(screen, self.shape, interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)
        observation["screen"] = resized
        return observation


class FrameStackWrapper(gym.Wrapper):
    """Stack frames for temporal information."""

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        # Get shape from screen
        old_shape = self.observation_space["screen"].shape
        height, width = old_shape[0], old_shape[1]

        # Stacked shape is (num_stack, height, width) - channels first for PyTorch
        new_shape = (num_stack, height, width)

        # Create new observation space - single Box (no longer dict)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)

        # Extract screen and remove channel dimension (height, width, 1) -> (height, width)
        screen = obs["screen"].squeeze(-1)

        # Fill frame stack with initial frame
        for _ in range(self.num_stack):
            self.frames.append(screen)

        return self._get_obs(), info

    def step(self, action):
        """Step environment and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract screen and remove channel dimension
        screen = obs["screen"].squeeze(-1)
        self.frames.append(screen)

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """Get stacked observation."""
        # Stack frames: (num_stack, height, width)
        return np.stack(self.frames, axis=0)


class ViZDoomBasicEnv:
    """
    Convenience wrapper that creates a fully preprocessed ViZDoom Basic environment.

    Pipeline:
    1. Create base ViZDoom Gymnasium env
    2. Convert to grayscale
    3. Resize to 84x84
    4. Stack 4 frames

    Final observation: Box(0, 255, shape=(4, 84, 84), dtype=uint8)
    """

    def __init__(self, frame_skip=4, **kwargs):
        """
        Args:
            frame_skip: Number of frames to skip per action
            **kwargs: Additional arguments for ViZDoom env
        """
        # Import here to avoid import errors if vizdoom not installed
        from vizdoom import gymnasium_wrapper

        # Create base environment
        self.env = gym.make("VizdoomBasic-v1", frame_skip=frame_skip, **kwargs)

        # Apply preprocessing pipeline
        self.env = GrayscaleObservationWrapper(self.env)
        self.env = ResizeObservationWrapper(self.env, shape=(84, 84))
        self.env = FrameStackWrapper(self.env, num_stack=4)

    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)


def make_vizdoom_basic_env(frame_skip=4, **kwargs):
    """
    Factory function to create a preprocessed ViZDoom Basic environment.

    Args:
        frame_skip: Number of frames to skip per action (default: 4)
        **kwargs: Additional arguments for ViZDoom env

    Returns:
        Preprocessed Gymnasium environment with observation shape (4, 84, 84)
    """
    env = ViZDoomBasicEnv(frame_skip=frame_skip, **kwargs)
    return env.env


if __name__ == "__main__":
    # Test environment creation
    print("Testing ViZDoom Basic environment creation...")

    env = make_vizdoom_basic_env()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test reset
    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min()}, {obs.max()}]")

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step observation shape: {obs.shape}")
    print(f"Reward: {reward}")

    env.close()
    print("\n✓ Environment creation and basic functionality test passed!")
