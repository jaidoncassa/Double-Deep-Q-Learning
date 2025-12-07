from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from Metrics import MetricLogger
from tqdm.auto import tqdm
from pathlib import Path
import gymnasium as gym
import ale_py
import torch

# Initialize the environment
env = gym.make("PongNoFrameskip-v4", render_mode="human")

# Environment settings
state, _ = env.reset()
num_actions = env.action_space.n
episode_reward = 0.0

done = False
while not done:
    action = env.action_space.sample()  # Take a random action initially
    _, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # Update stats
    episode_reward += reward

print(f"Episode Reward: {episode_reward}")
env.close()
