from Metrics import MetricLogger
from tqdm.auto import tqdm
from pathlib import Path
import gymnasium as gym
import torch

# Initialize the environment
env = gym.make('Acrobot-v1', render_mode="rgb_array")

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