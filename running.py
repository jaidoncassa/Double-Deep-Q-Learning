from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from Metrics import MetricLogger
from tqdm.auto import tqdm
from pathlib import Path
import gymnasium as gym
import ale_py
import torch

# Initialize the environment
env = gym.make(
    "MsPacmanNoFrameskip-v4", repeat_action_probability=0.25, render_mode="human"
)

# Render the state space as a square 84x84 gray image
env = AtariPreprocessing(
    env,
    noop_max=10,
    terminal_on_life_loss=False,
    screen_size=84,
    grayscale_obs=True,
    grayscale_newaxis=False,
)

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
