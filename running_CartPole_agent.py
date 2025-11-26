from Neural_Networks import MLP
from tqdm.auto import tqdm
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

env = gym.make("CartPole-v1", render_mode="human")
num_actions = env.action_space.n

def transform(state):
    t = torch.tensor(state, dtype=torch.float32)
    return t.unsqueeze(0)

# CNN model information
model = MLP(num_actions=num_actions)
model.load_state_dict(torch.load("CartPole_Environment/cartpole_DQN_agent.pt"))
model.eval()
state, info = env.reset()
done = False
total_reward = 0.0

while not done:

    s = transform(state)
    with torch.no_grad():
        q_values = model(s).squeeze(0)
        action = torch.argmax(q_values).item()

    # Take the action and see what happens
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        done = True

print(f"Total Reward: {total_reward}")
env.close()