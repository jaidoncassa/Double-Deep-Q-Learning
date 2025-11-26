from Agents import CartPoleDQNAgent, CartPoleDDQNAgent
from Metrics import MetricLogger
from tqdm.auto import tqdm
from pathlib import Path
import gymnasium as gym
import numpy as np
import random
import torch

USE_DOUBLE_DQN = False
ALGO_NAME = "DDQN" if USE_DOUBLE_DQN else "DQN"

# Initialize the environment
seed = 40
env = gym.make("CartPole-v1")

# Seed everything
# env.observation_space.seed(seed)
# env.action_space.seed(seed)
# torch.manual_seed(seed)
# random.seed(seed)
# env.reset(seed=seed)

# Environment settings
state, _ = env.reset()
MAX_TOTAL_EPISODES = 600
n_observations = len(state)
num_actions = env.action_space.n


# Agent settings
update_target_frequency = 1
max_steps_per_episode = 500
buffer_size = 10_000
update_frequency = 1
batch_size = 128
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2_500
# epsilon_decay = (start_epsilon - final_epsilon) / (epsilon_decay_steps)
discount = 0.99

# model parameters
LR = 3e-4

# initalize the agent
AgentClass = CartPoleDDQNAgent if USE_DOUBLE_DQN else CartPoleDQNAgent
agent = AgentClass(
    env=env,
    num_actions=num_actions,
    n_obs=n_observations,
    initial_epsilon=EPS_START,
    epsilon_decay=EPS_DECAY,
    final_epsilon=EPS_END,
    discount_factor=discount,
    buffer_size=buffer_size,
    batch_size=batch_size,
    update_frequency=update_frequency,
    update_target_frequency=update_target_frequency,
    model_lr=LR,
    seed=seed,
)

# Saving settings
save_dir = Path(f"CartPole_Environment/cartpole_{ALGO_NAME.lower()}/checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)
logger = MetricLogger(save_dir)

# Metric tracking
episode_count = 0
total_frames = 0

pbar = tqdm(
    total=MAX_TOTAL_EPISODES,
    desc=f"Training {ALGO_NAME}",
    dynamic_ncols=True,
    leave=True,
)
while episode_count < MAX_TOTAL_EPISODES:
    state, _ = env.reset()  # (s)
    action = agent.agent_start(state)  # (a)
    episode_reward = 0.0

    # Play one game until max steps
    for timestep in range(1, max_steps_per_episode + 1):
        # Agent makes action in environment (s, a), we get in return (s', R)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update stats
        episode_reward += reward
        total_frames += 1

        if terminated:
            loss, q = agent.agent_end(reward)
        else:
            action, loss, q = agent.agent_step(next_state, reward)

        # log a step
        logger.log_step(reward, loss, q)

        if done:
            break

    # Episode end
    episode_count += 1
    logger.log_episode()

    pbar.update(1)

    # Record and print moving averages (every 10 episodes is common)
    if episode_count % 10 == 0:
        logger.record(episode_count, agent.eps_threshold, total_frames)

pbar.close()
env.close()
torch.save(
    agent.main_net.state_dict(),
    f"CartPole_Environment/cartpole_{ALGO_NAME}_agent.pt",
)
torch.save(
    agent.delayed_net.state_dict(),
    f"CartPole_Environment/cartpole_{ALGO_NAME}_target_agent.pt",
)
