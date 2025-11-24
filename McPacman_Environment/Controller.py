from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from Agents import DDQNPacmentAgent
from Metrics import MetricLogger
import matplotlib.pyplot as plt
from pathlib import Path
import gymnasium as gym
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import ale_py
import torch


# Initialize the environment
seed = 0
env = gym.make("MsPacmanNoFrameskip-v4", repeat_action_probability=0.25)


# Render the state space as a square 84x84 gray image
env = AtariPreprocessing(
    env,
    noop_max=10,
    terminal_on_life_loss=True,
    screen_size=84,
    grayscale_obs=True,
    grayscale_newaxis=False,
)


# Stack the last 4 frames together
env = FrameStackObservation(env, stack_size=4)


# Settings part 1
MAX_TOTAL_FRAMES = 5_000_000
max_steps_per_episode = 5000
buffer_size = 100_000
update_target_frequency = 10_000
batch_size = 32


# Settings part 2, won't really change these
num_actions = env.action_space.n
lr = 0.00025
start_epsilon = 1.0
final_epsilon = 0.1
epsilon_decay_steps = 1_000_000
epsilon_decay = (start_epsilon - final_epsilon) / (epsilon_decay_steps)
discount = 0.99
update_frequency = 4


# CNN model parameters
cnn_lr = 0.00025


# Seed everything
env.action_space.seed(seed)
env.observation_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# initalize the agent
agent = DDQNPacmentAgent(
    env=env,
    num_actions=num_actions,
    learning_rate=lr,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount,
    buffer_size=buffer_size,
    batch_size=batch_size,
    update_frequency=update_frequency,
    update_target_frequency=update_target_frequency,
    cnn_lr=cnn_lr,
    seed=seed,
)


save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)
logger = MetricLogger(save_dir)


# Metric tracking
episode_count = 0
total_frames = 0
total_reward = 0


pbar = tqdm(total=MAX_TOTAL_FRAMES, desc="Training", dynamic_ncols=True, leave=True)
while total_frames < MAX_TOTAL_FRAMES:
    state, _ = env.reset(seed=seed + episode_count)  # (s)
    action = agent.agent_start(state)  # (a)
    episode_reward = 0

    # Play one game until max steps
    for timestep in range(1, max_steps_per_episode):
        # Agent makes action in environment (s, a), we get in return (s', R)
        new_state, reward, done, _, _ = env.step(action)
        reward = max(min(reward, 1), -1)

        # Update stats
        episode_reward += reward
        total_frames += 1

        if done:
            loss, q = agent.agent_end(reward)
        else:
            action, loss, q = agent.agent_step(new_state, reward)

        # Update state
        state = new_state

        # log a step
        logger.log_step(reward, loss, q)

        if done:
            break

        if total_frames % 100 == 0:
            pbar.update(100)

    # Update running reward to check condition for solving
    episode_count += 1

    # Log the end of the episode
    logger.log_episode()

    # Record and print moving averages (every 10 episodes is common)
    if episode_count % 10 == 0:
        logger.record(episode_count, agent.epsilon, total_frames)

    if total_frames > 0 and total_frames % 500_000 == 0:
        torch.save(
            agent.delayed_net.state_dict(),
            f"mspacman_dqn_mps_delayed_{episode_count}.pt",
        )
        torch.save(agent.main_net.state_dict(), f"mspacman_dqn_mps_{episode_count}.pt")


pbar.close()
env.close()
torch.save(agent.delayed_net.state_dict(), f"mspacman_dqn_mps_delayed_5.pt")
torch.save(agent.main_net.state_dict(), f"mspacman_dqn_mps_5.pt")
