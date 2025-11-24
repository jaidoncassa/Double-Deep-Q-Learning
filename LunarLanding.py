from Agents import LunarLandingDDQNAgent, LunarLandingDQNAgent
from Metrics import MetricLogger
from tqdm.auto import tqdm
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

USE_DOUBLE_DQN = False
ALGO_NAME = "DDQN" if USE_DOUBLE_DQN else "DQN"

# Initialize the environment
seed = 0
env = gym.make("LunarLanderBare-v0")

# Seed everything
env.observation_space.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)
np.random.seed(seed)


# Settings part 1
update_target_frequency = 1000
max_steps_per_episode = 1000
MAX_TOTAL_EPISODES = 500
buffer_size = 50_000
update_frequency = 4
batch_size = 64

# Settings part 2, won't really change these
start_epsilon = 1.0
final_epsilon = 0.05
epsilon_decay_steps = 50_000
epsilon_decay = (start_epsilon - final_epsilon) / (epsilon_decay_steps)
num_actions = env.action_space.n
discount = 0.99

# model parameters
mlp_lr = 1e-3

# initalize the agent
AgentClass = LunarLandingDDQNAgent if USE_DOUBLE_DQN else LunarLandingDQNAgent
agent = AgentClass(
    env=env,
    num_actions=num_actions,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount,
    buffer_size=buffer_size,
    batch_size=batch_size,
    update_frequency=update_frequency,
    update_target_frequency=update_target_frequency,
    model_lr=mlp_lr,
    seed=seed,
)

save_dir = Path(f"LunarLanding_Environment/lunarlander_{ALGO_NAME.lower()}/checkpoints")
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
    state, _ = env.reset(seed=seed + episode_count)  # (s)
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

        if done:
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
        logger.record(episode_count, agent.epsilon, total_frames)

pbar.close()
env.close()
torch.save(
    agent.main_net.state_dict(),
    f"LunarLanding_Environment/lunarlander_{ALGO_NAME}_agent.pt",
)
