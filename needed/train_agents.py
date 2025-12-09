from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from Metrics import MetricLogger
from tqdm.auto import tqdm
from pathlib import Path
import gymnasium as gym
import random
import Agents
import torch
import ale_py
import argparse

ENV_CONFIGS = {
    # -----------------------------
    # Atari: Pong
    # -----------------------------
    "PongNoFrameskip-v4": {
        "max_episodes": None,
        "max_frames": 2_900_000,
        "max_steps_per_eps": 10_000,
        "update_target_frequency": 10_000,
        "warmup_amt": 10_000,
        "buffer_size": 125_000,
        "update_frequency": 4,
        "batch_size": 32,
        "LR": 1e-4,
        "discount": 0.99,
        "n_step_buffer_sizes": [3],
        "EPS_START": 1.0,
        "EPS_END": 0.01,
        "EPS_DECAY": 1_000_000,
        "max_norm_clipping": 10,
        "agent_class": [
            Agents.AtariDQNAgent,
            Agents.AtariDDQNAgent,
            Agents.AtariNStepDDQNAgent,
        ],
        "is_atari": True,
    },
    # -----------------------------
    # Atari: MsPacman
    # -----------------------------
    "MsPacmanNoFrameskip-v4": {
        "max_episodes": None,
        "max_frames": 20_000_000,
        "max_steps_per_eps": 10_000,
        "update_target_frequency": 10_000,
        "warmup_amt": 50_000,
        "buffer_size": 1_000_000,
        "update_frequency": 4,
        "batch_size": 32,
        "LR": 1e-4,
        "discount": 0.99,
        "n_step_buffer_sizes": [3],
        "EPS_START": 1.0,
        "EPS_END": 0.1,
        "EPS_DECAY": 1_000_000,
        "max_norm_clipping": 10,
        "agent_class": [
            Agents.AtariDQNAgent,
            Agents.AtariDDQNAgent,
            Agents.AtariNStepDDQNAgent,
        ],
        "is_atari": True,
    },
    # -----------------------------
    # Classic Control: MountainCar
    # -----------------------------
    "MountainCar-v0": {
        "max_episodes": 2200,
        "max_frames": None,
        "max_steps_per_eps": 200,
        "update_target_frequency": 1,
        "warmup_amt": 1,
        "buffer_size": 10_000,
        "update_frequency": 1,
        "batch_size": 128,
        "LR": 3e-4,
        "discount": 0.95,
        "n_step_buffer_sizes": [3, 5, 6],
        "EPS_START": 1.0,
        "EPS_END": 0.01,
        "EPS_DECAY": 20_000,
        "max_norm_clipping": 1,
        "agent_class": [
            Agents.ControlDQN,
            Agents.ControlDDQN,
            Agents.ControlNStepDDQN,
        ],
        "is_atari": False,
    },
    # -----------------------------
    # Classic Control: Acrobot-v1
    # -----------------------------
    "Acrobot-v1": {
        "max_episodes": 2000,
        "max_frames": None,
        "max_steps_per_eps": 500,
        "update_target_frequency": 1,
        "warmup_amt": 1,
        "buffer_size": 10_000,
        "update_frequency": 1,
        "batch_size": 128,
        "LR": 3e-4,
        "discount": 0.99,
        "n_step_buffer_sizes": [3, 5, 6],
        "EPS_START": 0.9,
        "EPS_END": 0.01,
        "EPS_DECAY": 20_000,
        "max_norm_clipping": 100,
        "agent_class": [
            Agents.ControlDQN,
            Agents.ControlDDQN,
            Agents.ControlNStepDDQN,
        ],
        "is_atari": False,
    },
    # -----------------------------
    # Classic Control: Cartpole-v1
    # -----------------------------
    "CartPole-v1": {
        "max_episodes": 600,
        "max_steps_per_eps": 500,
        "max_frames": None,
        "update_target_frequency": 1,
        "warmup_amt": 1,
        "buffer_size": 10_000,
        "update_frequency": 1,
        "batch_size": 128,
        "LR": 3e-4,
        "discount": 0.99,
        "n_step_buffer_sizes": [3, 5, 6],
        "EPS_START": 0.9,
        "EPS_END": 0.01,
        "EPS_DECAY": 2_500,
        "max_norm_clipping": 100,
        "agent_class": [
            Agents.ControlDQN,
            Agents.ControlDDQN,
            Agents.ControlNStepDDQN,
        ],
        "is_atari": False,
    },
}


def get_args():
    """Added as part of a plan to incorporate parallelism."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--algo", type=str, required=True, choices=["DQN", "DDQN", "nStepDDQN"]
    )
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--nstep", type=int, default=None)

    return parser.parse_args()


def main():
    # Get the command line args
    args = get_args()
    SEED = args.seed
    ALGO_NAME = args.algo
    ENV_NAME = args.env
    NSTEP = args.nstep

    SAVE_RATE = 500_000
    FRAME_UPDATE = 5_000

    # Grab the game to run
    game = ENV_CONFIGS[ENV_NAME]

    print(f"\n=== Starting training ===")
    print(f"Env: {ENV_NAME}")
    print(f"Algo: {ALGO_NAME}")
    print(f"Seed: {SEED}")
    print(f"N-step: {NSTEP}\n")

    # Initialize the environment
    env = gym.make("PongNoFrameskip-v4")

    if game["is_atari"]:
        env = AtariPreprocessing(
            env,
            noop_max=10,
            terminal_on_life_loss=(
                True if ENV_NAME == "MsPacmanNoFrameskip-v4" else False
            ),
            screen_size=84,
            grayscale_obs=True,
            grayscale_newaxis=False,
        )

        # Contains frame-skipping of 4 + stack 4
        env = FrameStackObservation(env, 4)

    # Seed everything
    env.observation_space.seed(SEED)
    env.action_space.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    env.reset(seed=SEED)

    # Environment settings
    state, _ = env.reset()
    n_observations = len(state) if not game["is_atari"] else None
    num_actions = env.action_space.n
    MAX_EPISODES = game["max_episodes"]
    MAX_FRAMES = game["max_frames"]
    max_steps_per_episode = game["max_steps_per_eps"]

    # Initalize the agent to DQN
    AgentClass = game["agent_class"][0]
    if ALGO_NAME == "DDQN":
        AgentClass = game["agent_class"][1]
    elif ALGO_NAME == "nStepDDQN":
        AgentClass = game["agent_class"][2]

    # Create agent
    agent = AgentClass(
        env=env,
        num_actions=num_actions,
        n_obs=n_observations,
        initial_epsilon=game["EPS_START"],
        epsilon_decay=game["EPS_DECAY"],
        final_epsilon=game["EPS_END"],
        discount_factor=game["discount"],
        buffer_size=game["buffer_size"],
        batch_size=game["batch_size"],
        update_frequency=game["update_frequency"],
        update_target_frequency=game["update_target_frequency"],
        warmup_amt=game["warmup_amt"],
        model_lr=game["LR"],
        seed=SEED,
        max_clip=game["max_norm_clipping"],
        nstep_buffer_size=NSTEP,
    )

    # Saving settings
    prefix = f"{ENV_NAME}_Environment/{ALGO_NAME.lower()}/{SEED}_checkpoints/"
    if NSTEP != None:
        prefix = f"{ENV_NAME}_Environment/{ALGO_NAME.lower()}/seed_{SEED}/{NSTEP}_checkpoints/"
    save_dir = Path(prefix)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create a tracker class
    logger = MetricLogger(save_dir)

    # Metric tracking
    episode_count = 0
    total_frames = 0

    pbar = tqdm(
        total=(MAX_FRAMES if game["max_episodes"] == None else MAX_EPISODES),
        desc=f"Training {ALGO_NAME}",
        dynamic_ncols=True,
        leave=True,
    )
    while True:
        state, _ = env.reset()  # (s)
        action = agent.agent_start(state)  # (a)
        episode_reward = 0.0

        # Play one game until max steps
        for _ in range(1, max_steps_per_episode + 1):
            # Agent makes action in environment (s, a), we get in return (s', R)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update stats
            episode_reward += reward
            total_frames += 1

            # Update progress bar only if driven by frames
            if MAX_EPISODES == None:
                pbar.update(1)

            # Either we step or we terminate and reset
            if done:
                loss, q, mean_max_q = agent.agent_end(reward)
            else:
                action, loss, q, mean_max_q = agent.agent_step(next_state, reward)

            # log a step
            logger.log_step(reward, loss, q, mean_max_q)

            # Specific MsPacman logging and saving
            if game["is_atari"]:
                if total_frames > 0 and total_frames % FRAME_UPDATE == 0:
                    logger.record(episode_count, agent.eps_threshold, total_frames)

                # Save a progress model
                if total_frames % SAVE_RATE == 0:
                    torch.save(
                        agent.main_net.state_dict(),
                        f"{prefix}/{ALGO_NAME.lower()}_{total_frames // SAVE_RATE}_agent.pt",
                    )

            # Break the episode loop
            if done:
                break

        # Update progress bar only if driven by episodes
        if MAX_EPISODES != None:
            pbar.update(1)

        # Episode end
        episode_count += 1
        logger.log_episode()

        # Record and print moving averages
        # Im using different conditions based on if im driving with episodes or frames
        if MAX_FRAMES == None and episode_count % 10 == 0:
            logger.record(episode_count, agent.eps_threshold, total_frames)

        # Exit loop conditions
        if MAX_EPISODES != None and episode_count >= MAX_EPISODES:
            break
        elif MAX_FRAMES != None and total_frames >= MAX_FRAMES:
            break

    pbar.update(pbar.total - pbar.n)
    logger.save_plots()
    pbar.close()
    env.close()

    # Save the model
    postfix = f"{ALGO_NAME.lower()}_final_agent.pt"
    if NSTEP != None:
        postfix = f"{ALGO_NAME.lower()}_{SEED}_{NSTEP}_agent.pt"
    torch.save(
        agent.main_net.state_dict(),
        f"{prefix}/{postfix}",
    )


if __name__ == "__main__":
    main()
