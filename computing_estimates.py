import os
import torch
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from Neural_Networks import CNN

############################################################################
#  Global settings
############################################################################

PONG_RANDOM = -20.7
PONG_HUMAN = 9.3
seeds = [0, 42, 123]
n_steps = [3, 5, 6]  # global list; Pong will internally restrict to [3]
metrics = ["Reward", "Loss"]

# All environments we care about
environments = [
    "Acrobot-v1",
    "CartPole-v1",
    "MountainCar-v0",
    "PongNoFrameskip-v4",
]

# TeX-style label for Q-value plots
Q_YLABEL = r"$\mathbb{E}[\max_{a} Q_{\theta}(S_t, a; \theta)]$"
Q_YLABEL_2 = r"$\mathbb{E}[Q_{\theta}(S_t, a; \theta)]$"


############################################################################
#  Utilities
############################################################################

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def transform(state):
    # Now state is np.ndarray shape (4,84,84)
    return torch.from_numpy(state).float().unsqueeze(0) / 255.0


def load_csv_series(path, column: str):
    """
    Generic loader for whitespace-separated CSVs with headers.
    Works for both episode_metrics.csv and log.csv.
    """
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    return df[column].to_numpy()


def clip_all_curves(curves_by_algo):
    """
    curves_by_algo: dict[str, list[np.ndarray]]
    Returns:
        clipped_curves_by_algo, min_len
    """
    min_len = min(len(arr) for curves in curves_by_algo.values() for arr in curves)
    clipped = {
        algo: [arr[:min_len] for arr in curves]
        for algo, curves in curves_by_algo.items()
    }
    return clipped, min_len


def plot_curves_from_paths(
    metric_dict,
    column,
    title,
    ylabel,
    colors,
    save_path,
    baselines=None,
):
    """
    Generic plotting function that:
      - Loads curves from given file paths
      - Clips all to global min length
      - Computes median + 10/90% quantiles
      - Optionally overlays horizontal baselines

    metric_dict = {
        "DQN":  [path_seed0, path_seed42, ...],
        "DDQN": [...],
        "NstepDDQN": [...]
    }
    """
    plt.figure(figsize=(10, 6))

    # 1) Load all curves
    curves_by_algo = {}
    for algo, file_list in metric_dict.items():
        curves = []
        for fpath in file_list:
            if not os.path.exists(fpath):
                print(f"[WARN] Missing file for {algo}: {fpath}")
                continue
            arr = load_csv_series(fpath, column=column)
            curves.append(arr)
        if curves:
            curves_by_algo[algo] = curves

    if not curves_by_algo:
        print(f"[WARN] No data available for plot: {title}")
        plt.close()
        return

    # 2) Clip to global min length
    clipped, min_len = clip_all_curves(curves_by_algo)
    print(f"[Clip] {title}: global minimum length = {min_len}")

    x = np.arange(min_len)

    # 3) Plot median + quantiles
    for algo, curves in clipped.items():
        arr = np.vstack(curves)
        median = np.median(arr, axis=0)
        q10 = np.quantile(arr, 0.10, axis=0)
        q90 = np.quantile(arr, 0.90, axis=0)

        plt.plot(x, median, label=algo, color=colors[algo], linewidth=2)
        plt.fill_between(x, q10, q90, color=colors[algo], alpha=0.2)

    # 4) Optional baselines (horizontal lines)
    if baselines is not None:
        for algo, value in baselines.items():
            if algo in curves_by_algo:
                plt.axhline(
                    y=value,
                    color=colors[algo],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"{algo} baseline",
                )

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVE] {save_path}")


############################################################################
#  Paths for metrics / logs / agents
############################################################################

def base_env_dir(env_name: str) -> str:
    return f"{env_name}_Environment"


def metrics_paths_dqn(env_name: str):
    base = base_env_dir(env_name)
    return [
        os.path.join(base, "dqn", f"{seed}_checkpoints", "episode_metrics.csv")
        for seed in seeds
    ]


def metrics_paths_ddqn(env_name: str):
    base = base_env_dir(env_name)
    return [
        os.path.join(base, "ddqn", f"{seed}_checkpoints", "episode_metrics.csv")
        for seed in seeds
    ]


def metrics_paths_nstep(env_name: str, n: int):
    base = base_env_dir(env_name)
    return [
        os.path.join(
            base, "nstepddqn", f"seed_{seed}", f"{n}_checkpoints", "episode_metrics.csv"
        )
        for seed in seeds
    ]


def log_paths_dqn(env_name: str):
    base = base_env_dir(env_name)
    return [
        os.path.join(base, "dqn", f"{seed}_checkpoints", "log.csv")
        for seed in seeds
    ]


def log_paths_ddqn(env_name: str):
    base = base_env_dir(env_name)
    return [
        os.path.join(base, "ddqn", f"{seed}_checkpoints", "log.csv")
        for seed in seeds
    ]


def log_paths_nstep(env_name: str, n: int):
    base = base_env_dir(env_name)
    return [
        os.path.join(
            base, "nstepddqn", f"seed_{seed}", f"{n}_checkpoints", "log.csv"
        )
        for seed in seeds
    ]


def agent_path(env_name: str, algo: str, seed: int, n: int | None = None) -> str:
    """
    Build path to a trained agent.

    algo ∈ {"DQN", "DDQN", "NstepDDQN"}
    MountainCar-v0 uses special naming: algo_final_agent.pt
    PongNoFrameskip-v4 uses special naming: algo_5_agent.pt
    Others use algo_seed_agent.pt
    NstepDDQN uses: nstepddqn_{seed}_{n}_agent.pt in the n-step checkpoint folder.
    """
    base = base_env_dir(env_name)

    if algo in ("DQN", "DDQN"):
        algo_l = algo.lower()
        ckpt_dir = os.path.join(base, algo_l, f"{seed}_checkpoints")

        if env_name == "MountainCar-v0":
            fname = f"{algo_l}_final_agent.pt"
        elif env_name == "PongNoFrameskip-v4":
            fname=f"{algo_l}_5_agent.pt"
        else:
            fname = f"{algo_l}_{seed}_agent.pt"

        return os.path.join(ckpt_dir, fname)

    elif algo == "NstepDDQN":
        if n is None:
            raise ValueError("n must be provided for NstepDDQN agent path.")
        ckpt_dir = os.path.join(
            base, "nstepddqn", f"seed_{seed}", f"{n}_checkpoints"
        )

        if env_name == "PongNoFrameskip-v4":
            fname = f"nstepddqn_5_agent.pt"
        else:
            fname = f"nstepddqn_{seed}_{n}_agent.pt"
        
        return os.path.join(ckpt_dir, fname)
    else:
        raise ValueError(f"Unknown algo: {algo}")


############################################################################
#  Learning curves: Reward & Loss
############################################################################

def plot_learning_curves_for_env_and_n(env_name: str, n: int):
    """
    For a given environment and n-step value, plot:
        DQN vs DDQN vs NstepDDQN for Reward and Loss.
    """
    colors = {"DQN": "orange", "DDQN": "blue", "NstepDDQN": "green"}

    for metric in metrics:
        metric_dict = {
            "DQN": metrics_paths_dqn(env_name),
            "DDQN": metrics_paths_ddqn(env_name),
            "NstepDDQN": metrics_paths_nstep(env_name, n),
        }

        title = f"{env_name} {metric} Learning Curve (n={n})"
        save_path = f"plots/{env_name.lower()}_{metric.lower()}_dqn_ddqn_nstepddqn_n{n}.png"

        plot_curves_from_paths(
            metric_dict=metric_dict,
            column=metric,
            title=title,
            ylabel=metric,
            colors=colors,
            save_path=save_path,
        )


def plot_all_learning_curves():
    """
    Wrapper: loops over all environments and n-values.
    PongNoFrameskip-v4 only uses n=3.
    """
    for env_name in environments:
        if env_name == "PongNoFrameskip-v4":
            env_n_list = [3]  # Pong only ran n=3
        else:
            env_n_list = n_steps

        for n in env_n_list:
            plot_learning_curves_for_env_and_n(env_name, n)


############################################################################
#  Q-value curves
############################################################################

def plot_qvalues_for_env_and_n_generic(env_name: str, n: int):
    """
    For non-Pong environments:
        - Use log.csv
        - Column: MeanQValue
        - No discounted-return baseline.
    """
    colors = {"DQN": "orange", "DDQN": "blue", "NstepDDQN": "green"}

    metric_dict = {
        "DQN": log_paths_dqn(env_name),
        "DDQN": log_paths_ddqn(env_name),
        "NstepDDQN": log_paths_nstep(env_name, n),
    }

    title = f"{env_name} Q-Value Learning Curve (n={n})"
    save_path = f"plots/{env_name.lower()}_meanqvalue_dqn_ddqn_nstepddqn_n{n}.png"

    plot_curves_from_paths(
        metric_dict=metric_dict,
        column="MeanQValue",
        title=title,
        ylabel=Q_YLABEL if env_name == "PongNoFrameskip-v4" else Q_YLABEL_2,
        colors=colors,
        save_path=save_path,
    )


############################################################################
#  Policy evaluation (Pong only)
############################################################################

def evaluate_agent(env_name: str, agent_path_str: str, gamma: float = 0.99, episodes: int = 50) -> float:
    """
    Evaluate a single trained agent by running it in the environment and
    returning the average discounted return over 'episodes' episodes.
    """
    if not os.path.exists(agent_path_str):
        print(f"[WARN] Agent file does not exist, skipping: {agent_path_str}")
        return np.nan

    env = gym.make(env_name)
    if env_name == "PongNoFrameskip-v4":
        # # Render the state space as a square 84x84 gray image
        env = AtariPreprocessing(
            env,
            noop_max=10,
            terminal_on_life_loss=False,
            screen_size=84,
            grayscale_obs=True,
            grayscale_newaxis=False,
        )

        env = FrameStackObservation(env, 4)

    num_actions = env.action_space.n

    model = CNN(num_actions=num_actions)
    model.load_state_dict(torch.load(agent_path_str))
    model.eval()

    returns = []

    for _ in range(episodes):
        state, info = env.reset()
        done = False
        truncated = False
        G = 0.0
        discount = 1.0

        while not (done or truncated):
            s_t = transform(state)
            with torch.no_grad():
                q_values = model(s_t)
                action = torch.argmax(q_values, dim=1).item()

            next_state, reward, done, truncated, info = env.step(action)
            G += discount * reward
            discount *= gamma
            state = next_state

        returns.append(G)

    env.close()
    return float(np.mean(returns))


def evaluate_pong_discounted_returns(n: int = 3, gamma: float = 0.99, episodes: int = 50):
    """
    Compute discounted-return baselines for PongNoFrameskip-v4 ONLY:
        - DQN
        - DDQN
        - NstepDDQN (for the given n, here n=3)
    Returns dict: { "DQN": baseline, "DDQN": baseline, "NstepDDQN": baseline }
    """
    env_name = "PongNoFrameskip-v4"
    baselines = {}

    for algo in ("DQN", "DDQN", "NstepDDQN"):
        if algo == "NstepDDQN" and n != 3:
            continue  # Pong only has n=3

        seed_returns = []
        for seed in seeds:
            try:
                path = agent_path(env_name, algo, seed, n if algo == "NstepDDQN" else None)
            except ValueError:
                continue

            mean_return = evaluate_agent(env_name, path, gamma=gamma, episodes=episodes)
            if not np.isnan(mean_return):
                seed_returns.append(mean_return)

        if seed_returns:
            baselines[algo] = float(np.mean(seed_returns))
            print(f"[EVAL] Pong {algo} (n={n}): baseline = {baselines[algo]:.3f}")

    return baselines


def plot_qvalues_for_env_and_n_pong(env_name: str, n: int):
    """
    Pong special case:
        - Q tracking is in episode_metrics.csv
        - Column: Mean_Max_Q
        - Also overlays discounted-return baselines.
    """
    assert env_name == "PongNoFrameskip-v4", "This function is Pong-specific."

    colors = {"DQN": "orange", "DDQN": "blue", "NstepDDQN": "green"}

    metric_dict = {
        "DQN": metrics_paths_dqn(env_name),
        "DDQN": metrics_paths_ddqn(env_name),
        "NstepDDQN": metrics_paths_nstep(env_name, n),  # n=3
    }

    baselines = evaluate_pong_discounted_returns(n=n, gamma=0.99, episodes=50)

    title = f"{env_name} Action-Value Estimates vs Discounted Return (n={n})"
    save_path = f"plots/{env_name.lower()}_qvalues_with_baseline_n{n}.png"

    plot_curves_from_paths(
        metric_dict=metric_dict,
        column="Mean_Max_Q",  # Pong per-episode argmax-Q column
        title=title,
        ylabel=Q_YLABEL if env_name == "PongNoFrameskip-v4" else Q_YLABEL_2,
        colors=colors,
        save_path=save_path,
        baselines=baselines,
    )


def plot_all_qvalue_curves():
    """
    Wrapper:
      - For non-Pong envs: use log.csv & MeanQValue (no baseline).
      - For PongNoFrameskip-v4: use episode_metrics & Mean_Max_Q with baselines.
    """
    for env_name in environments:
        if env_name == "PongNoFrameskip-v4":
            # Pong only ran n=3, and we want baselines here
            plot_qvalues_for_env_and_n_pong(env_name, n=3)
        else:
            for n in n_steps:
                plot_qvalues_for_env_and_n_generic(env_name, n)


def normalize_score(score_agent, score_random, score_human):
    return (score_agent - score_random) / (score_human - score_random)


def evaluate_agent_pong(env_name, agent, epsilon=0.05, max_frames=18000):
    env = gym.make(env_name)
    env = AtariPreprocessing(
        env,
        noop_max=10,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
    )

    # Contains frame-skipping of 4 + stack 4
    env = FrameStackObservation(env, 4)
    state, info = env.reset()

    episode_rewards = []
    total_reward_current_episode = 0
    frames = 0

    while frames < max_frames:

        # epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = agent(transform(state))
                action = torch.argmax(q).item()

        next_state, reward, done, truncated, info = env.step(action)

        total_reward_current_episode += reward
        frames += 1
        state = next_state

        if done or truncated:
            # store single episode score
            episode_rewards.append(total_reward_current_episode)

            # reset for next episode
            total_reward_current_episode = 0
            state, info = env.reset()

    env.close()

    # Atari evaluation returns average score per episode
    return np.mean(episode_rewards), episode_rewards


def load_agent(env_name, algo, seed, n=None):
    base = f"{env_name}_Environment/{algo.lower()}"

    if algo == "DQN" or algo == "DDQN":
        path = f"{base}/{seed}_checkpoints/{algo.lower()}_5_agent.pt"
        agent = CNN(num_actions=gym.make(env_name).action_space.n)
        agent.load_state_dict(torch.load(path))
        agent.eval()
        return agent

    else:
        # filename pattern: nstepddqn_<seed>_<n>_agent.pt
        path = f"{base}/seed_{seed}/{n}_checkpoints/nstepddqn_5_agent.pt"
        agent = CNN(num_actions=gym.make(env_name).action_space.n)
        agent.load_state_dict(torch.load(path))
        agent.eval()
        return agent


def evaluate_pong_all_agents(n_value):
    env = "PongNoFrameskip-v4"
    results = {}

    algos = ["DQN", "DDQN", "NstepDDQN"]
    seeds = [0, 123]

    for algo in algos:
        seed_means = []

        for seed in seeds:
            agent = load_agent(env, algo, seed, n_value)

            mean_score, per_ep_scores = evaluate_agent_pong(env, agent)
            seed_means.append(mean_score)

        avg_raw_score = np.mean(seed_means)
        normalized = normalize_score(avg_raw_score, PONG_RANDOM, PONG_HUMAN)

        results[algo] = {
            "per_seed_scores": seed_means,
            "avg_raw_reward": avg_raw_score,
            "normalized_score": normalized,
        }

    return results



############################################################################
#  Main
############################################################################

def main():
    # # 1) Reward/Loss learning curves
    # plot_all_learning_curves()

    # # 2) Q-value curves (with Pong baselines)
    # plot_all_qvalue_curves()

    print("\n=== Evaluating Pong (Raw & Normalized Scores) ===")
    pong_results = evaluate_pong_all_agents(n_value=3)

    for algo, data in pong_results.items():
        print(f"\n{algo}:")
        print("  Per-seed averages:", data["per_seed_scores"])
        print("  Avg raw reward:   ", data["avg_raw_reward"])
        print("  Normalized score: ", data["normalized_score"])

if __name__ == "__main__":
    main()
