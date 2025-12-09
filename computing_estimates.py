import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Neural_Networks import MLP
import pandas as pd

############################################################################
#  Settings for plotting learning curves
############################################################################
seeds = [0, 42, 123]
n_steps = [3, 5, 6]
metrics = ["Reward", "Loss"]
# metrics = ["Reward", "Loss", "Mean_Max_Q"]
end_1 = "0_checkpoints/episode_metrics.csv"
end_2 = "42_checkpoints/episode_metrics.csv"
end_3 = "123_checkpoints/episode_metrics.csv"
end_n_1 = "3_checkpoints/episode_metrics.csv"
end_n_2 = "5_checkpoints/episode_metrics.csv"
end_n_3 = "6_checkpoints/episode_metrics.csv"

end_q_1 = "0_checkpoints/log.csv"
end_q_2 = "42_checkpoints/log.csv"
end_q_3 = "123_checkpoints/log.csv"
end_qn_1 = "3_checkpoints/log.csv"
end_qn_2 = "5_checkpoints/log.csv"
end_qn_3 = "6_checkpoints/log.csv"
environments = ["Acrobot-v1", "CartPole-v1", "MountainCar-v0"]
# environments = ["PongNoFrameskip-v4"]


def load_csv_series(path, column="Reward"):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
    )

    # print(df[column].head())
    return df[column].to_numpy()


def plot_ddqn_style(metric_dict, title, ylabel, colors, save_path):
    """
    metric_dict = {
        "DQN":  [file_seed1.csv, file_seed2.csv, file_seed3.csv],
        "DDQN": [file_seed1.csv, file_seed2.csv, file_seed3.csv],
        "NstepDDQN": [...]
    }
    """

    plt.figure(figsize=(10, 6))

    # -------------------------------------------------------
    # 1) Load ALL curves first to find global minimum length
    # -------------------------------------------------------
    all_curves = []
    for algo, file_list in metric_dict.items():
        for f in file_list:
            arr = load_csv_series(f, column=ylabel)
            all_curves.append(arr)

    min_len = min(len(arr) for arr in all_curves)
    print(f"[Clip] Global minimum length across all seeds = {min_len}")

    # -------------------------------------------------------
    # 2) Plot each algorithm using clipped curves
    # -------------------------------------------------------
    for algo, file_list in metric_dict.items():

        # Load & clip each seed curve
        curves = [
            load_csv_series(f, column=ylabel)[:min_len]
            for f in file_list
        ]

        arr = np.vstack(curves)

        # Compute median and quantiles
        median = np.median(arr, axis=0)
        q10 = np.quantile(arr, 0.10, axis=0)
        q90 = np.quantile(arr, 0.90, axis=0)

        x = np.arange(min_len)

        # Plot median line
        plt.plot(x, median, label=algo, color=colors[algo], linewidth=2)

        # Quantile shading
        plt.fill_between(x, q10, q90, color=colors[algo], alpha=0.2)

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)


def transform(state):
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)


# -------------------------------
# Load environment
# -------------------------------
"""
env = gym.make("Acrobot-v1")
discount = 0.99
EPISODES = 200
STEPS = 500
num_actions = env.action_space.n

# -------------------------------
# Load trained DQN + DDQN
# -------------------------------
dqn = MLP(num_actions=num_actions)
ddqn_online = MLP(num_actions=num_actions)
ddqn_target = MLP(num_actions=num_actions)

dqn.load_state_dict(torch.load("Acrobot_Environment/acrobot_DQN_agent.pt"))
ddqn_online.load_state_dict(torch.load("Acrobot_Environment/acrobot_DDQN_agent.pt"))
ddqn_target.load_state_dict(torch.load("Acrobot_Environment/acrobot_DDQN_target_agent.pt"))

dqn.eval()
ddqn_online.eval()
ddqn_target.eval()

# ------------------------------------
# Compute maximization bias
# ------------------------------------
dqn_biases = []
ddqn_biases = []

for episode in range(EPISODES):
    s, info = env.reset()
    for t in range(STEPS):

        s_t = transform(s)

        # ----------------------
        # 1. DQN maximization bias
        # ----------------------
        with torch.no_grad():
            q_values = dqn(s_t)                   # Q(s, a)
            a_max = torch.argmax(q_values).item() # action chosen by max Q
            q_max = q_values[0, a_max].item()

            # Target is same network for DQN → Q(s', a_max)
            dqn_biases.append(q_max - q_values[0, a_max].item())

        # ----------------------
        # 2. DDQN maximization bias
        # ----------------------
        with torch.no_grad():
            q_online = ddqn_online(s_t)                  # online network picks action
            a_ddqn = torch.argmax(q_online).item()

            q_eval = ddqn_target(s_t)[0, a_ddqn].item()  # target network evaluates
            q_online_val = q_online[0, a_ddqn].item()

            ddqn_biases.append(q_online_val - q_eval)

        # Step environment
        s, _, done, truncated, _ = env.step(a_max)
        if done or truncated:
            break

env.close()
"""

############################################################################
# Plotting DQN vs DDQN vs NstepDDQN Learning Curves
############################################################################
def dqn_vs_ddqn_nstep(n):
    colors = {"DQN": "orange", "DDQN": "blue", "NstepDDQN": "green"}
    for env_name in environments:
        for metric in metrics:
            # Plot DQN vs DDQN vs N-step DDQN Metrics averaged and quantiled over seeds
            plot_ddqn_style(
                metric_dict={
                    "DQN": [
                        f"{env_name}_Environment/dqn/{end_1}",
                        f"{env_name}_Environment/dqn/{end_2}",
                        f"{env_name}_Environment/dqn/{end_3}",
                    ],
                    "DDQN": [
                        f"{env_name}_Environment/ddqn/{end_1}",
                        f"{env_name}_Environment/ddqn/{end_2}",
                        f"{env_name}_Environment/ddqn/{end_3}",
                    ],
                    "NstepDDQN": [
                        f"{env_name}_Environment/nstepddqn/seed_0/{end_n_1}",
                        f"{env_name}_Environment/nstepddqn/seed_42/{end_n_1}",
                        f"{env_name}_Environment/nstepddqn/seed_123/{end_n_1}",
                    ],
                },
                title=f"{env_name} {metric} Learning Curve",
                ylabel=metric,
                colors=colors,
                save_path=f"plots/{env_name.lower()}_{metric.lower()}_dqn_ddqn_nstepddqn-{n}.png",
            )


############################################################################
# Plotting DQN vs DDQN vs NstepDDQN Q Curves
############################################################################
def plot_qvalues(metric_dict, title, colors, save_path):
    """
    metric_dict = {
        "DQN":  [path1, path2, path3],
        "DDQN": [...],
        "NstepDDQN": [...]
    }
    """

    plt.figure(figsize=(10, 6))

    # -------------------------------------------------------
    # 1) Load ALL raw curves first to inspect their lengths
    # -------------------------------------------------------
    all_curves = []

    for algo, file_list in metric_dict.items():
        for f in file_list:
            arr = load_csv_series(f, column="MeanQValue")
            all_curves.append(arr)

    # -------------------------------------------------------
    # 2) Find global minimum length across ALL algorithms
    # -------------------------------------------------------
    min_len = min(len(arr) for arr in all_curves)
    print("Global minimum training length =", min_len)

    # -------------------------------------------------------
    # 3) Now reprocess algorithm by algorithm with trimming
    # -------------------------------------------------------
    for algo, file_list in metric_dict.items():

        curves = [
            load_csv_series(f, column="MeanQValue")[:min_len]
            for f in file_list
        ]

        arr = np.vstack(curves)

        median = np.median(arr, axis=0)
        q10 = np.quantile(arr, 0.10, axis=0)
        q90 = np.quantile(arr, 0.90, axis=0)

        x = np.arange(min_len)

        plt.plot(x, median, label=algo, color=colors[algo], linewidth=2)
        plt.fill_between(x, q10, q90, color=colors[algo], alpha=0.2)

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("MeanQValue")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def qvalues_dqn_vs_ddqn_nstep():
    colors = {"DQN": "orange", "DDQN": "blue", "NstepDDQN": "green"}

    for env_name in environments:

        plot_qvalues(
            metric_dict={
                "DQN": [
                    f"{env_name}_Environment/dqn/{end_q_1}",
                    f"{env_name}_Environment/dqn/{end_q_2}",
                    f"{env_name}_Environment/dqn/{end_q_3}",
                ],
                "DDQN": [
                    f"{env_name}_Environment/ddqn/{end_q_1}",
                    f"{env_name}_Environment/ddqn/{end_q_2}",
                    f"{env_name}_Environment/ddqn/{end_q_3}",
                ],
                "NstepDDQN": [
                    f"{env_name}_Environment/nstepddqn/seed_0/{end_qn_2 if env_name != "PongNoFrameskip-v4" else end_qn_1}",
                    f"{env_name}_Environment/nstepddqn/seed_42/{end_qn_2 if env_name != "PongNoFrameskip-v4" else end_qn_1}",
                    f"{env_name}_Environment/nstepddqn/seed_123/{end_qn_2 if env_name != "PongNoFrameskip-v4" else end_qn_1}",
                ],
            },
            title=f"{env_name} Moving Avg Q-Value (Across Seeds)",
            colors=colors,
            save_path=f"plots/{env_name.lower()}_meanqvalue_dqn_ddqn_nstepddqn.png",
        )


def main():
    dqn_vs_ddqn_nstep() # per step reward
    qvalues_dqn_vs_ddqn_nstep() # Moving Q's

main()
