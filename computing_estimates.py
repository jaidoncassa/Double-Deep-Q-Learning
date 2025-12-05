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
metrics = ["Reward", "Loss", "Length"]
end_1 = "0_checkpoints/episode_metrics.csv"
end_2 = "42_checkpoints/episode_metrics.csv"
end_3 = "123_checkpoints/episode_metrics.csv"
end_n_1 = "3_checkpoints/episode_metrics.csv"
end_n_2 = "5_checkpoints/episode_metrics.csv"
end_n_3 = "6_checkpoints/episode_metrics.csv"
environments = ["Acrobot-v1", "CartPole-v1"]


def load_csv_series(path, column="Reward"):
    df = pd.read_csv(
        path,
        skiprows=1,
        sep=r"\s+",
        names=["Reward", "Loss", "Length"],
        engine="python",
    )
    return df[column].to_numpy()


def plot_ddqn_style(metric_dict, title, ylabel, colors, save_path):
    """
    metric_dict = {
        "DQN":  [file_seed1.csv, file_seed2.csv, file_seed3.csv],
        "DDQN": [file_seed1.csv, file_seed2.csv, file_seed3.csv],
    }
    """

    plt.figure(figsize=(10, 6))

    for algo, file_list in metric_dict.items():

        # Load all seeds for this algorithm
        curves = [load_csv_series(f, column=ylabel) for f in file_list]

        # Stack into array shape (seeds, episodes)
        arr = np.vstack(curves)

        # Compute median and quantiles
        median = np.median(arr, axis=0)
        q10 = np.quantile(arr, 0.10, axis=0)
        q90 = np.quantile(arr, 0.90, axis=0)

        x = np.arange(len(median))

        # Plot median
        plt.plot(x, median, label=algo, color=colors[algo], linewidth=2)

        # Plot quantile shading
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
# Plotting different N-step values for N-step DDQN
############################################################################
n_steps = [3, 5, 6]
colors = {"N-step=3": "orange", "N-step=5": "blue", "N-step=6": "green"}


def nstepddqn_comparison():
    for env_name in environments:
        for metric in metrics:
            plot_ddqn_style(
                metric_dict={
                    "N-step=3": [
                        f"{env_name}_Environment/nstepddqn/seed_0/{end_n_1}",
                        f"{env_name}_Environment/nstepddqn/seed_42/{end_n_1}",
                        f"{env_name}_Environment/nstepddqn/seed_123/{end_n_1}",
                    ],
                    "N-step=5": [
                        f"{env_name}_Environment/nstepddqn/seed_0/{end_n_2}",
                        f"{env_name}_Environment/nstepddqn/seed_42/{end_n_2}",
                        f"{env_name}_Environment/nstepddqn/seed_123/{end_n_2}",
                    ],
                    "N-step=6": [
                        f"{env_name}_Environment/nstepddqn/seed_0/{end_n_3}",
                        f"{env_name}_Environment/nstepddqn/seed_42/{end_n_3}",
                        f"{env_name}_Environment/nstepddqn/seed_123/{end_n_3}",
                    ],
                },
                title=f"{env_name} {metric} Learning Curve for different N values",
                ylabel=metric,
                colors=colors,
                save_path=f"plots/{env_name.lower()}_{metric.lower()}_nstepddqn_comparison.png",
            )


############################################################################
# Plotting DQN vs DDQN vs NstepDDQN Learning Curves
############################################################################
def dqn_vs_ddqn():
    colors = {"DQN": "orange", "DDQN": "blue"}
    for env_name in environments:
        metric_dict = {
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
        }
        for metric in metrics:
            # Plot DQN vs DDQN Metrics averaged and quantiled over seeds
            plot_ddqn_style(
                metric_dict=metric_dict,
                title=f"{env_name} {metric} Learning Curve",
                ylabel=metric,
                colors=colors,
                save_path=f"plots/{env_name.lower()}_{metric.lower()}_dqn_ddqn.png",
            )


def dqn_vs_ddqn_nstep():
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
                save_path=f"plots/{env_name.lower()}_{metric.lower()}_dqn_ddqn_nstepddqn.png",
            )


def main():
    nstepddqn_comparison()
    dqn_vs_ddqn_nstep()


main()
