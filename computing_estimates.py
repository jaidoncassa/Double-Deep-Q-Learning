import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Neural_Networks import MLP

# -------------------------------
# Load environment
# -------------------------------
env = gym.make("CartPole-v1")
discount = 0.99
EPISODES = 50
STEPS = 500

def transform(state):
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

# -------------------------------
# Load your trained DQN + DDQN
# -------------------------------
dqn = MLP(num_actions=2)
ddqn_online = MLP(num_actions=2)
ddqn_target = MLP(num_actions=2)

dqn.load_state_dict(torch.load("CartPole_Environment/cartpole_DQN_agent.pt"))
ddqn_online.load_state_dict(torch.load("CartPole_Environment/cartpole_DDQN_agent.pt"))
ddqn_target.load_state_dict(torch.load("CartPole_Environment/cartpole_DDQN_target_agent.pt"))

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

# ---------------------------------
# Plot results
# ---------------------------------
plt.figure(figsize=(8,5))
plt.plot(dqn_biases, alpha=0.7, label="DQN Bias (Online - Target)")
plt.plot(ddqn_biases, alpha=0.7, label="DDQN Bias (Online - Target)")
plt.axhline(0, color="black", linewidth=1)
plt.title("Maximization Bias Comparison (CartPole)")
plt.xlabel("Time Steps Across Episodes")
plt.ylabel("Q(s,a) - Q_target(s,a)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("max_bias_comparison.png")
plt.show()

