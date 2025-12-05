from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from Metrics import MetricLogger
from Neural_Networks import CNN, MLP
from tqdm.auto import tqdm
from pathlib import Path
import gymnasium as gym
import ale_py
import torch

# Initialize the environment
env = gym.make("MountainCar-v0", render_mode="human")

# # Render the state space as a square 84x84 gray image
# env = AtariPreprocessing(
#     env,
#     noop_max=10,
#     terminal_on_life_loss=False,
#     screen_size=84,
#     grayscale_obs=True,
#     grayscale_newaxis=False,
# )

# env = FrameStackObservation(env, 4)

def transform(state):
    t = torch.tensor(state, dtype=torch.float32)
    return t.unsqueeze(0)


# def transform(img):
#     # Must normalize and turn into a tensor object
#     return torch.from_numpy(img).float().unsqueeze(0) / 255.0


state, info = env.reset()
num_actions = env.action_space.n
n_observations = len(state)
done = False
total_reward = 0.0
n_obs = len(state)

# print(state)

# CNN model information
model = MLP(n_observations=n_observations, num_actions=num_actions)
model.load_state_dict(
    torch.load(
        "MountainCar-v0_Environment/ddqn/123_checkpoints/ddqn_123_agent.pt"
    )
)
# model.load_state_dict(
#     torch.load("models/mspacman_dqn_mps_5.pt")
# )
model.eval()

while not done:

    s = transform(state)
    with torch.no_grad():
        q_values = model(s).squeeze(0)
        action = torch.argmax(q_values).item()

        # Debugging info
        print(f"Q-values: {q_values}, Action: {action}")

    # Take the action and see what happens
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        done = True

print(f"Total Reward: {total_reward}")
env.close()
