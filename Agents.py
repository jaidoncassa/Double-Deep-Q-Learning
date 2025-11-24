import torchvision.transforms as transforms
from Neural_Networks import CNN, MLP
from collections import deque
import torch.optim as optim
import gymnasium as gym
import torch.nn as nn
import numpy as np
import random
import torch


class DQN:
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        buffer_size: int,
        batch_size: int,
        update_frequency: int,
        update_target_frequency: int,
        model_lr: float,
        main_net,
        delayed_net,
        seed: int,
    ):
        """Intialize a DQN learning agent


        Args:
            env: The training environment
            Learning_rate:n How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate
            epsilon_decay: How much wereduce epsilon each episode
            final_epsilon: Minimum exploration rate
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env
        self.num_actions = num_actions

        # Create a random number generator with the provided seed to seed the agent for reproducibility
        self.random_generator = np.random.RandomState(seed)

        self.q_values = ...
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Neural Network initialization steps
        self.main_net = main_net
        self.update_frequency = update_frequency  # HYPERPARAM

        # Delayed target network for bootstrap yi
        self.delayed_net = delayed_net
        self.update_target_frequency = update_target_frequency

        # keep track of observations fixed FIFI queue implementation
        self.replay_buffer = deque(maxlen=buffer_size)  # HYPERPARAM
        self.batch_size = batch_size
        self.frame_count = 0

        # NN properties
        self.model_lr = model_lr
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = self.optimizer = optim.RMSprop(
            self.main_net.parameters(),
            lr=self.model_lr,
            alpha=0.95,
            eps=0.01,
            centered=True,
        )

        # Parallelism with GPU for macbooks, NO NVIDIA GPU ANYMORE!!!!
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # place the NN on the GPU
        self.main_net.to(device=self.device)
        self.delayed_net.to(device=self.device)

    def update_cnn_weights(self):
        # Should never happen because we have a check already, but lets be safe
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0

        # Randomly sample from reply buffer of the form (theta(s), a, R, theta(s'))
        samples = random.sample(self.replay_buffer, k=self.batch_size)

        bootstrap_targets = []
        predicted_labels = []
        for s1, a, r, s2, done in samples:
            # Initalize to tensor object
            label = torch.tensor(r, dtype=torch.float32, device=self.device)
            if not done:
                with torch.no_grad():
                    # [Q(s', a1; delayed), ...]
                    q_vals = self.delayed_net(s2)

                    # yi = r + y*argmaxQ(s', a'; delayed)
                    label += self.discount_factor * torch.max(q_vals)

            # Predict Q(s, a; main)
            prediction = self.main_net(s1).squeeze(0)[a]

            # Append to trackers
            bootstrap_targets.append(label)
            predicted_labels.append(prediction)

        # Convert python lists to tensors
        bootstrap_targets = torch.stack(bootstrap_targets)
        predicted_labels = torch.stack(predicted_labels)

        # Calculate average loss
        loss = self.criterion(predicted_labels, bootstrap_targets)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # average Q(s, a) value
        avg_q_value = torch.mean(predicted_labels).item()

        # Return the loss and the average Q-value for logging
        return loss.item(), avg_q_value

    def reset_target_network(self):
        self.delayed_net.load_state_dict(self.main_net.state_dict())

    def get_action(self, q_values):
        """Choose an action using Epsilon greedy approach

        Args:
            q_values: list of Q(s, a)'s computed by the CNN
        """
        if self.random_generator.rand() < self.epsilon:
            return self.random_generator.randint(self.num_actions)

        else:
            return q_values.argmax().item()

    def agent_start(self, state):
        """The first method called when the episode starts, called after the environment starts

        Args:
            state: The initial starting state given to use from the environment.
        Returns:
            The action the agent is taking.
        """
        # Turn numpy array into Tensor obj
        input = self.transform(state)

        # Tell Pytorch object not to track the gradient
        with torch.no_grad():
            # Predict the Q(s, a) values!
            q_values = self.main_net(input)

        # Get the action and return it to the main controller
        action = self.get_action(q_values)

        # Mark down some facts
        self.prev_state = input  # s
        self.prev_action = action  # a

        return action

    def agent_step(self, state, reward):
        """A step taken by the agent.


        Args:
            reward (float): The reward recieved for taking the last action that was taken
            state: The new state returned to us from the environment after we made t-1 step.


        Returns:
            The action the agent is taking
        """
        # Save the full observation (s, a, R, s')
        self.incr_count()

        # Turn into Torcher vector
        input = self.transform(state)

        # Tell Pytorch object not to track the gradient
        with torch.no_grad():
            # Predict the Q(s, a) values!
            q_values = self.main_net(input)

        # Get the action and return it to the main controller
        action = self.get_action(q_values)

        # Save observation
        obs = (self.prev_state, self.prev_action, reward, input, False)
        self.replay_buffer.append(obs)

        # Update trackers
        self.prev_state = input  # s
        self.prev_action = action  # a'

        # Update the weights of the Neural Network every n steps past 32 steps
        q = None
        loss = None
        if (
            self.frame_count % self.update_frequency == 0
            and len(self.replay_buffer) > self.batch_size
        ):
            loss, q = self.update_cnn_weights()

        # Reset the delayed network every m steps
        if (
            self.frame_count > 0
            and self.frame_count % self.update_target_frequency == 0
        ):
            self.reset_target_network()

        self.update_epsilon()
        return action, loss, q

    def agent_end(self, reward):
        """Run when the agent terminates

        Args:
            reward: the reward the agent recieved for entering the terminal state.
        """
        # Save the full observation (s, a, R, s')
        obs = (self.prev_state, self.prev_action, reward, None, True)
        self.replay_buffer.append(obs)
        self.incr_count()

        # Update the weights of the Neural Network every n steps past batch_size steps
        q = None
        loss = None
        if (
            self.frame_count % self.update_frequency == 0
            and len(self.replay_buffer) > self.batch_size
        ):
            loss, q = self.update_cnn_weights()

        # Reset the delayed network every m steps
        if (
            self.frame_count > 0
            and self.frame_count % self.update_target_frequency == 0
        ):
            self.reset_target_network()

        self.update_epsilon()

        self.prev_state = None
        self.prev_action = None
        return loss, q

    def transform(self, img):
        # Must normalize and turn into a tensor object
        return torch.from_numpy(img).float().unsqueeze(0).to(device=self.device) / 255.0

    def incr_count(self):
        self.frame_count += 1

    def update_epsilon(self):
        """Linearly decays epsilon based on the number of frames processed."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class DDQN(DQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        buffer_size: int,
        batch_size: int,
        update_frequency: int,
        update_target_frequency: int,
        model_lr: float,
        main_net,
        delayed_net,
        seed: int,
    ):
        super().__init__(
            env,
            num_actions,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
            buffer_size,
            batch_size,
            update_frequency,
            update_target_frequency,
            model_lr,
            main_net,
            delayed_net,
            seed,
        )

    def update_cnn_weights(self):
        # Should never happen because we have a check already, but lets be safe
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0

        # Randomly sample from reply buffer of the form (theta(s), a, R, theta(s'))
        samples = random.sample(self.replay_buffer, k=self.batch_size)

        bootstrap_targets = []
        predicted_labels = []
        for s1, a, r, s2, done in samples:

            # Initalize to tensor object
            yi = torch.tensor(r, dtype=torch.float32, device=self.device)
            if not done:
                with torch.no_grad():

                    # Choose greedily argmaxQ(s', a; main)
                    greedy_action = self.main_net(s2).argmax().item()

                    # Evaluate the greedy decision Q(s', argmaxQ(s', a; main); delayed)
                    q_val = self.delayed_net(s2).squeeze(0)[greedy_action]

                    # yi = r + y*argmaxQ(s', a'; delayed)
                    yi += self.discount_factor * q_val

            # Predict Q(s, a; main)
            prediction = self.main_net(s1).squeeze(0)[a]

            # Append to trackers
            bootstrap_targets.append(yi)
            predicted_labels.append(prediction)

        # Convert python lists to tensors
        bootstrap_targets = torch.stack(bootstrap_targets)
        predicted_labels = torch.stack(predicted_labels)

        # Calculate average loss (yi - Q(s, a))
        loss = self.criterion(predicted_labels, bootstrap_targets)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()

        # w <-- w + lr(yi - Q(s, a; main))Gradient
        self.optimizer.step()

        # average Q(s, a; main) value
        avg_q_value = torch.mean(predicted_labels).item()

        # Return the loss and the average Q-value for logging
        return loss.item(), avg_q_value


class MsPacmanDQNAgent(DQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        buffer_size: int,
        batch_size: int,
        update_frequency: int,
        update_target_frequency: int,
        model_lr: float,
        seed: int,
    ):
        main = CNN(num_actions)
        delayed = CNN(num_actions)
        super().__init__(
            env,
            num_actions,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
            buffer_size,
            batch_size,
            update_frequency,
            update_target_frequency,
            model_lr,
            main,
            delayed,
            seed,
        )

    def transform(self, img):
        # Must normalize and turn into a tensor object
        return torch.from_numpy(img).float().unsqueeze(0).to(device=self.device) / 255.0


class McPacmanDDQNAgent(DDQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        buffer_size: int,
        batch_size: int,
        update_frequency: int,
        update_target_frequency: int,
        model_lr: float,
        seed: int,
    ):
        main = CNN(num_actions)
        delayed = CNN(num_actions)
        super().__init__(
            env,
            num_actions,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
            buffer_size,
            batch_size,
            update_frequency,
            update_target_frequency,
            model_lr,
            main,
            delayed,
            seed,
        )

    def transform(self, img):
        # Must normalize and turn into a tensor object
        return torch.from_numpy(img).float().unsqueeze(0).to(device=self.device) / 255.0


class LunarLandingDQNAgent(DQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        buffer_size: int,
        batch_size: int,
        update_frequency: int,
        update_target_frequency: int,
        model_lr: float,
        seed: int,
    ):
        main = MLP(num_actions)
        delayed = MLP(num_actions)
        super().__init__(
            env,
            num_actions,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
            buffer_size,
            batch_size,
            update_frequency,
            update_target_frequency,
            model_lr,
            main,
            delayed,
            seed,
        )

    def transform(self, state):
        t = torch.tensor(state, dtype=torch.float32, device=self.device)
        return t.unsqueeze(0)


class LunarLandingDDQNAgent(DDQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        buffer_size: int,
        batch_size: int,
        update_frequency: int,
        update_target_frequency: int,
        model_lr: float,
        seed: int,
    ):
        main = MLP(num_actions)
        delayed = MLP(num_actions)
        super().__init__(
            env,
            num_actions,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
            buffer_size,
            batch_size,
            update_frequency,
            update_target_frequency,
            model_lr,
            main,
            delayed,
            seed,
        )

    def transform(self, state):
        t = torch.tensor(state, dtype=torch.float32, device=self.device)
        return t.unsqueeze(0)
