from Neural_Networks import CNN, MLP
from collections import namedtuple
from collections import deque
import torch.optim as optim
import gymnasium as gym
import torch.nn as nn
import numpy as np
import random
import torch
import math


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
        self.discount_factor = discount_factor

        # Neural Network initialization steps
        self.main_net = main_net
        self.update_frequency = update_frequency  # HYPERPARAM

        # Delayed target network for bootstrap yi
        self.delayed_net = delayed_net
        self.update_target_frequency = update_target_frequency
        self.delayed_net.load_state_dict(self.main_net.state_dict())

        # keep track of observations fixed FIFI queue implementation
        self.replay_buffer = deque(maxlen=buffer_size)  # HYPERPARAM
        self.Transition = namedtuple(
            "Transition", ("state", "action", "reward", "next_state", "done")
        )
        self.batch_size = batch_size
        self.frame_count = 0

        # Exploration parameters
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.eps_threshold = 0
        self.set_epsilon()

        # NN properties
        self.optimizer = self.optimizer = optim.AdamW(
            self.main_net.parameters(), lr=model_lr, amsgrad=True
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
        batch = self.Transition(*zip(*samples))

        # 3. Process States, Actions, and Rewards
        # The states were saved as NumPy arrays, so we need to transform and cat them.
        # We also need to filter out None next_states (terminal states)

        # Identify non-final next states for mask
        non_final_next_states_np = [s for s in batch.next_state if s is not None]
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device,
            dtype=torch.bool,
        )

        # Transform and concatenate all batch elements
        state_batch = torch.cat([self.transform(s) for s in batch.state])
        action_batch = torch.tensor(
            batch.action, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        )

        # Transform and concatenate non-final next states
        if non_final_next_states_np:
            non_final_next_states = torch.cat(
                [self.transform(s) for s in non_final_next_states_np]
            )
        else:
            non_final_next_states = torch.empty(
                0, state_batch.shape[1], device=self.device
            )  # Handle empty case

        # --- Q-Value Calculation ---

        # 4. Compute Q(s_t, a) from the main network (State Action Values)
        # policy_net(state_batch) gives Q(s_t) for all actions. gather(1, action_batch) selects the Q-value for the action taken.
        state_action_values = self.main_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        # 5. Compute the Target V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if non_final_next_states.numel() > 0:
                # DQN logic: max_a Q(s', a; target)
                next_state_values[non_final_mask] = (
                    self.delayed_net(non_final_next_states).max(1).values
                )

        # 6. Compute Expected Q Values (Bellman Target)
        # Expected Q = R + gamma * V(s_{t+1})
        expected_state_action_values = (
            next_state_values * self.discount_factor
        ) + reward_batch

        # 7. Compute Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 8. Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # *** ADDED: GRADIENT CLIPPING FOR STABILITY ***
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), max_norm=100)

        self.optimizer.step()

        # Average Q(s, a) value
        avg_q_value = torch.mean(state_action_values).item()

        # Return the loss and the average Q-value for logging
        return loss.item(), avg_q_value

    def reset_target_network(self):
        TAU = 0.005
        with torch.no_grad():
            for target_param, param in zip(
                self.delayed_net.parameters(), self.main_net.parameters()
            ):
                target_param.data.copy_(
                    TAU * param.data + (1 - TAU) * target_param.data
                )
        # TAU = 0.005
        # # Soft update of the target network's weights
        # # θ′ ← τ θ + (1 −τ )θ′
        # target_net_state_dict = self.delayed_net.state_dict()
        # policy_net_state_dict = self.main_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[
        #         key
        #     ] * TAU + target_net_state_dict[key] * (1 - TAU)
        # self.delayed_net.load_state_dict(target_net_state_dict)

    def get_action(self, q_values):
        """Choose an action using Epsilon greedy approach

        Args:
            q_values: list of Q(s, a)'s computed by the CNN
        """
        self.set_epsilon()

        # Increase number of frames seen
        self.incr_count()

        sample = self.random_generator.random()
        if sample < self.eps_threshold:
            return self.env.action_space.sample()

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
        with torch.no_grad():
            # Predict the Q(s, a) values!
            q_values = self.main_net(input)

        # Get the action and return it to the main controller
        action = self.get_action(q_values)

        # Mark down some facts
        self.prev_state = state  # s
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
        obs = (self.prev_state, self.prev_action, reward, state, False)
        self.replay_buffer.append(obs)

        # Turn into Torcher vector
        input = self.transform(state)
        with torch.no_grad():
            # Predict the Q(s, a) values!
            q_values = self.main_net(input)

        # Get the action and return it to the main controller
        action = self.get_action(q_values)

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

        # Update trackers
        self.prev_state = state  # s
        self.prev_action = action  # a'

        return action, loss, q

    def agent_end(self, reward):
        """Run when the agent terminates

        Args:
            reward: the reward the agent recieved for entering the terminal state.
        """
        # Save the full observation (s, a, R, s')
        obs = (self.prev_state, self.prev_action, reward, None, True)
        self.replay_buffer.append(obs)

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

        self.prev_state = None
        self.prev_action = None
        return loss, q

    def transform(self, state):
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def incr_count(self):
        self.frame_count += 1

    def set_epsilon(self):
        # Exp decays epsilon based on the number of frames processed.
        self.eps_threshold = self.final_epsilon + (
            self.initial_epsilon - self.final_epsilon
        ) * math.exp(-1.0 * self.frame_count / self.epsilon_decay)


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
        batch = self.Transition(*zip(*samples))

        # 3. Process States, Actions, and Rewards
        # Identify non-final next states for mask
        non_final_next_states_np = [s for s in batch.next_state if s is not None]
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device,
            dtype=torch.bool,
        )

        # Transform and concatenate all batch elements
        state_batch = torch.cat([self.transform(s) for s in batch.state])
        action_batch = torch.tensor(
            batch.action, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        )

        # Transform and concatenate non-final next states
        if non_final_next_states_np:
            non_final_next_states = torch.cat(
                [self.transform(s) for s in non_final_next_states_np]
            )
        else:
            non_final_next_states = torch.empty(
                0, state_batch.shape[1], device=self.device
            )

        # 4. Compute Q(s_t, a) from the main network (State Action Values)
        state_action_values = self.main_net(state_batch).gather(1, action_batch)

        # --- DDQN Q-Value Calculation ---

        # 5. Compute the Target V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if non_final_next_states.numel() > 0:
                # DDQN Logic: Use main_net to SELECT action, target_net to EVALUATE

                # Selection: argmax_a Q(s', a; main)
                # Find the action indices that maximize Q in the MAIN network
                best_actions = (
                    self.main_net(non_final_next_states).argmax(1).unsqueeze(1)
                )

                # Evaluation: Q(s', best_action; target)
                # Use those indices to look up the Q-value from the TARGET network
                next_state_values[non_final_mask] = (
                    self.delayed_net(non_final_next_states)
                    .gather(1, best_actions)
                    .squeeze(1)
                )

        # 6. Compute Expected Q Values (Bellman Target)
        expected_state_action_values = (
            next_state_values * self.discount_factor
        ) + reward_batch

        # 7. Compute Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 8. Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # *** ADDED: GRADIENT CLIPPING FOR STABILITY ***
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), max_norm=100)

        self.optimizer.step()

        # Average Q(s, a) value
        avg_q_value = torch.mean(state_action_values).item()

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
        n_obs: int,
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
        main = CNN(n_obs, num_actions)
        delayed = CNN(n_obs, num_actions)
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


class CartPoleDQNAgent(DQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        n_obs: int,
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
        main = MLP(n_obs, num_actions)
        delayed = MLP(n_obs, num_actions)
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


class CartPoleDDQNAgent(DDQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        n_obs: int,
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
        main = MLP(n_obs, num_actions)
        delayed = MLP(n_obs, num_actions)
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


class MountainCarDQNAgent(DQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        n_obs: int,
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
        main = MLP(n_obs, num_actions)
        delayed = MLP(n_obs, num_actions)
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


class MountainCarDDQNAgent(DDQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        n_obs: int,
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
        main = MLP(n_obs, num_actions)
        delayed = MLP(n_obs, num_actions)
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


class AcrobotDQNAgent(DQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        n_obs: int,
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
        main = MLP(n_obs, num_actions)
        delayed = MLP(n_obs, num_actions)
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


class AcrobotDDQNAgent(DDQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        n_obs: int,
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
        main = MLP(n_obs, num_actions)
        delayed = MLP(n_obs, num_actions)
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
