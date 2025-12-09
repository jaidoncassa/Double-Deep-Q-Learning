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


class LazyFrames:
    """Optimizes memory by avoiding duplicated frame storage in frame stacks."""

    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.stack(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]


#####################################################################
# Main superclasses
#####################################################################
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
        warmup_amt: int,
        model_lr: float,
        main_net,
        delayed_net,
        seed: int,
        max_clip: int,
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
        self.warmup_amt = warmup_amt

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
        self.max_clip = max_clip

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

        # Process States, Actions, and Rewards
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
        state_batch = torch.stack([self.transform(s).squeeze(0) for s in batch.state])
        action_batch = torch.tensor(
            batch.action, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        )

        # Transform and concatenate non-final next states
        if non_final_next_states_np:
            non_final_next_states = torch.stack(
                [self.transform(s).squeeze(0) for s in non_final_next_states_np]
            )
        else:
            non_final_next_states = torch.empty(
                0, state_batch.shape[1], device=self.device
            )  # Handle empty case

        # Q-Value Calculation
        # Compute Q(s_t, a) from the main network (State Action Values)
        q_values_all = self.main_net(state_batch)  # shape [B, A]
        state_action_values = q_values_all.gather(1, action_batch)

        # For logging purposes, I want to log the maximization bias by observing the max action from the current main network
        with torch.no_grad():
            mean_max_q = q_values_all.max(1).values.mean().item()

        # Compute the Target V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # Now perform max_a with target network: MAXIMIZATION BIAS
            if non_final_next_states.numel() > 0:
                # DQN logic: max_a Q(s', a; target)
                next_state_values[non_final_mask] = (
                    self.delayed_net(non_final_next_states).max(1).values
                )

        # Compute Expected Q Values (Bellman Target)
        # Expected Q = R + gamma * V(s_{t+1})
        expected_state_action_values = (
            next_state_values * self.discount_factor
        ) + reward_batch

        # Compute Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # ADDED: GRADIENT CLIPPING FOR STABILITY
        torch.nn.utils.clip_grad_norm_(
            self.main_net.parameters(), max_norm=self.max_clip
        )

        self.optimizer.step()

        # Average Q(s, a) value
        avg_q_value = torch.mean(state_action_values).item()

        # Return the loss and the average Q-value for logging
        return loss.item(), avg_q_value, mean_max_q

    def reset_target_network(self):
        # Soft update of target network's weights
        TAU = 0.005
        with torch.no_grad():
            for target_param, param in zip(
                self.delayed_net.parameters(), self.main_net.parameters()
            ):
                target_param.data.copy_(
                    TAU * param.data + (1 - TAU) * target_param.data
                )

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
        self.push_transition(obs)

        # Turn into Torcher vector
        input = self.transform(state)
        with torch.no_grad():
            # Predict the Q(s, a) values!
            q_values = self.main_net(input)

        # Get the action and return it to the main controller
        action = self.get_action(q_values)

        # Update the weights of the Neural Network every n steps past 32 steps
        # Include a warmup
        q = None
        loss = None
        mean_max_q = None
        if (
            self.frame_count > self.warmup_amt
            and self.frame_count % self.update_frequency == 0
            and len(self.replay_buffer) > self.batch_size
        ):
            loss, q, mean_max_q = self.update_cnn_weights()

        # Reset the delayed network every m steps
        if (
            self.frame_count > self.warmup_amt
            and self.frame_count > 0
            and self.frame_count % self.update_target_frequency == 0
        ):
            self.reset_target_network()

        # Update trackers
        self.prev_state = state  # s
        self.prev_action = action  # a'

        return action, loss, q, mean_max_q

    def agent_end(self, reward):
        """Run when the agent terminates

        Args:
            reward: the reward the agent recieved for entering the terminal state.
        """
        # Save the full observation (s, a, R, s')
        obs = (self.prev_state, self.prev_action, reward, None, True)
        self.push_transition(obs)

        # Update the weights of the Neural Network every n steps past batch_size steps
        q = None
        loss = None
        mean_max_q = None
        if (
            self.frame_count % self.update_frequency == 0
            and len(self.replay_buffer) > self.batch_size
        ):
            loss, q, mean_max_q = self.update_cnn_weights()

        # Reset the delayed network every m steps
        if (
            self.frame_count > 0
            and self.frame_count % self.update_target_frequency == 0
        ):
            self.reset_target_network()

        self.prev_state = None
        self.prev_action = None
        return loss, q, mean_max_q

    def push_transition(self, transition):
        # Save the full observation (s, a, R, s', done)
        self.replay_buffer.append(transition)

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
        warmup_amt: int,
        model_lr: float,
        main_net,
        delayed_net,
        seed: int,
        max_norm: int,
        nstep_buffer_size=None,
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
            warmup_amt,
            model_lr,
            main_net,
            delayed_net,
            seed,
            max_norm,
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
        state_batch = torch.stack([self.transform(s).squeeze(0) for s in batch.state])
        action_batch = torch.tensor(
            batch.action, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        )

        # Transform and concatenate non-final next states
        if non_final_next_states_np:
            non_final_next_states = torch.stack(
                [self.transform(s).squeeze(0) for s in non_final_next_states_np]
            )
        else:
            non_final_next_states = torch.empty(
                0, state_batch.shape[1], device=self.device
            )

        # Compute Q(s_t, a) from the main network (State Action Values)
        q_values_all = self.main_net(state_batch)  # shape [B, A]
        state_action_values = q_values_all.gather(1, action_batch)

        # For loggic purposes, I want to log the maximization bias by observing the max action from the current main network
        with torch.no_grad():
            mean_max_q = q_values_all.max(1).values.mean().item()

        # DDQN Q-Value Calculation

        # Compute the Target V(s_{t+1}) for all next states
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

        # Compute Expected Q Values (Bellman Target) R + γQ(s_t+1, argmaxQ(s_t+1, a'; 𝜃); 𝜃_-1)
        # Must t
        expected_state_action_values = (
            next_state_values * self.discount_factor
        ) + reward_batch

        # Compute Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # ADDED: GRADIENT CLIPPING FOR STABILITY
        torch.nn.utils.clip_grad_norm_(
            self.main_net.parameters(), max_norm=self.max_clip
        )

        self.optimizer.step()

        # Average Q(s, a) value
        avg_q_value = torch.mean(state_action_values).item()

        # Return the loss and the average Q-value for logging
        return loss.item(), avg_q_value, mean_max_q


class nStepDDQN(DDQN):
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
        warmup_amt: int,
        model_lr: float,
        main_net,
        delayed_net,
        seed: int,
        max_norm: int,
        nstep_buffer_size=None,
    ):
        # Maintain two buffers now, think of the nstep_buffer as a sliding window
        self.nstep_buffer = deque(maxlen=nstep_buffer_size)
        self.n = nstep_buffer_size
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
            warmup_amt,
            model_lr,
            main_net,
            delayed_net,
            seed,
            max_norm,
        )

    def update_cnn_weights(self):
        # Should never happen because we have a check already, but lets be safe
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0

        # Randomly sample from reply buffer of the form (theta(s), a, G_sum, theta(s'_t+n+1))
        samples = random.sample(self.replay_buffer, k=self.batch_size)
        batch = self.Transition(*zip(*samples))

        # Process States, Actions, and Rewards
        # Identify non-final next states for mask
        non_final_next_states_np = [s for s in batch.next_state if s is not None]

        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device,
            dtype=torch.bool,
        )

        # Transform and concatenate all batch elements
        state_batch = torch.stack([self.transform(s).squeeze(0) for s in batch.state])
        action_batch = torch.tensor(
            batch.action, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        )

        # Transform and concatenate non-final next states
        if non_final_next_states_np:
            non_final_next_states = torch.stack(
                [self.transform(s).squeeze(0) for s in non_final_next_states_np]
            )
        else:
            non_final_next_states = torch.empty(
                0, state_batch.shape[1], device=self.device
            )

        # Compute Q(s_t, a) from the main network (State Action Values)
        q_values_all = self.main_net(state_batch)  # shape [B, A]
        q_sa = q_values_all.gather(1, action_batch)

        # For loggic purposes, I want to log the maximization bias by observing the max action from the current main network
        with torch.no_grad():
            mean_max_q = q_values_all.max(1).values.mean().item()

        # DDQN Q-Value Calculation

        # Compute the Target V(s_{t+1}) for all next states
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

        # Calculate gamma^n
        gamma_n = self.discount_factor**self.n

        # Compute Expected Q Values (Bellman Target) G_sum + (γ**n)Q(s_t+n+1, argmaxQ(s_t+n+1, a'; 𝜃); 𝜃_-1)
        # We must rememeber for nstep bootstrapping, we must weight the Q function of our target network
        # by γ**n, not just weighting by γ.
        targets = reward_batch + gamma_n * next_state_values

        # For terminal states: remove bootstrapping
        targets[~non_final_mask] = reward_batch[~non_final_mask]

        # Compute Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_sa, targets.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # ADDED: GRADIENT CLIPPING FOR STABILITY ***
        torch.nn.utils.clip_grad_norm_(
            self.main_net.parameters(), max_norm=self.max_clip
        )

        self.optimizer.step()

        # Average Q(s, a) value
        avg_q_value = torch.mean(q_sa).item()

        # Return the loss and the average Q-value for logging
        return loss.item(), avg_q_value, mean_max_q

    def push_transition(self, transition):
        self.nstep_buffer.append(transition)

        # If terminal, flush shorter ones
        if transition[-1]:
            self.flush_nstep_buffer()
            return

        if len(self.nstep_buffer) == self.nstep_buffer.maxlen:
            # Create full n-step transition
            self.create_and_store_nstep()

    def create_and_store_nstep(self):
        # window length (may be < n during flush)
        k = len(self.nstep_buffer)

        s0, a0 = self.nstep_buffer[0][0], self.nstep_buffer[0][1]

        rewards = [tr[2] for tr in self.nstep_buffer]
        G = sum([self.discount_factor**i * rewards[i] for i in range(k)])

        _, _, _, s_k, done_k = self.nstep_buffer[-1]

        # store (s0, a0, G, s_k, done)
        self.replay_buffer.append((s0, a0, G, s_k, done_k))

    def flush_nstep_buffer(self):
        while len(self.nstep_buffer) > 0:
            self.create_and_store_nstep()
            self.nstep_buffer.popleft()


#####################################################################
# Atari subclasses
#####################################################################
class AtariDQNAgent(DQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        n_obs: int | None,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        buffer_size: int,
        batch_size: int,
        update_frequency: int,
        update_target_frequency: int,
        warmup_amt: int,
        model_lr: float,
        seed: int,
        max_clip: int,
        nstep_buffer_size=None,
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
            warmup_amt,
            model_lr,
            main,
            delayed,
            seed,
            max_clip,
        )

    def transform(self, state):
        # Convert LazyFrames to numpy array automatically
        if isinstance(state, LazyFrames):
            state = np.array(state)

        # Now state is np.ndarray shape (4,84,84)
        return torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0

    def reset_target_network(self):
        self.delayed_net.load_state_dict(self.main_net.state_dict())

    def set_epsilon(self):
        # Linear decays epsilon based on the number of frames processed.
        self.eps_threshold = (
            self.final_epsilon
            + (self.initial_epsilon - self.final_epsilon)
            * max(0, (self.epsilon_decay - self.frame_count))
            / self.epsilon_decay
        )

    def push_transition(self, transition):
        # transition = (state, action, reward, next_state, done)
        state, action, reward, next_state, done = transition

        # Wrap the state using LazyFrames
        if state is not None:
            state = LazyFrames(list(state))
        if next_state is not None:
            next_state = LazyFrames(list(next_state))

        self.replay_buffer.append((state, action, reward, next_state, done))


class AtariDDQNAgent(DDQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        n_obs: int | None,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        buffer_size: int,
        batch_size: int,
        update_frequency: int,
        update_target_frequency: int,
        warmup_amt: int,
        model_lr: float,
        seed: int,
        max_clip: int,
        nstep_buffer_size=None,
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
            warmup_amt,
            model_lr,
            main,
            delayed,
            seed,
            max_clip,
        )

    def transform(self, state):
        # Convert LazyFrames to numpy array automatically
        if isinstance(state, LazyFrames):
            state = np.array(state)

        # Now state is np.ndarray shape (4,84,84)
        return torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0

    def set_epsilon(self):
        # Linear decays epsilon based on the number of frames processed.
        self.eps_threshold = (
            self.final_epsilon
            + (self.initial_epsilon - self.final_epsilon)
            * max(0, (self.epsilon_decay - self.frame_count))
            / self.epsilon_decay
        )

    def reset_target_network(self):
        self.delayed_net.load_state_dict(self.main_net.state_dict())

    def push_transition(self, transition):
        # transition = (state, action, reward, next_state, done)
        state, action, reward, next_state, done = transition

        # Wrap the state using LazyFrames
        if state is not None:
            state = LazyFrames(list(state))
        if next_state is not None:
            next_state = LazyFrames(list(next_state))

        self.replay_buffer.append((state, action, reward, next_state, done))


class AtariNStepDDQNAgent(nStepDDQN):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        n_obs: int | None,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        buffer_size: int,
        batch_size: int,
        update_frequency: int,
        update_target_frequency: int,
        warmup_amt: int,
        model_lr: float,
        seed: int,
        max_clip: int,
        nstep_buffer_size: int,
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
            warmup_amt,
            model_lr,
            main,
            delayed,
            seed,
            max_clip,
            nstep_buffer_size,
        )

    def transform(self, state):
        # Convert LazyFrames to numpy array automatically
        if isinstance(state, LazyFrames):
            state = np.array(state)

        assert state.shape == (4, 84, 84), f"BAD SHAPE: {state.shape}"

        # Now state is np.ndarray shape (4,84,84)
        return torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0

    def set_epsilon(self):
        # Linear decays epsilon based on the number of frames processed.
        self.eps_threshold = (
            self.final_epsilon
            + (self.initial_epsilon - self.final_epsilon)
            * max(0, (self.epsilon_decay - self.frame_count))
            / self.epsilon_decay
        )

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

        # Converts states into LazyFrames objects
        self.push_transition(obs)

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
        mean_max_q = None
        # Added a new warmup mechanism
        if (
            self.frame_count > self.warmup_amt
            and self.frame_count % self.update_frequency == 0
            and len(self.replay_buffer) > self.batch_size
        ):
            loss, q, mean_max_q = self.update_cnn_weights()

        # Reset the delayed network every m steps
        if (
            self.frame_count > self.warmup_amt
            and self.frame_count > 0
            and self.frame_count % self.update_target_frequency == 0
        ):
            self.reset_target_network()

        # Update trackers
        self.prev_state = state  # s
        self.prev_action = action  # a'

        return action, loss, q, mean_max_q

    def agent_end(self, reward):
        """Run when the agent terminates

        Args:
            reward: the reward the agent recieved for entering the terminal state.
        """
        # Save the full observation (s, a, R, s')
        obs = (self.prev_state, self.prev_action, reward, None, True)
        self.push_transition(obs)

        # Update the weights of the Neural Network every n steps past batch_size steps
        q = None
        loss = None
        mean_max_q = None
        # Added a new warmup mechanism
        if (
            self.frame_count > self.warmup_amt
            and self.frame_count % self.update_frequency == 0
            and len(self.replay_buffer) > self.batch_size
        ):
            loss, q, mean_max_q = self.update_cnn_weights()

        # Reset the delayed network every m steps
        if (
            self.frame_count > self.warmup_amt
            and self.frame_count > 0
            and self.frame_count % self.update_target_frequency == 0
        ):
            self.reset_target_network()

        self.prev_state = None
        self.prev_action = None
        return loss, q, mean_max_q

    def reset_target_network(self):
        self.delayed_net.load_state_dict(self.main_net.state_dict())

    def push_transition(self, transition):
        # transition = (state, action, reward, next_state, done)
        state, action, reward, next_state, done = transition

        # Wrap the state using LazyFrames
        if state is not None:
            state = LazyFrames(list(state))
        if next_state is not None:
            next_state = LazyFrames(list(next_state))

        # Store the wrapped transition
        new_transition = (state, action, reward, next_state, done)
        self.nstep_buffer.append(new_transition)

        # If terminal, flush shorter ones
        if done:
            self.flush_nstep_buffer()
            return

        if len(self.nstep_buffer) == self.nstep_buffer.maxlen:
            # Create full n-step transition
            self.create_and_store_nstep()


#####################################################################
# Control Environment Agents (CartPole, MountainCar, Acrobot)
#####################################################################
class ControlDQN(DQN):
    """DQN agent for classic control environments using MLP."""

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
        warmup_amt: int,
        model_lr: float,
        seed: int,
        max_clip: int,
        nstep_buffer_size=None,
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
            warmup_amt,
            model_lr,
            main,
            delayed,
            seed,
            max_clip,
        )


class ControlDDQN(DDQN):
    """DDQN agent for classic control environments using MLP."""

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
        warmup_amt: int,
        model_lr: float,
        seed: int,
        max_clip: int,
        nstep_buffer_size=None,
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
            warmup_amt,
            model_lr,
            main,
            delayed,
            seed,
            max_clip,
        )


class ControlNStepDDQN(nStepDDQN):
    """N-Step DDQN agent for classic control environments using MLP."""

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
        warmup_amt: int,
        model_lr: float,
        seed: int,
        max_clip: int,
        nstep_buffer_size: int,
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
            warmup_amt,
            model_lr,
            main,
            delayed,
            seed,
            max_clip,
            nstep_buffer_size,
        )
