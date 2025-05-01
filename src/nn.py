import os
import random
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set up device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(DQN, self).__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))  # Add batch normalization
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta_start  # Importance sampling correction
        self.beta_frames = beta_frames  # Frames over which to anneal beta to 1
        self.frame = 1  # Current frame counter
        self.epsilon = 1e-6  # Small positive constant to prevent zero priority
        self.segments = 8

    def add(self, state, action, reward, next_state, done):
        # Add with max priority when new experience comes in
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return (
                torch.tensor(
                    np.array([s[0] for s in self.buffer]), dtype=torch.float32
                ).to(device),
                torch.tensor(
                    np.array([s[1] for s in self.buffer]), dtype=torch.int64
                ).to(device),
                torch.tensor(
                    np.array([s[2] for s in self.buffer]), dtype=torch.float32
                ).to(device),
                torch.tensor(
                    np.array([s[3] for s in self.buffer]), dtype=torch.float32
                ).to(device),
                torch.tensor(
                    np.array([s[4] for s in self.buffer]), dtype=torch.float32
                ).to(device),
                torch.tensor(np.ones(len(self.buffer)), dtype=torch.float32).to(device),
                list(range(len(self.buffer))),
            )

        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Use stratified sampling across segments to ensure diversity
        samples_per_segment = batch_size // self.segments
        remainder = batch_size % self.segments

        indices = []
        segment_size = len(self.buffer) // self.segments

        for i in range(self.segments):
            # Get segment boundaries
            start_idx = i * segment_size
            end_idx = (
                (i + 1) * segment_size if i < self.segments - 1 else len(self.buffer)
            )

            # Calculate segment probabilities
            segment_probs = probs[start_idx:end_idx]
            if segment_probs.sum() > 0:  # Avoid division by zero
                segment_probs = segment_probs / segment_probs.sum()

                # Sample from this segment
                segment_samples = samples_per_segment + (1 if i < remainder else 0)
                if segment_samples > 0:
                    segment_indices = np.random.choice(
                        np.arange(start_idx, end_idx),
                        size=min(segment_samples, end_idx - start_idx),
                        p=segment_probs,
                        replace=False,
                    )
                    indices.extend(segment_indices)

        # If we couldn't get enough samples with stratification, fill the rest randomly
        if len(indices) < batch_size:
            remaining = batch_size - len(indices)
            additional_indices = np.random.choice(
                len(self.buffer), size=remaining, p=probs, replace=False
            )
            indices.extend(additional_indices)

        # Get samples based on indices
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.frame / self.beta_frames)
        self.frame += 1

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(device),
            torch.tensor(np.array(actions), dtype=torch.int64).to(device),
            torch.tensor(np.array(rewards), dtype=torch.float32).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(np.array(dones), dtype=torch.float32).to(device),
            torch.tensor(weights, dtype=torch.float32).to(device),
            indices,
        )

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + self.epsilon

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 0.8,
        epsilon_end: float = 0.1,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        use_dueling_dqn: bool = True,
        reward_scaling: float = 1.0,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_dueling_dqn = use_dueling_dqn
        self.reward_scaling = reward_scaling

        # Initialize networks
        self.policy_net = DQN(state_size, hidden_sizes, action_size).to(device)
        self.target_net = DQN(state_size, hidden_sizes, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction="none")

        # Initialize replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_size)

        self.training_steps = 0

    def save(self, path):
        """Save the model to a file"""
        try:
            torch.save(
                {
                    "policy_net": self.policy_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                    "training_steps": self.training_steps,
                },
                path,
            )
            return True
        except Exception as e:
            print(f"Warning: Error saving model: {e}")
            torch.save(
                {
                    "policy_net": self.policy_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epsilon": float(self.epsilon),
                    "training_steps": int(self.training_steps),
                },
                path,
            )
            return True

    def load(self, path):
        """Load the model from a file"""
        if not os.path.exists(path):
            print(f"Checkpoint file {path} not found")
            return False

        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint["epsilon"]
            self.training_steps = checkpoint["training_steps"]
            print(f"Successfully loaded model from {path}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            try:
                print("Attempting to load model weights only...")
                checkpoint = torch.load(path, map_location=device, weights_only=True)
                self.policy_net.load_state_dict(checkpoint["policy_net"])
                self.target_net.load_state_dict(checkpoint["target_net"])
                print("Successfully loaded model weights (without optimizer state)")
                return True
            except Exception as e2:
                print(f"Error loading model weights: {e2}")
                return False

    def act(self, state, evaluate=False):
        # Convert numpy array to tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)

        # Temporarily set policy_net to eval mode for action selection
        with torch.no_grad():
            training_mode = self.policy_net.training  # Save current mode
            self.policy_net.eval()  # Set to evaluation mode

            if state.dim() == 1:
                state = state.unsqueeze(0)

            # Use epsilon-greedy policy
            if random.random() > self.epsilon or evaluate:
                q_values = self.policy_net(state)
                action = torch.argmax(q_values).item()
            else:
                action = random.randrange(self.action_size)

            # Restore previous training mode
            self.policy_net.train(training_mode)

            return action

    def update_epsilon(self, current_episode, total_episodes):
        """
        Update epsilon based on the current episode number.

        Args:
            current_episode: The current episode number
            total_episodes: Total number of episodes for training
        """
        # Ensure smooth decay from start to end over the course of training
        progress = min(1.0, current_episode / total_episodes)

        # Linear interpolation between epsilon_start and epsilon_end
        self.epsilon = self.epsilon_start - progress * (
            self.epsilon_start - self.epsilon_end
        )

        # Ensure we don't go below the minimum value
        self.epsilon = max(self.epsilon_end, self.epsilon)

    def memorize(self, state, action, reward, next_state, done):
        reward = reward * self.reward_scaling
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0

        # Sample from buffer
        states, actions, rewards, next_states, dones, weights, indices = (
            self.memory.sample(self.batch_size)
        )

        # Get current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get next Q values
        # select action using policy net, evaluate using target net
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(q_values, target_q_values.detach())

        # Apply importance sampling weights if using prioritized replay
        weighted_loss = (loss * weights).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.mean().item()
