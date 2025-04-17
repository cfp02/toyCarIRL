import os
import random
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import your car environment

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define DQN model using PyTorch
class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(DQN, self).__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Fixed memory buffer implementation


class ReplayBuffer:
    """Standard experience replay buffer"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample random batch from buffer"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(device),
            torch.tensor(np.array(actions), dtype=torch.int64).to(device),
            torch.tensor(np.array(rewards), dtype=torch.float32).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(np.array(dones), dtype=torch.float32).to(device),
            torch.ones(batch_size).to(device),  # Uniform weights
            None,  # No indices needed
        )

    def update_priorities(self, indices, errors):
        """No-op for standard replay buffer (only used by prioritized replay)"""
        pass

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use
        self.beta = beta  # Importance sampling weight
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with maximum priority"""
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample batch based on priorities"""
        if self.size < batch_size:
            indices = np.random.choice(self.size, batch_size, replace=True)
            # Define uniform probabilities in this case
            probabilities = np.ones(self.size) / self.size
        else:
            priorities = self.priorities[: self.size]
            probabilities = priorities**self.alpha
            probabilities /= probabilities.sum()

            indices = np.random.choice(self.size, batch_size, p=probabilities)

        # Importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        samples = [self.buffer[idx] for idx in indices]

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
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Small constant for stability

    def __len__(self):
        return self.size


# Agent class
class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 200000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        use_prioritized_replay: bool = True,
        use_double_dqn: bool = True,
        use_dueling_dqn: bool = False,
        reward_scaling: float = 1.0,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_prioritized_replay = use_prioritized_replay
        self.use_double_dqn = use_double_dqn
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
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size)
        else:
            self.memory = ReplayBuffer(buffer_size)

        self.training_steps = 0

    def save(self, path):
        """Save the model to a file"""
        # Try to handle NumPy scalar serialization issues
        try:
            # Just save directly without custom serialization handlers
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
            # Convert any problematic NumPy values to Python native types
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
            # Explicitly set weights_only=False to maintain compatibility
            # Only do this with trusted checkpoint files
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
            # Fall back to loading just the model weights if full loading fails
            try:
                print("Attempting to load model weights only...")
                checkpoint = torch.load(path, map_location=device, weights_only=True)
                self.policy_net.load_state_dict(checkpoint["policy_net"])
                self.target_net.load_state_dict(checkpoint["target_net"])
                # Can't load optimizer state and other components in weights_only mode
                print("Successfully loaded model weights (without optimizer state)")
                return True
            except Exception as e2:
                print(f"Error loading model weights: {e2}")
                return False

    def act(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def update_epsilon(self, frame):
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(
            -1.0 * frame / self.epsilon_decay
        )

    def memorize(self, state, action, reward, next_state, done):
        reward = reward * self.reward_scaling  # Scale the reward
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0

        # Sample from buffer - both buffer types return the same format
        states, actions, rewards, next_states, dones, weights, indices = (
            self.memory.sample(self.batch_size)
        )

        # Get current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get next Q values
        if self.use_double_dqn:
            # Double DQN: select action using policy net, evaluate using target net
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = (
                self.target_net(next_states).gather(1, next_actions).squeeze(1)
            )
        else:
            # Regular DQN
            next_q_values = self.target_net(next_states).max(1)[0]

        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(q_values, target_q_values.detach())

        # Apply importance sampling weights if using prioritized replay
        weighted_loss = (loss * weights).mean()

        # Update priorities in the replay buffer if using prioritized replay
        if self.use_prioritized_replay and indices is not None:
            with torch.no_grad():
                td_errors = torch.abs(q_values - target_q_values).cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 1.0
        )  # Gradient clipping
        self.optimizer.step()

        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.mean().item()
