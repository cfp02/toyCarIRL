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

    def add(self, state, action, reward, next_state, done):
        # Add with max priority when new experience comes in
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities * self.alpha
        probs /= probs.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.frame / self.beta_frames)
        self.frame += 1

        # Calculate weights
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
        learning_rate: float = 2e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 200000,
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
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
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
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        self.policy_net.eval()

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state)

        if not evaluate:
            self.policy_net.train()
        return q_values.argmax().item()

    def update_epsilon(self, frame):
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(
            -1.0 * frame / self.epsilon_decay
        )

    def memorize(self, state, action, reward, next_state, done):
        reward = reward * self.reward_scaling
        self.memory.add(state, action, reward, next_state, done)

    def update_target_network(self):
        tau = 0.05  # Soft update parameter
        for target_param, local_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

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
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.training_steps += 1
        self.update_target_network()

        return loss.mean().item()
