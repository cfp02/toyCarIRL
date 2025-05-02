"""
Utility functions and classes for the car environment.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

NUM_STATES = 16  # Updated for pushing task
GAMMA = 0.9  # Discount factor
DEFAULT_FPS = 30  # Increased for smoother control
MAX_EPISODE_LENGTH = 5000
DATA_DIR = Path("demonstrations")


def get_demo_dir(task_name: str) -> Path:
    """Get the directory path for a specific task's demonstrations."""
    demo_dir = DATA_DIR / task_name
    demo_dir.mkdir(parents=True, exist_ok=True)
    return demo_dir


class TrajectoryRecorder:
    """Records and manages expert demonstration trajectories."""

    def __init__(self, weights: List[float], task_name: str, gamma: float = GAMMA):
        """
        Initialize the trajectory recorder.

        Args:
            weights: The feature weights (used for dimensionality)
            task_name: Name of the task (e.g., "pushing")
            gamma: Discount factor
        """
        self.weights = weights
        self.feature_dim = len(weights)
        self.gamma = gamma
        self.task_name = task_name
        self.reset()

    def reset(self) -> None:
        """Reset the recorder for a new trajectory."""
        self.states = []
        self.actions = []
        self.features = []
        self.collisions = []  # Track collision count at each step
        self.feature_expectations = np.zeros(self.feature_dim)
        self.prev_feature_expectations = np.zeros(self.feature_dim)
        self.step_count = 0
        self.current_collision_count = 0
        self.is_in_goal = False

    def record_step(
        self,
        state: np.ndarray,
        action: int,
        features: List[float],
        collision_count: int,
        is_in_goal: bool,
    ) -> None:
        """
        Record a single step in the trajectory.

        Args:
            state: The environment state
            action: The action taken
            features: The feature vector
            collision_count: Current collision count
            is_in_goal: Whether the object is in the goal zone
        """
        self.states.append(state.copy())
        self.actions.append(action)
        self.features.append(features.copy())
        self.collisions.append(collision_count)
        self.is_in_goal = is_in_goal

        # Detect collisions
        if collision_count > self.current_collision_count:
            self.current_collision_count = collision_count

        self.step_count += 1

        # Update feature expectations
        if self.step_count > 100:
            self.feature_expectations += (
                self.gamma ** (self.step_count - 101)
            ) * np.array(features)

    def get_change_percentage(self) -> float:
        """Calculate the percentage change in feature expectations."""
        norm_fe = np.linalg.norm(self.feature_expectations)
        if norm_fe > 0:
            change = (
                np.linalg.norm(
                    self.feature_expectations - self.prev_feature_expectations
                )
                * 100.0
            ) / norm_fe
        else:
            change = 0.0
        return float(change)

    def update_prev_expectations(self) -> None:
        """Update previous feature expectations for change calculation."""
        self.prev_feature_expectations = np.array(self.feature_expectations)

    def save_trajectory(self, filepath: str) -> None:
        """
        Save the collected trajectory to a file.

        Args:
            filepath: Path to save the trajectory data
        """
        # Generate timestamp-based filename if no specific filepath is provided
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            demo_dir = get_demo_dir(self.task_name)
            filepath = str(demo_dir / f"demo_{timestamp}.json")

        trajectory_data = {
            "states": [
                s.tolist() if isinstance(s, np.ndarray) else s for s in self.states
            ],
            "actions": self.actions,
            "features": self.features,
            "collisions": self.collisions,
            "feature_expectations": self.feature_expectations.tolist(),
            "metadata": {
                "length": self.step_count,
                "total_collisions": self.current_collision_count,
                "is_in_goal": self.is_in_goal,
                "gamma": self.gamma,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "task": self.task_name,
            },
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(trajectory_data, f, indent=2)

        print(f"Trajectory saved to {filepath}") 