"""
Manually control the agent to provide expert trajectories for IRL/RL.

This script allows a human expert to control the car in the environment
to generate demonstration trajectories for IRL and RL

Controls:
- LEFT ARROW: Turn left
- RIGHT ARROW: Turn right
- UP ARROW: Move forward (default)
- DOWN ARROW: Exit and save trajectory
- R: Reset current demonstration
- Q: Quit collection

Always exit using the DOWN ARROW key rather than Ctrl+C to ensure proper cleanup.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pygame
from tqdm import tqdm

from flat_game import carmunk

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


def play_and_record(
    track_file: Optional[str] = None,
    output_file: Optional[str] = None,
    fps: int = DEFAULT_FPS,
    task_name: str = "pushing",
) -> None:
    """
    Play the game with manual control and record expert trajectories.

    Args:
        track_file: Path to track configuration file
        output_file: Path to save the trajectory data
        fps: Frames per second for game display
        task_name: Name of the task (e.g., "pushing")
    """
    # Initialize weights to all 1.0
    weights = [1.0] * NUM_STATES
    game_state = (
        carmunk.GameState(weights)
        if track_file is None
        else carmunk.GameState(weights, track_file)
    )

    demo_dir = get_demo_dir(task_name)

    # Print instructions
    print("\n--- Manual Control for Expert Demonstrations ---")
    print(f"Task: {task_name}")
    print("Use arrow keys to control the car:")
    print("  LEFT: Turn left")
    print("  RIGHT: Turn right")
    print("  UP: Move forward (default)")
    print("  DOWN: Exit and save trajectory")
    print("  R: Reset current demonstration")
    print("  Q: Quit collection")
    print(f"Recording at {fps} FPS. Max episode length: {MAX_EPISODE_LENGTH} steps.")
    print(f"Demonstrations will be saved to: {demo_dir}\n")

    while True:  # Continue until user quits
        recorder = TrajectoryRecorder(weights, task_name)
        reward, state, features, collision_count = game_state.frame_step(2)  # Start with forward motion

        clock = pygame.time.Clock()
        progress = tqdm(
            total=MAX_EPISODE_LENGTH,
            desc="Recording demonstration",
            unit="steps",
            dynamic_ncols=True,
        )

        # Main game loop
        try:
            running = True
            action = 2  # Default action is forward
            while running and recorder.step_count < MAX_EPISODE_LENGTH:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            action = 1
                        elif event.key == pygame.K_RIGHT:
                            action = 0
                        elif event.key == pygame.K_UP:
                            action = 2
                        elif event.key == pygame.K_DOWN:
                            running = False
                            break
                        elif event.key == pygame.K_r:
                            running = False
                            break
                        elif event.key == pygame.K_q:
                            return
                    elif event.type == pygame.KEYUP:
                        # When key is released, go back to forward motion
                        if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                            action = 2

                # Take action in environment
                reward, new_state, features, collision_count = game_state.frame_step(action)

                # Record step
                recorder.record_step(
                    state, action, features, collision_count, game_state.is_in_goal
                )
                state = new_state

                # Update progress bar every 10 steps
                if recorder.step_count % 10 == 0:
                    progress.n = recorder.step_count
                    progress.update(0)
                    progress.set_postfix(
                        {
                            "features": f"{features[:3]}...",
                            "collisions": collision_count,
                            "in_goal": game_state.is_in_goal,
                        }
                    )

                # Check if demonstration is complete
                if game_state.is_in_goal:
                    print(f"✅ Demonstration completed successfully!")
                    print(f"Debug: is_in_goal={game_state.is_in_goal}, step_count={recorder.step_count}")
                    print(f"Debug: Current collision count: {collision_count}")
                    # Save the trajectory immediately
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = str(demo_dir / f"demo_{timestamp}.json")
                    print(f"Debug: Attempting to save trajectory to {filename}")
                    recorder.save_trajectory(filename)
                    print(f"Debug: Trajectory saved successfully")
                    running = False
                    break

                # Control frame rate
                clock.tick(fps)
        finally:
            progress.close()

        # Reset environment for next demonstration
        game_state = carmunk.GameState(weights, track_file)


def main():
    parser = argparse.ArgumentParser(description="Collect expert demonstrations")
    parser.add_argument(
        "--track",
        type=str,
        default="tracks/pushing.json",
        help="Track configuration file (default: tracks/pushing.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for the trajectory data",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Frames per second (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pushing",
        help="Name of the task (default: pushing)",
    )
    args = parser.parse_args()

    # Collect demonstration
    print(f"Collecting demonstration using track: {args.track}")
    play_and_record(
        track_file=args.track,
        output_file=args.output,
        fps=args.fps,
        task_name=args.task,
    )


if __name__ == "__main__":
    main()
