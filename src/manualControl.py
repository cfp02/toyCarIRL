"""
Manually control the agent to provide expert trajectories for IRL/RL.

This script allows a human expert to control the car in the environment
to generate demonstration trajectories for IRL and RL

Controls:
- LEFT ARROW: Turn left
- RIGHT ARROW: Turn right
- UP ARROW: Move forward (default)
- DOWN ARROW: Exit and save trajectory

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

NUM_STATES = 10
GAMMA = 0.9  # Discount factor
DEFAULT_FPS = 10
MAX_EPISODE_LENGTH = 5000
DATA_DIR = Path("demonstrations")


class TrajectoryRecorder:
    """Records and manages expert demonstration trajectories."""

    def __init__(self, weights: List[float], gamma: float = GAMMA):
        """
        Initialize the trajectory recorder.

        Args:
            weights: The feature weights (used for dimensionality)
            gamma: Discount factor
        """
        self.weights = weights
        self.feature_dim = len(weights)
        self.gamma = gamma
        self.reset()

    def reset(self) -> None:
        """Reset the recorder for a new trajectory."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.features = []
        self.collisions = []  # Track collision count at each step
        self.feature_expectations = np.zeros(self.feature_dim)
        self.prev_feature_expectations = np.zeros(self.feature_dim)
        self.step_count = 0
        self.current_collision_count = 0

    def record_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        features: List[float],
        collision_count: int,
    ) -> None:
        """
        Record a single step in the trajectory.

        Args:
            state: The environment state
            action: The action taken
            reward: The reward received
            features: The feature vector
            collision_count: Current collision count
        """
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.features.append(features.copy())
        self.collisions.append(collision_count)

        # Detect collisions
        if collision_count > self.current_collision_count:
            print(f"Collision #{collision_count} at step {self.step_count}")
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
        trajectory_data = {
            "states": [
                s.tolist() if isinstance(s, np.ndarray) else s for s in self.states
            ],
            "actions": self.actions,
            "rewards": self.rewards,
            "features": self.features,
            "collisions": self.collisions,
            "feature_expectations": self.feature_expectations.tolist(),
            "metadata": {
                "length": self.step_count,
                "total_collisions": self.current_collision_count,
                "gamma": self.gamma,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            },
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(trajectory_data, f, indent=2)

        print(f"Trajectory saved to {filepath}")


def play_and_record(
    obstacle_file: Optional[str] = None,
    output_file: Optional[str] = None,
    fps: int = DEFAULT_FPS,
) -> None:
    """
    Play the game with manual control and record expert trajectory.

    Args:
        obstacle_file: Path to obstacle configuration file
        output_file: Path to save the trajectory data
        fps: Frames per second for game display

    Returns:
        Trajectory data dictionary
    """
    # Initialize weights to all 1.0
    weights = [1.0] * NUM_STATES
    game_state = (
        carmunk.GameState(weights)
        if obstacle_file is None
        else carmunk.GameState(weights, obstacle_file)
    )

    recorder = TrajectoryRecorder(weights)
    reward, state, features, collision_count = game_state.frame_step(
        2
    )  # Start with forward motion

    clock = pygame.time.Clock()
    progress = tqdm(
        total=MAX_EPISODE_LENGTH,
        desc="Recording trajectory",
        unit="steps",
        dynamic_ncols=True,
    )

    # Print instructions
    print("\n--- Manual Control for Expert Demonstrations ---")
    print("Use arrow keys to control the car:")
    print("  LEFT: Turn left")
    print("  RIGHT: Turn right")
    print("  UP: Move forward (default)")
    print("  DOWN: Exit and save trajectory")
    print(f"Recording at {fps} FPS. Max episode length: {MAX_EPISODE_LENGTH} steps.\n")

    # Main game loop
    try:
        running = True
        action = 2  # Default action is forward
        while running and recorder.step_count < MAX_EPISODE_LENGTH:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
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
                elif event.type == pygame.KEYUP:
                    # When key is released, go back to forward motion
                    if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        action = 2

            # Take action in environment - FIX: Unpack 4 values instead of 3
            reward, new_state, features, collision_count = game_state.frame_step(action)

            # Record step
            recorder.record_step(state, action, reward, features, collision_count)
            state = new_state

            # Update progress bar every 10 steps
            if recorder.step_count % 10 == 0:
                progress.n = recorder.step_count
                progress.update(0)
                progress.set_postfix(
                    {
                        "change": f"{recorder.get_change_percentage():.2f}%",
                        "collisions": collision_count,
                    }
                )

            # Calculate and display feature expectation changes periodically
            if recorder.step_count % 100 == 0:
                recorder.get_change_percentage()
                recorder.update_prev_expectations()

            # Control frame rate
            clock.tick(fps)
    finally:
        progress.close()

        # Calculate feature expectations
        if recorder.step_count > 100:
            # Normalize FE to be length-independent
            recorded_steps = recorder.step_count - 100
            norm_factor = (
                (1 - GAMMA) / (1 - GAMMA**recorded_steps) if recorded_steps > 0 else 1
            )
            feature_expectations = recorder.feature_expectations * norm_factor
        else:
            feature_expectations = np.zeros(recorder.feature_dim)

        print("\n=== Feature Expectations for IRL ===")
        print("[")
        for i, fe in enumerate(feature_expectations):
            end_char = "," if i < len(feature_expectations) - 1 else ""
            print(f"    {fe:.8e}{end_char}")
        print("]")
        print("====================================")

        if output_file is None:
            DATA_DIR.mkdir(exist_ok=True)
            track_name = (
                os.path.splitext(os.path.basename(obstacle_file))[0]
                if obstacle_file
                else "default"
            )
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = str(DATA_DIR / f"trajectory_{track_name}_{timestamp}.json")

        # Save trajectory
        recorder.save_trajectory(output_file)

        print(f"\nRecorded {recorder.step_count} steps")
        print(f"Total collisions: {recorder.current_collision_count}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Manually control the car and record expert trajectories for IRL/RL."
    )
    parser.add_argument(
        "--track",
        "-t",
        type=str,
        default=None,
        help="Path or name of obstacle configuration file (default: tracks/default.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save the trajectory data (default: demonstrations/trajectory_<track>_<timestamp>.json)",
    )
    parser.add_argument(
        "--fps",
        "-f",
        type=int,
        default=DEFAULT_FPS,
        help=f"Frames per second for game display (default: {DEFAULT_FPS})",
    )
    args = parser.parse_args()

    # Resolve track path if provided
    obstacle_file = None
    if args.track:
        # Try to locate the track file
        if os.path.isfile(args.track):
            obstacle_file = args.track
        elif os.path.isfile(f"tracks/{args.track}"):
            obstacle_file = f"tracks/{args.track}"
        elif os.path.isfile(f"tracks/{args.track}.json"):
            obstacle_file = f"tracks/{args.track}.json"
        if not obstacle_file:
            print(f"Warning: Could not find track file at {args.track}")
            print("Using default track instead.")

    # Play and record trajectory
    trajectory_data = play_and_record(
        obstacle_file=obstacle_file, output_file=args.output, fps=args.fps
    )
