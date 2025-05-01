"""
Manually control the agent to provide expert trajectories for IRL/RL.

This script allows a human expert to control the car in the environment
to generate demonstration trajectories for IRL and RL

Controls:
- LEFT ARROW: Turn left
- RIGHT ARROW: Turn right
- UP ARROW: Move forward (default)
- DOWN ARROW: Exit current episode
- ESC: Stop recording episodes and save aggregated data

Always exit using the ESC key rather than Ctrl+C to ensure proper cleanup.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pygame
from tqdm import tqdm

from flat_game import carmunk

NUM_STATES = 9
GAMMA = 0.9  # Discount factor
DEFAULT_FPS = 9
MAX_EPISODE_LENGTH = 500
DEFAULT_NUM_EPISODES = 5
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

        # Multi-episode tracking
        self.all_episodes = []
        self.combined_feature_expectations = np.zeros(self.feature_dim)
        self.episode_count = 0

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
            # print(f"Collision #{collision_count} at step {self.step_count}")
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

    def complete_episode(self) -> Dict:
        """
        Complete the current episode and return the trajectory data.

        Returns:
            Dict containing the episode trajectory data
        """
        # Calculate normalized feature expectations for this episode
        if self.step_count > 100:
            # Normalize FE to be length-independent
            recorded_steps = self.step_count - 100
            norm_factor = (
                (1 - self.gamma) / (1 - self.gamma**recorded_steps)
                if recorded_steps > 0
                else 1
            )
            episode_fe = self.feature_expectations * norm_factor
        else:
            episode_fe = np.zeros(self.feature_dim)

        # Create episode data dictionary
        episode_data = {
            "states": [
                s.tolist() if isinstance(s, np.ndarray) else s for s in self.states
            ],
            "actions": self.actions,
            "rewards": self.rewards,
            "features": self.features,
            "collisions": self.collisions,
            "ep_feature_expectations": episode_fe.tolist(),
            "metadata": {
                "episode_number": self.episode_count + 1,
                "length": self.step_count,
                "total_collisions": self.current_collision_count,
                "gamma": self.gamma,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            },
        }

        # Add to our collection of episodes
        self.all_episodes.append(episode_data)

        # Update combined feature expectations (we'll average at the end)
        self.combined_feature_expectations += episode_fe
        self.episode_count += 1

        return episode_data

    def save_all_trajectories(self, filepath: str) -> None:
        """
        Save all collected trajectories and average feature expectations.

        Args:
            filepath: Path to save the trajectory data
        """
        if self.episode_count == 0:
            print("No episodes recorded.")
            return

        # Calculate average feature expectations
        avg_fe = self.combined_feature_expectations / self.episode_count

        # Create aggregate data
        all_trajectory_data = {
            "episodes": self.all_episodes,
            "feature_expectations": avg_fe.tolist(),
            "metadata": {
                "total_episodes": self.episode_count,
                "gamma": self.gamma,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            },
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(all_trajectory_data, f, indent=2)

        print(f"\nAll trajectories saved to {filepath}")

        # Also print the average feature expectations for easy copying
        print("\n=== Average Feature Expectations ===")
        print("[")
        for i, fe in enumerate(avg_fe):
            end_char = "," if i < len(avg_fe) - 1 else ""
            print(f"    {fe:.8e}{end_char}")
        print("]")
        print("====================================")


def play_and_record(
    track_file: str,
    behavior: Optional[str] = "Custom",
    fps: int = DEFAULT_FPS,
    num_episodes: int = DEFAULT_NUM_EPISODES,
) -> None:
    """
    Play the game with manual control and record expert trajectory across multiple episodes.

    Args:
        track_file: Path to obstacle configuration file
        behavior: name of behavior demonstration
        fps: Frames per second for game display
        num_episodes: Number of episodes to record
    """
    # Initialize weights to all 1.0
    weights = [1.0] * NUM_STATES
    game_state = (
        carmunk.GameState(weights)
        if track_file is None
        else carmunk.GameState(weights, track_file)
    )

    recorder = TrajectoryRecorder(weights)
    clock = pygame.time.Clock()

    # Print instructions
    print("\n--- Manual Control for Expert Demonstrations ---")
    print("Use arrow keys to control the car:")
    print("  LEFT: Turn left")
    print("  RIGHT: Turn right")
    print("  UP: Move forward (default)")
    print("  DOWN: Exit current episode")
    print("  ESC: Stop recording episodes and save aggregated data")
    print(f"Recording at {fps} FPS. Max episode length: {MAX_EPISODE_LENGTH} steps.")
    print(f"Target number of episodes: {num_episodes}\n")

    # Main loop for multiple episodes
    stop_all_recording = False
    completed_episodes = 0

    while completed_episodes < num_episodes and not stop_all_recording:
        # Reset for new episode
        recorder.reset()
        game_state.reset_car()
        reward, state, features, collision_count, done = game_state.frame_step(
            2
        )  # Start with forward

        print(f"\nStarting Episode {completed_episodes + 1}/{num_episodes}")

        progress = tqdm(
            total=MAX_EPISODE_LENGTH,
            desc=f"Episode {completed_episodes + 1}/{num_episodes}",
            unit="steps",
            dynamic_ncols=True,
        )

        # Episode game loop
        try:
            running = True
            action = 2  # Default action is forward
            while running and recorder.step_count < MAX_EPISODE_LENGTH:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        stop_all_recording = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            action = 1
                        elif event.key == pygame.K_RIGHT:
                            action = 0
                        elif event.key == pygame.K_UP:
                            action = 2
                        elif event.key == pygame.K_DOWN:
                            running = False  # End this episode
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                            stop_all_recording = True  # End all recording
                    elif event.type == pygame.KEYUP:
                        # When key is released, go back to forward motion
                        if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                            action = 2

                # Take action in environment
                reward, new_state, features, collision_count, done = (
                    game_state.frame_step(action)
                )

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

                if done:
                    break

        finally:
            progress.close()

            # Complete the episode
            if recorder.step_count > 0:
                recorder.complete_episode()
                completed_episodes += 1
                print(
                    f"\nEpisode {completed_episodes} completed with {recorder.step_count} steps"
                )
                print(f"Collisions this episode: {recorder.current_collision_count}")

    # Save all trajectories
    DATA_DIR.mkdir(exist_ok=True)
    track_name = os.path.basename(track_file)
    if track_name.endswith(".json"):
        track_name = track_name[:-5]  # Remove .json extension

    # Create output filename
    behavior_str = behavior if behavior else "custom"
    output_file = str(DATA_DIR / f"{behavior_str}_{track_name}_demo.json")

    # Save trajectories
    recorder.save_all_trajectories(output_file)

    print(f"\nRecorded {completed_episodes} episodes")
    print("Average feature expectations saved for expert policy")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Manually control the car and record expert trajectories for IRL/RL across multiple episodes."
    )
    parser.add_argument(
        "--track",
        "-t",
        type=str,
        default=None,
        help="Path or name of obstacle configuration file (default: tracks/default.json)",
    )
    parser.add_argument(
        "--behavior",
        "-b",
        type=str,
        default=None,
        help="Behavior name for saving the trajectory data",
    )
    parser.add_argument(
        "--fps",
        "-f",
        type=int,
        default=DEFAULT_FPS,
        help=f"Frames per second for game display (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help=f"Number of episodes to record (default: {DEFAULT_NUM_EPISODES})",
    )
    args = parser.parse_args()

    # Resolve track path if provided
    track_file = "tracks/default.json"
    if args.track:
        # Try to locate the track file
        if os.path.isfile(args.track):
            track_file = args.track
        elif os.path.isfile(f"tracks/{args.track}"):
            track_file = f"tracks/{args.track}"
        elif os.path.isfile(f"tracks/{args.track}.json"):
            track_file = f"tracks/{args.track}.json"
        if not track_file:
            print(f"Warning: Could not find track file at {args.track}")
            print("Using default track instead.")

    # Play and record trajectory
    play_and_record(
        track_file=track_file,
        behavior=args.behavior,
        fps=args.fps,
        num_episodes=args.episodes,
    )
