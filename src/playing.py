"""
Once a model is learned, use this to play it. That is run/exploit a policy to get the feature expectations of the policy.
Adapted for PyTorch DQN implementation.
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from flat_game import carmunk
from nn import DQNAgent

NUM_STATES = 16  # Updated for pushing task
GAMMA = 0.9


def play(agent, weights, track_file="tracks/pushing.json", trial_num: int = 0) -> Dict[str, Any]:
    car_distance = 0
    print(f"Trial {trial_num + 1}: Playing with track file: {track_file}")
    game_state = carmunk.GameState(weights, track_file)
    game_state.load_environment(track_file)
    game_state.reset()

    # Get initial state
    _, state, __, ___ = game_state.frame_step((2))
    state = state.reshape(-1)  # Ensure state is properly shaped

    featureExpectations = np.zeros(len(weights))
    trial_data = {
        "steps": [],
        "rewards": [],
        "collisions": [],
        "features": [],
        "goal_reached": False,
        "total_steps": 0,
        "total_reward": 0,
        "total_collisions": 0
    }

    # Move.
    while True:
        car_distance += 1

        # Choose action using the agent (evaluate=True to disable exploration)
        action = agent.act(state, evaluate=True)

        # Take action.
        reward, state, readings, collision_count = game_state.frame_step(int(action))
        state = state.reshape(-1)  # Ensure state is properly shaped

        # Convert readings to numpy array for feature expectations calculation
        readings_array = np.array(readings)

        # Record trial data
        trial_data["steps"].append(car_distance)
        trial_data["rewards"].append(float(reward))
        trial_data["collisions"].append(collision_count)
        trial_data["features"].append(readings)  # Store original list

        # Start recording feature expectations only after 100 frames
        if car_distance > 100:
            featureExpectations += (GAMMA ** (car_distance - 101)) * readings_array

        # Check for goal reached or collision
        if game_state.is_in_goal or collision_count > 5:
            trial_data["goal_reached"] = game_state.is_in_goal
            trial_data["total_steps"] = car_distance
            trial_data["total_reward"] = sum(trial_data["rewards"])
            trial_data["total_collisions"] = collision_count
            print(f"Trial {trial_num + 1} ended: {'Goal reached' if game_state.is_in_goal else 'Collision'}")
            print(f"Total steps: {car_distance}, Total reward: {sum(trial_data['rewards']):.2f}, Collisions: {collision_count}")
            break

        # Tell us something every 1000 frames
        if car_distance % 1000 == 0:
            print(
                f"Trial {trial_num + 1}: Current distance: {car_distance} frames. Collisions: {collision_count}"
            )

    return featureExpectations, trial_data


def run_trials(
    agent: DQNAgent,
    weights: List[float],
    track_file: str,
    num_trials: int = 100,
    output_dir: str = "trials"
) -> Dict[str, Any]:
    """Run multiple trials and aggregate results."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results
    all_feature_expectations = []
    all_trial_data = []
    summary = {
        "success_rate": 0,
        "avg_steps": 0,
        "avg_reward": 0,
        "avg_collisions": 0,
        "feature_expectations": None
    }
    
    # Run trials
    for i in range(num_trials):
        print(f"\nStarting trial {i + 1}/{num_trials}")
        fe, trial_data = play(agent, weights, track_file, i)
        all_feature_expectations.append(fe)
        all_trial_data.append(trial_data)
        
        # Save individual trial data
        trial_file = os.path.join(output_dir, f"trial_{i + 1}.json")
        with open(trial_file, "w") as f:
            json.dump(trial_data, f, indent=2)
    
    # Calculate summary statistics
    success_count = sum(1 for trial in all_trial_data if trial["goal_reached"])
    summary["success_rate"] = success_count / num_trials
    summary["avg_steps"] = np.mean([trial["total_steps"] for trial in all_trial_data])
    summary["avg_reward"] = np.mean([trial["total_reward"] for trial in all_trial_data])
    summary["avg_collisions"] = np.mean([trial["total_collisions"] for trial in all_trial_data])
    summary["feature_expectations"] = np.mean(all_feature_expectations, axis=0).tolist()
    
    # Save summary
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nTrial Summary:")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    print(f"Average Steps: {summary['avg_steps']:.1f}")
    print(f"Average Reward: {summary['avg_reward']:.2f}")
    print(f"Average Collisions: {summary['avg_collisions']:.1f}")
    print(f"Feature Expectations: {summary['feature_expectations']}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the car environment")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model file",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="tracks/pushing.json",
        help="Path to track file (default: tracks/pushing.json)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials to run (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trials",
        help="Output directory for trial data (default: trials)",
    )
    args = parser.parse_args()

    # Initialize weights for the reward function
    weights = [1.0] * NUM_STATES  # Equal weights for all features

    agent = DQNAgent(
        state_size=NUM_STATES,
        action_size=3,
        hidden_sizes=[164, 150],
        is_pushing_task=True  # Enable pushing task mode
    )

    # Load the model
    if not agent.load(args.model):
        print(f"Failed to load model from {args.model}")
        exit(1)

    # Set to evaluation mode
    agent.policy_net.eval()

    # Run multiple trials and get summary
    summary = run_trials(agent, weights, args.track, args.trials, args.output)
