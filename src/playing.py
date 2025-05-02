"""
Once a model is learned, use this to play it. That is run/exploit a policy to get the feature expectations of the policy.
Adapted for PyTorch DQN implementation.
"""

import argparse
import os

import numpy as np

from flat_game import carmunk
from nn import DQNAgent

NUM_STATES = 16  # Updated for pushing task
GAMMA = 0.9


def play(agent, weights, track_file="tracks/pushing.json"):
    car_distance = 0
    print(f"Playing with track file: {track_file}")
    game_state = carmunk.GameState(weights, track_file)
    game_state.load_environment(track_file)
    game_state.reset()

    # Get initial state
    _, state, __, ___ = game_state.frame_step((2))
    state = state.reshape(-1)  # Ensure state is properly shaped

    featureExpectations = np.zeros(len(weights))

    # Move.
    while True:
        car_distance += 1

        # Choose action using the agent (evaluate=True to disable exploration)
        action = agent.act(state, evaluate=True)

        # Take action.
        _, state, readings, collision_count = game_state.frame_step(int(action))
        state = state.reshape(-1)  # Ensure state is properly shaped

        # Start recording feature expectations only after 100 frames
        if car_distance > 100:
            featureExpectations += (GAMMA ** (car_distance - 101)) * np.array(readings)

        # Check for goal reached or collision
        if game_state.is_in_goal or collision_count > 5:
            print(f"Episode ended: {'Goal reached' if game_state.is_in_goal else 'Collision'}")
            break

        # Tell us something every 1000 frames
        if car_distance % 1000 == 0:
            print(
                f"Current distance: {car_distance} frames. Collisions: {collision_count}"
            )

    return featureExpectations


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

    # Run the agent and get feature expectations
    feature_expectations = play(agent, weights, args.track)
    print("Feature expectations:")
    print(feature_expectations)
