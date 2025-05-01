"""
Once a model is learned, use this to play it. That is run/exploit a policy to get the feature expectations of the policy.
Adapted for PyTorch DQN implementation.
"""

import argparse
import os

import numpy as np

from flat_game import carmunk
from nn import DQNAgent

NUM_STATES = 9
GAMMA = 0.9


def play(agent, weights, track_file="tracks/default.json"):
    car_distance = 0
    print(f"Playing with track file: {track_file}")
    game_state = carmunk.GameState(weights, track_file)

    _, state, __, ___, done = game_state.frame_step((2))

    featureExpectations = np.zeros(len(weights))

    # Move.
    while True:
        car_distance += 1

        # Choose action using the agent (evaluate=True to disable exploration)
        action = agent.act(state, evaluate=True)

        # Take action.
        _, state, readings, collision_count, done = game_state.frame_step(int(action))

        # Start recording feature expectations only after 100 frames
        if car_distance > 100:
            featureExpectations += (GAMMA ** (car_distance - 101)) * np.array(readings)

        # Tell us something and break after a certain distance
        if done:
            break

    return featureExpectations


if __name__ == "__main__":
    # Added default args for red model
    parser = argparse.ArgumentParser(
        description="Run a trained model with a specific track"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="saved-models/checkpoints/best_model.pth",
        help="Model path (default: saved-models/checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--track",
        "-t",
        type=str,
        default="tracks/default.json",
        help="Path to obstacle configuration file (default: tracks/default.json)",
    )
    parser.add_argument(
        "--frames",
        "-f",
        type=int,
        default=2000,
        help="Number of frames to run (default: 2000)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        exit(1)

    # Resolve the track file path if provided
    track_file = "tracks/default.json"
    if args.track:
        potential_paths = [
            args.track,
            os.path.join("tracks", args.track),
            os.path.join(
                "tracks", f"{args.track}.json"
            ),  # In tracks folder with .json extension
        ]

        for path in potential_paths:
            if os.path.exists(path):
                track_file = path
                print(f"Using track file: {path}")
                break
            else:
                print(f"Warning: Could not find track file at {args.track}")
                print(f"Tried: {potential_paths}")
                print("Using default track instead.")

    # Model path for the PyTorch model
    print(f"Loading model from: {args.model}")

    # change this to change reward function, this should match whatever model you load. TODO: make arg
    weights = [
        3.96957075e-01,
        1.60768375e-01,
        3.85956376e-01,
        2.17107187e-01,
        5.31013934e-01,
        1.89519667e-01,
        0.00000000e00,
        6.23592131e-02,
        5.14246354e-01,
        3.43368382e-02,
    ]

    agent = DQNAgent(
        state_size=NUM_STATES,
        action_size=3,
        hidden_sizes=[164, 150],
    )

    # Load the model
    if not agent.load(args.model):
        print(f"Failed to load model from {args.model}")
        exit(1)

    # Set to evaluation mode
    agent.policy_net.eval()

    # Run the agent and get feature expectations
    feature_expectations = play(agent, weights, track_file)
    print("Feature expectations:")
    print(feature_expectations)
