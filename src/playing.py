"""
Once a model is learned, use this to play it. That is run/exploit a policy to get the feature expectations of the policy.
Adapted for PyTorch DQN implementation.
"""

import argparse
import os

import numpy as np

from flat_game import carmunk
from nn import DQNAgent

NUM_STATES = 10
GAMMA = 0.9


def play(agent, weights, track_file="tracks/default.json"):
    car_distance = 0
    print(f"Playing with track file: {track_file}")
    game_state = carmunk.GameState(weights, track_file)

    _, state, __, ___ = game_state.frame_step((2))

    featureExpectations = np.zeros(len(weights))

    # Move.
    while True:
        car_distance += 1

        # Choose action using the agent (evaluate=True to disable exploration)
        action = agent.act(state, evaluate=True)

        # Take action.
        _, state, readings, collision_count = game_state.frame_step(int(action))

        # Start recording feature expectations only after 100 frames
        if car_distance > 100:
            featureExpectations += (GAMMA ** (car_distance - 101)) * np.array(readings)

        # Tell us something and break after a certain distance
        if car_distance % 2000 == 0:
                print(f"Current distance: {car_distance} frames. Collisions: {collision_count}")
                break

    return featureExpectations


if __name__ == "__main__":
    # Added default args for red model
    parser = argparse.ArgumentParser(
        description="Run a trained model with a specific track"
    )
    parser.add_argument("behavior", nargs="?", default="red", help="Behavior name")
    parser.add_argument("iteration", nargs="?", default="3", help="Iteration number")
    parser.add_argument("frame", nargs="?", default="100000", help="Frame number")
    parser.add_argument(
        "--track",
        "-t",
        type=str,
        default="tracks/default.json",
        help="Path to obstacle configuration file (default: tracks/default.json)",
    )
    args = parser.parse_args()

    BEHAVIOR = args.behavior
    ITERATION = args.iteration
    FRAME = args.frame

    # Resolve the track file path if provided
    track_file = "tracks/default.json"  # Set a default value
    if args.track:
        # Try different path combinations
        potential_paths = [
            args.track,  # Direct path
            os.path.join("tracks", args.track),  # In tracks folder
            os.path.join(
                "tracks", f"{args.track}.json"
            ),  # In tracks folder with .json extension
        ]

        for path in potential_paths:
            if os.path.exists(path):
                track_file = path
                print(f"Using track file: {path}")
                break
        else:  # This else belongs to the for loop, runs if no break occurred
            print(f"Warning: Could not find track file at {args.track}")
            print(f"Tried: {potential_paths}")
            print("Using default track instead.")

    # Model path for the PyTorch model
    saved_model = (
        "saved-models/"
        + "saved-models_"
        + BEHAVIOR
        + "/evaluatedPolicies/"
        + str(ITERATION)
        + "-164-150-100-50000-"
        + str(FRAME)
        + ".pt"  # Changed from .h5 to .pt for PyTorch
    )

    weights = [
        -0.79380502,
        0.00704546,
        0.50866139,
        0.29466834,
        -0.07636144,
        0.09153848,
        -0.02632325,
        -0.09672041,
    ]

    # Initialize the DQNAgent
    agent = DQNAgent(
        state_size=NUM_STATES,
        action_size=3,  # Assuming 3 actions: left, right, no-op
        hidden_sizes=[164, 150],
        use_prioritized_replay=False,  # Not needed for inference
    )

    # Load the model
    if not agent.load(saved_model):
        print(f"Failed to load model from {saved_model}")
        exit(1)

    # Set to evaluation mode
    agent.policy_net.eval()

    # Run the agent and get feature expectations
    print(play(agent, weights, track_file))
