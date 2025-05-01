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
            featureExpectations += (GAMMA * (car_distance - 101)) * np.array(readings)

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
        "--behavior",
        "-b",
        type=str,
        default="custom",
        help="Behavior name (default: custom)",
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

    # Update global variables from arguments
    BEHAVIOR = args.behavior
    FRAMES = args.frames
    track_file = args.track

    if not os.path.exists(track_file):
        print(
            f"Warning: Track file {track_file} not found, checking alternative paths..."
        )
        alt_path = os.path.join("tracks", track_file)
        if os.path.exists(alt_path):
            track_file = alt_path
            print(f"Using track file: {track_file}")
        else:
            alt_path = os.path.join("tracks", f"{track_file}.json")
            if os.path.exists(alt_path):
                track_file = alt_path
                print(f"Using track file: {track_file}")
            else:
                print("Could not find track file, using default: tracks/default.json")
                track_file = "tracks/default.json"

    print(f"Using track file: {track_file}")
    print(f"Behavior: {BEHAVIOR}")
    print(f"Playing frames: {FRAMES}")

    # Define path to the weights file
    behavior_str = BEHAVIOR if BEHAVIOR else "custom"
    weights_file = f"weights-{behavior_str}.txt"

    try:
        # Load weights from file - handling brackets properly
        with open(weights_file, "r") as f:
            weights_text = f.read()
            # Strip brackets and split by whitespace
            weights_text = weights_text.strip("[]")
            weights = np.fromstring(weights_text, sep=" ")
        print(f"Loaded weights from {weights_file}")
        print(f"Weights: {weights}")
    except FileNotFoundError:
        print(f"Warning: Weights file {weights_file} not found, using default weights")
        # Default weights if file not found
        weights = np.ones(NUM_STATES) / NUM_STATES

    agent = DQNAgent(
        state_size=NUM_STATES,
        action_size=3,
        hidden_sizes=[164, 150],
    )

    # Load the model
    # Construct path to model file
    model_path = (
        f"saved-models/{BEHAVIOR}_models/evaluatedPolicies/checkpoints_1/best_model.pth"
    )

    # Try to load the model
    if not agent.load(model_path):
        print(f"Failed to load model from {model_path}")
        exit(1)
    else:
        print(f"Successfully loaded model from {model_path}")

    # Set to evaluation mode
    agent.policy_net.eval()

    # Run the agent and get feature expectations
    feature_expectations = play(agent, weights, track_file)
    print("Feature expectations:")
    print(feature_expectations)
