"""
Once a model is learned, use this to play it. that is run/exploit a policy to get the feature expectations of the policy
"""
import argparse
import os

import numpy as np

from flat_game import carmunk
from nn import neural_net

NUM_STATES = 8
GAMMA = 0.9


def play(model, weights, track_file='tracks/default.json'):
    car_distance = 0
    game_state = carmunk.GameState(weights, track_file)

    _, state, __ = game_state.frame_step((2))

    featureExpectations = np.zeros(len(weights))

    # Move.
    # time.sleep(15)
    while True:
        car_distance += 1

        # Choose action.
        action = np.argmax(model.predict(state, batch_size=1))
        # print ("Action ", action)

        # Take action.
        immediateReward, state, readings = game_state.frame_step(action)
        # print ("immeditate reward:: ", immediateReward)
        # print ("readings :: ", readings)
        # start recording feature expectations only after 100 frames
        if car_distance > 100:
            featureExpectations += (GAMMA ** (car_distance - 101)) * np.array(readings)
        # print ("Feature Expectations :: ", featureExpectations)
        # Tell us something.
        if car_distance % 2000 == 0:
            print("Current distance: %d frames." % car_distance)
            break

    return featureExpectations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a trained model with a specific track')
    parser.add_argument('behavior', default='red', help='Behavior name')
    parser.add_argument('iteration', default='8', help='Iteration number')
    parser.add_argument('frame', default='50000', help='Frame number')
    parser.add_argument('--track', '-t', type=str, default='tracks/default.json', 
                        help='Path to obstacle configuration file (default: tracks/default.json)')
    args = parser.parse_args()
    
    BEHAVIOR = args.behavior
    ITERATION = args.iteration
    FRAME = args.frame
    
    # Resolve the track file path if provided
    track_file = 'tracks/default.json'  # Set a default value
    if args.track:
        # Try different path combinations
        potential_paths = [
            args.track,  # Direct path
            os.path.join('tracks', args.track),  # In tracks folder
            os.path.join('tracks', f"{args.track}.json")  # In tracks folder with .json extension
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
        
    saved_model = (
        "saved-models_"
        + BEHAVIOR
        + "/evaluatedPolicies/"
        + str(ITERATION)
        + "-164-150-100-50000-"
        + str(FRAME)
        + ".h5"
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
    model = neural_net(NUM_STATES, [164, 150], saved_model)
    print(play(model, weights, track_file))
