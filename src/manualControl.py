"""
Manually control the agent to provide expert trajectories.
The main aim is to get the feature expectations respective to the expert trajectories manually given by the user
Use the arrow keys to move the agent around
Left arrow key: turn Left
right arrow key: turn right
up arrow key: dont turn, move forward
down arrow key: exit

Also, always exit using down arrow key rather than Ctrl+C or your terminal will be tken over by curses
"""

import argparse
import os

import numpy as np
import pygame

from flat_game import carmunk

NUM_STATES = 8
GAMMA = 0.9  # the discount factor for RL algorithm


def play(obstacle_file=None):
    car_distance = 0
    weights = [1, 1, 1, 1, 1, 1, 1, 1]  # just some random weights
    game_state = (
        carmunk.GameState(weights)
        if obstacle_file is None
        else carmunk.GameState(weights, obstacle_file)
    )
    _, state, __ = game_state.frame_step((2))
    featureExpectations = np.zeros(len(weights))
    Prev = np.zeros(len(weights))

    # Create a clock to control frame rate
    clock = pygame.time.Clock()
    FPS = 5  # Target frames per second

    running = True
    action = 2  # Default action is forward
    last_action = 2

    print("Starting manual control...")
    print("Use arrow keys to control the car:")
    print("LEFT: Turn left")
    print("RIGHT: Turn right")
    print("DOWN: Exit")
    print("The car moves forward by default")

    while running:
        car_distance += 1

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 1
                elif event.key == pygame.K_RIGHT:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    running = False
                    break
            elif event.type == pygame.KEYUP:
                # When key is released, go back to forward motion
                if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                    action = 2

        # Only take action if it's different from last action
        if action != last_action:
            # Take action
            immediateReward, state, readings = game_state.frame_step(action)
            last_action = action
        else:
            # Keep moving in the same direction
            immediateReward, state, readings = game_state.frame_step(action)

        if car_distance > 100:
            featureExpectations += (GAMMA ** (car_distance - 101)) * np.array(readings)

        # Only print updates every 100 steps
        if car_distance % 100 == 0:
            norm_fe = np.linalg.norm(featureExpectations)
            if norm_fe > 0:
                changePercentage = (
                    np.linalg.norm(featureExpectations - Prev) * 100.0
                ) / norm_fe
            else:
                changePercentage = (
                    0.0  # If norm is zero, there's no meaningful percentage
                )
            print(
                f"\rDistance: {car_distance}, Change: {changePercentage:.2f}%", end=""
            )
            Prev = np.array(featureExpectations)

        # Control frame rate
        clock.tick(FPS)

        if car_distance % 2000 == 0:
            break

    print("\nFinal feature expectations:", featureExpectations)
    return featureExpectations


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Manually control the car and record feature expectations."
    )
    parser.add_argument(
        "--track",
        "-t",
        type=str,
        default=None,
        help="Path to obstacle configuration file (default: tracks/default.json)",
    )
    args = parser.parse_args()

    # Resolve path if provided
    obstacle_file = None
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
                obstacle_file = path
                break

        if not obstacle_file:
            print(f"Warning: Could not find track file at {args.track}")
            print(f"Tried: {potential_paths}")
            print("Using default track instead.")

    result = play(obstacle_file)
    print("Final feature expectations:", result)
