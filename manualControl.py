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
from flat_game import carmunk
import numpy as np
from nn import neural_net
import pygame

NUM_STATES = 8
GAMMA = 0.9 # the discount factor for RL algorithm

def play():
    car_distance = 0
    weights = [1, 1, 1, 1, 1, 1, 1, 1]  # just some random weights
    game_state = carmunk.GameState(weights)
    _, state, __ = game_state.frame_step((2))
    featureExpectations = np.zeros(len(weights))
    Prev = np.zeros(len(weights))
    
    # Create a clock to control frame rate
    clock = pygame.time.Clock()
    FPS = 30  # Target frames per second
    
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
            featureExpectations += (GAMMA**(car_distance-101))*np.array(readings)
        
        # Only print updates every 100 steps
        if car_distance % 100 == 0:
            changePercentage = (np.linalg.norm(featureExpectations - Prev)*100.0)/np.linalg.norm(featureExpectations)
            print(f"\rDistance: {car_distance}, Change: {changePercentage:.2f}%", end="")
            Prev = np.array(featureExpectations)

        # Control frame rate
        clock.tick(FPS)

        if car_distance % 2000 == 0:
            break

    print("\nFinal feature expectations:", featureExpectations)
    return featureExpectations

if __name__ == "__main__":
    result = play()
    print("Final feature expectations:", result)
