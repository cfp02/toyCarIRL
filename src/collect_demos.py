"""
Collect expert demonstrations for the pushing task.
This will record state-action pairs as you control the car to push objects to goals.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pygame

from flat_game import carmunk

def collect_demonstrations(track_file="tracks/pushing.json", num_demos=5):
    """Collect expert demonstrations by letting the user control the car."""
    # Initialize game
    game_state = carmunk.GameState([1.0] * 16, track_file)  # Using equal weights during collection
    
    # Initialize pygame for keyboard input
    pygame.init()
    clock = pygame.time.Clock()
    
    demonstrations = []
    demo_count = 0
    
    print("\n🎮 Demonstration Collection Mode")
    print("Controls:")
    print("  ← Left arrow  : Turn left")
    print("  → Right arrow : Turn right")
    print("  ↑ Up arrow   : Go straight")
    print("  R            : Reset current demonstration")
    print("  Q            : Quit collection")
    print("\nCollect demonstrations by pushing objects to the goal zone!")
    
    while demo_count < num_demos:
        # Start new demonstration
        print(f"\nStarting demonstration {demo_count + 1}/{num_demos}")
        current_demo = []
        
        # Reset environment
        action = 2  # Start going straight
        _, state, _, _ = game_state.frame_step(action)
        state = state.reshape(-1)
        
        collecting = True
        while collecting:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return demonstrations
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Quit
                        return demonstrations
                    elif event.key == pygame.K_r:  # Reset current demo
                        collecting = False
                        print("Resetting current demonstration...")
                        break
            
            # Get keyboard state
            keys = pygame.key.get_pressed()
            
            # Determine action based on keyboard input
            if keys[pygame.K_LEFT]:
                action = 0  # Turn left
            elif keys[pygame.K_RIGHT]:
                action = 1  # Turn right
            else:
                action = 2  # Go straight
            
            # Take step in environment
            reward, next_state, _, _ = game_state.frame_step(action)
            next_state = next_state.reshape(-1)
            
            # Store state-action pair
            current_demo.append({
                'state': state.tolist(),
                'action': action,
                'next_state': next_state.tolist(),
                'reward': float(reward)
            })
            
            # Update state
            state = next_state
            
            # Check if demonstration is complete (object in goal)
            if game_state.is_in_goal:
                print(f"✅ Demonstration {demo_count + 1} completed successfully!")
                demonstrations.append(current_demo)
                demo_count += 1
                collecting = False
            
            # Cap framerate
            clock.tick(30)
    
    return demonstrations

def save_demonstrations(demonstrations, track_file):
    """Save demonstrations to a file."""
    # Create demonstrations directory if it doesn't exist
    os.makedirs('demonstrations', exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    track_name = os.path.splitext(os.path.basename(track_file))[0]
    filename = f"demonstrations/trajectory_{track_name}_{timestamp}.json"
    
    # Save demonstrations
    with open(filename, 'w') as f:
        json.dump({
            'track_file': track_file,
            'demonstrations': demonstrations
        }, f)
    
    print(f"\n💾 Saved {len(demonstrations)} demonstrations to {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description="Collect expert demonstrations for the pushing task")
    parser.add_argument('--track', type=str, default='tracks/pushing.json',
                      help='Track file to use (default: tracks/pushing.json)')
    parser.add_argument('--num-demos', type=int, default=5,
                      help='Number of demonstrations to collect (default: 5)')
    args = parser.parse_args()
    
    # Collect demonstrations
    print(f"Collecting {args.num_demos} demonstrations using track: {args.track}")
    demonstrations = collect_demonstrations(args.track, args.num_demos)
    
    # Save if we got any demonstrations
    if demonstrations:
        save_demonstrations(demonstrations, args.track)

if __name__ == "__main__":
    main() 