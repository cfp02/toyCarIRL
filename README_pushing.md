# Car Pushing Task Implementation

This document outlines the implementation of a pushing task in the car environment, where the car must push an object to a goal area using Inverse Reinforcement Learning (IRL).

## Implementation Status

### ✅ Completed
1. Basic physics setup
   - Added pushable object physics
   - Implemented goal zone detection
   - Set up collision handlers
2. State space expansion
   - Extended state space to 16 dimensions
   - Added object and goal-related features
3. Demonstration Collection
   - Implemented manual control interface
   - Added trajectory recording
   - Support for saving demonstrations in JSON format
4. IRL Framework
   - Basic IRL implementation with feature expectations
   - Reward weight optimization
   - Expert policy comparison

### 🚧 In Progress
1. IRL for Pushing Task
   - Updating feature expectations for 16-dimensional state space
   - Modifying reward weight optimization
   - Integrating pushing-specific features
2. Training Pipeline
   - Creating IRL-specific training script
   - Implementing demonstration loading
   - Adding pushing-specific metrics

### 📝 Todo
1. Complete IRL Implementation
   - Update state space handling
   - Modify feature expectations calculation
   - Test with pushing demonstrations
2. Training and Evaluation
   - Create comprehensive training pipeline
   - Add performance metrics
   - Implement visualization tools

## Training Options

### Training from Scratch
You can train the agent from scratch using:
```bash
python3 src/train.py --mode train --env-path tracks/pushing.json --episodes 1000 --hidden-sizes 164 150 --lr 1e-4 --gamma 0.99 --buffer-size 100000 --batch-size 64
```

### Using Demonstrations (Recommended)
While the agent can learn from scratch, demonstrations can significantly speed up learning. To use demonstrations:

1. Collect demonstrations:
   - Run the environment in demonstration mode (coming soon)
   - Use arrow keys to control the car
   - Successfully push objects to goal zones
   - Demonstrations will be saved automatically

2. Train with demonstrations:
   - Coming soon: Integration with demonstration data
   - Will include behavior cloning pre-training
   - Will add demonstration mixing during RL training

## Current State Space (16 dimensions)
1. Sonar readings (3 dimensions)
2. Environment features (6 dimensions)
   - Black space
   - Yellow obstacles
   - Brown obstacles
   - Out of bounds
   - Red walls
   - Collision count
3. Pushing task features (7 dimensions)
   - Distance to object
   - Angle to object (sin and cos)
   - Distance to goal
   - Is pushing (contact state)
   - Is in goal
   - Crashed

## IRL Implementation

### Demonstration Collection
To collect expert demonstrations:
```bash
python3 src/manualControl.py --task pushing
```
Controls:
- LEFT ARROW: Turn left
- RIGHT ARROW: Turn right
- UP ARROW: Move forward (default)
- DOWN ARROW: Exit and save trajectory
- R: Reset current demonstration
- Q: Quit collection

### Training with IRL
The IRL process:
1. Collect expert demonstrations
2. Learn reward function from demonstrations
3. Train agent using learned rewards

Current IRL parameters:
- State space: 16 dimensions
- Feature expectations calculation
- Reward weight optimization
- Expert policy comparison

## Next Steps
1. Complete IRL implementation for pushing task
2. Create comprehensive training pipeline
3. Add performance metrics:
   - Success rate
   - Average completion time
   - Path efficiency
4. Implement visualization tools

## Notes
- The pushing task is functional with basic physics
- IRL framework is in place but needs updating for pushing task
- Demonstrations can be collected and saved
- Focus is on learning from demonstrations without explicit rewards

## Overview

The goal is to use IRL to learn how to push objects to goal areas by:
1. Collecting expert demonstrations
2. Learning the reward function from demonstrations
3. Training the agent using the learned rewards
4. Evaluating performance on the pushing task

## Implementation Plan

### 1. IRL Framework Updates

#### State Space Handling
- Update feature expectations calculation
- Modify reward weight optimization
- Integrate pushing-specific features

#### Demonstration Processing
- Load and process demonstrations
- Calculate feature expectations
- Optimize reward weights

### 2. Training Pipeline

#### IRL Training
- Load demonstrations
- Learn reward function
- Train agent with learned rewards

#### Evaluation
- Success rate
- Average completion time
- Path efficiency
- Contact maintenance

### 3. Testing and Validation

#### Metrics
- Success rate
- Average completion time
- Path efficiency
- Contact maintenance

#### Visualization
- Object trajectory
- Contact points
- Goal progress
- Feature expectations

## Notes
- Focus on learning from demonstrations
- No explicit reward function needed
- Monitor feature expectations
- Track convergence of learned rewards

## Progress Tracking

- [ ] Basic physics implementation
- [ ] State space modification
- [ ] Reward structure implementation
- [ ] Initial training setup
- [ ] Basic pushing behavior
- [ ] Performance optimization
- [ ] Advanced features

## Notes

- Start with simplest possible implementation
- Iterate quickly on physics parameters
- Focus on stable contact mechanics
- Use visualization for debugging
- Track metrics from the beginning
- Document successful parameter combinations 