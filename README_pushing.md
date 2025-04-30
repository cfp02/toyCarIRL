# Car Pushing Task Implementation

This document outlines the implementation of a pushing task in the car environment, where the car must push an object to a goal area.

## Implementation Status

### ✅ Completed
1. Basic physics setup
   - Added pushable object physics
   - Implemented goal zone detection
   - Set up collision handlers
2. State space expansion
   - Extended state space to 16 dimensions
   - Added object and goal-related features
3. Training infrastructure
   - Updated DQN agent for pushing task
   - Modified reward structure
   - Implemented proper environment loading

### 🚧 In Progress
1. Fine-tuning physics parameters
2. Optimizing reward weights
3. Evaluating training performance

### 📝 Todo
1. Add visualization improvements
2. Implement demonstration collection
3. Add performance metrics specific to pushing task

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

## Reward Structure
Current reward components:
- Distance reduction to goal
- Goal reached bonus
- Contact maintenance reward
- Collision penalties
- Time penalty

## Next Steps
1. Implement demonstration collection interface
2. Add visualization for:
   - Object trajectory
   - Contact points
   - Goal progress
3. Fine-tune reward weights
4. Add performance metrics:
   - Success rate
   - Average completion time
   - Path efficiency

## Notes
- The pushing task is now functional with basic physics
- Training can be done from scratch but may take longer to converge
- Consider collecting demonstrations to speed up learning
- Monitor reward scaling and adjust if needed

## Overview

The goal is to modify the existing car environment to create a pushing task where:
1. The car must push a movable object
2. The object needs to reach a goal area
3. The environment should be simplified to focus on the pushing mechanics

## Implementation Plan

### 1. Physics Modifications

#### New Physics Bodies
- **Pushable Object**:
  - Rectangular shape (initially)
  - Moderate mass (enough to be challenging but not impossible to move)
  - Friction coefficients to allow sliding but not too easily
  - Collision detection with car and environment

- **Goal Area**:
  - Static rectangular zone
  - Visual indicator (different color)
  - Collision detection with pushable object

#### Physics Parameters
- Pushable object mass: 2-3x car mass
- Friction coefficients:
  - Object-ground: 0.3-0.5
  - Car-object: 0.2-0.4
- Elasticity: 0.1-0.3 (some bounce but not too much)

### 2. State Space Expansion

Current state space (10 dimensions) will be expanded to include:
1. Pushable object information:
   - Position (x, y)
   - Velocity (vx, vy)
   - Rotation
   - Angular velocity

2. Goal information:
   - Position
   - Distance to goal
   - Direction to goal

3. Contact information:
   - Is car in contact with object
   - Contact point relative to car
   - Contact force magnitude

Total new state space: ~20 dimensions

### 3. Reward Structure

The reward function will combine multiple components:

1. **Primary Rewards**:
   - Distance reduction to goal: +1 per unit closer
   - Goal reached: +100
   - Maintaining contact: +0.1 per step

2. **Penalties**:
   - Losing contact: -1
   - Collisions with walls: -5
   - Time penalty: -0.01 per step

3. **Shaping Rewards**:
   - Direction alignment: +0.1 when pushing in correct direction
   - Velocity alignment: +0.05 when object moving toward goal

### 4. Environment Modifications

#### Track Design
- Open space with minimal obstacles
- Clear boundaries to prevent objects from leaving
- Goal zone positioned at a reasonable distance
- Initial positions for car and pushable object

#### Visualization
- Different colors for:
  - Pushable object
  - Goal zone
  - Contact indicators
  - Force vectors (optional)

### 5. Training Strategy

#### Initial Phase
1. Start with simplified environment
2. Focus on basic pushing mechanics
3. Use dense rewards for contact maintenance
4. Gradually increase difficulty

#### Training Parameters
- Batch size: 64
- Learning rate: 1e-4
- Gamma: 0.99
- Epsilon: 1.0 to 0.1
- Target update frequency: 1000 steps

#### Evaluation Metrics
1. Success rate (reaching goal)
2. Average time to goal
3. Contact maintenance percentage
4. Path efficiency
5. Collision frequency

### 6. Implementation Steps

1. **Phase 1: Basic Setup**
   - Add pushable object physics
   - Implement goal zone
   - Modify state space
   - Basic reward structure

2. **Phase 2: Core Mechanics**
   - Implement contact detection
   - Add force calculations
   - Basic pushing behavior
   - Initial training runs

3. **Phase 3: Refinement**
   - Optimize physics parameters
   - Fine-tune reward structure
   - Add visualization features
   - Performance optimization

4. **Phase 4: Advanced Features**
   - Multiple objects
   - Dynamic goals
   - Obstacles
   - Different object shapes

### 7. Potential Challenges

1. **Learning Difficulties**
   - Maintaining contact while moving
   - Dealing with object momentum
   - Balancing exploration vs exploitation
   - Local optima in pushing strategies

2. **Physics Challenges**
   - Stable contact forces
   - Realistic friction behavior
   - Preventing object jitter
   - Handling edge cases

3. **Training Challenges**
   - Sparse rewards in complex scenarios
   - Credit assignment for long-term goals
   - Sample efficiency
   - Generalization across different setups

### 8. Future Extensions

1. **Task Variations**
   - Multiple pushable objects
   - Moving goals
   - Obstacle courses
   - Different object shapes

2. **Advanced Features**
   - Hierarchical reinforcement learning
   - Imitation learning from human demonstrations
   - Multi-agent scenarios
   - Real-world transfer

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