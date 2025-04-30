import json
import math
import os
import random

import numpy as np
import pygame
import pymunk
from pygame.color import THECOLORS
from pymunk.pygame_util import DrawOptions
from pymunk.vec2d import Vec2d

# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Create a draw options object for pymunk
draw_options = DrawOptions(screen)

# Showing sensors and redrawing slows things down.
flag = 1
show_sensors = 0  # Turn off sensor visualization
draw_screen = flag


class GameState:
    def __init__(self, weights, obstacle_file="tracks/default.json"):
        # Global-ish.
        self.crashed = False
        # Add collision counter
        self.collision_count = 0
        self.was_crashed_last_frame = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0.0, 0.0)
        self.space.damping = 0.07  # Add damping to the entire space
        self.W = weights  # weights for the reward function

        # Record steps.
        self.num_steps = 0
        self.num_obstacles_type = 5  # Track 5 types consistently

        # Set up collision handler
        self.collision_handler = self.space.add_collision_handler(1, 2)
        self.collision_handler.begin = self.handle_collision

        # Add collision handler for car and pushable object
        self.push_collision_handler = self.space.add_collision_handler(2, 3)
        self.push_collision_handler.begin = self.handle_push_collision
        self.push_collision_handler.separate = self.handle_push_separation

        # Add collision handler for pushable object and goal
        self.goal_collision_handler = self.space.add_collision_handler(3, 4)
        self.goal_collision_handler.begin = self.handle_goal_collision

        # Track contact state
        self.is_pushing = False
        self.is_in_goal = False

        # Store track file for reset
        self.obstacle_file = obstacle_file
        
        # Initialize environment
        self.load_environment(obstacle_file)

    def reset(self):
        """Reset the environment with random positions"""
        # Clear the space
        for body in self.space.bodies:
            self.space.remove(body)
        for shape in self.space.shapes:
            self.space.remove(shape)
            
        # Reset state variables
        self.crashed = False
        self.collision_count = 0
        self.was_crashed_last_frame = False
        self.is_pushing = False
        self.is_in_goal = False
        self.num_steps = 0
        
        # Generate random positions
        # Keep objects within bounds (100 to 700 for both x and y)
        car_x = random.randint(100, 700)
        car_y = random.randint(100, 700)
        block_x = random.randint(100, 700)
        block_y = random.randint(100, 700)
        
        # Ensure car and block are not too close to each other
        while abs(car_x - block_x) < 100 and abs(car_y - block_y) < 100:
            block_x = random.randint(100, 700)
            block_y = random.randint(100, 700)
        
        # Recreate environment with new positions
        self.load_environment(self.obstacle_file, car_pos=(car_x, car_y), block_pos=(block_x, block_y))

    def load_environment(self, obstacle_file, car_pos=None, block_pos=None):
        """Load environment from file with optional positions"""
        # Get the project root directory (one level up from src)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # Construct the full path to the track file
        track_path = os.path.join(project_root, obstacle_file)
        
        try:
            with open(track_path, 'r') as f:
                config = json.load(f)
                print(f"🚗 Using track file: {track_path}")
        except FileNotFoundError:
            print(f"Warning: Track file {track_path} not found, checking alternative paths...")
            # Try alternative paths
            alt_paths = [
                os.path.join(project_root, 'tracks', os.path.basename(obstacle_file)),
                os.path.join(project_root, obstacle_file),
                obstacle_file
            ]
            for path in alt_paths:
                try:
                    with open(path, 'r') as f:
                        config = json.load(f)
                        print(f"🚗 Using track file: {path}")
                        break
                except FileNotFoundError:
                    continue
            else:
                print("⚠️ Could not find track file, using default: tracks/default.json")
                default_path = os.path.join(project_root, 'tracks', 'default.json')
                with open(default_path, 'r') as f:
                    config = json.load(f)

        # Create boundary walls
        self.create_boundary_walls()

        # Create walls from config
        for wall in config.get('walls', []):
            self.create_obstacle(
                wall['xy1'],
                wall['xy2'],
                wall.get('radius', 7),
                wall.get('color', 'red')
            )

        # Get car configuration
        car_start = config.get('car_start', {'x': 150, 'y': 350, 'r': 15, 'angle': 1.4})
        
        # Create car with optional position
        if car_pos:
            x, y = car_pos
        else:
            x, y = car_start['x'], car_start['y']
        self.create_car(x, y, car_start['r'])
        self.car_body.angle = car_start.get('angle', 1.4)

        # Get pushable object configuration
        obj = config.get('pushable_object', {'x': 400, 'y': 400, 'width': 30, 'height': 30, 'mass': 2.0})
        
        # Create pushable object with optional position
        if block_pos:
            x, y = block_pos
        else:
            x, y = obj['x'], obj['y']
        self.create_pushable_object(x, y, obj['width'], obj['height'], obj.get('mass', 2.0))

        # Create goal zone if specified
        if 'goal_zone' in config:
            goal = config['goal_zone']
            self.create_goal_zone(
                goal['x'],
                goal['y'],
                goal['width'],
                goal['height']
            )

    def handle_collision(self, arbiter, space, data):
        """Handle collision between car and obstacles"""
        self.crashed = True
        # Increment collision counter
        self.collision_count += 1
        # Allow the collision to occur and generate response force
        return True

    def handle_push_collision(self, arbiter, space, data):
        """Handle collision between car and pushable object"""
        self.is_pushing = True
        return True

    def handle_push_separation(self, arbiter, space, data):
        """Handle separation between car and pushable object"""
        self.is_pushing = False
        return True

    def handle_goal_collision(self, arbiter, space, data):
        """Handle collision between pushable object and goal zone"""
        # Get the shapes involved in the collision
        shape1, shape2 = arbiter.shapes
        
        # Debug print
        print(f"Goal collision detected! Shape1 type: {shape1.collision_type}, Shape2 type: {shape2.collision_type}")
        
        # Check if one of the shapes is the pushable object (type 3) and the other is the goal (type 4)
        if (shape1.collision_type == 3 and shape2.collision_type == 4) or \
           (shape1.collision_type == 4 and shape2.collision_type == 3):
            print("Pushable object entered goal zone!")
            self.is_in_goal = True
            return True
        return False

    def create_boundary_walls(self):
        """Create the boundary walls of the environment."""
        static = [
            pymunk.Segment(self.space.static_body, (0, 1), (0, height), 1),
            pymunk.Segment(self.space.static_body, (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body, (width - 1, height), (width - 1, 1), 1
            ),
            pymunk.Segment(self.space.static_body, (1, 1), (width, 1), 1),
        ]
        for s in static:
            s.friction = 1.0
            s.elasticity = 0.7  # Add some bounce
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS["red"]
            self.space.add(s)

    def create_obstacle(self, xy1, xy2, r, color):
        # Create static body for obstacles
        c_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        c_shape = pymunk.Segment(c_body, xy1, xy2, r)
        c_shape.friction = 1.0
        c_shape.elasticity = 0.7  # Add some bounce
        c_shape.group = 1
        c_shape.collision_type = 1
        c_shape.color = THECOLORS[color]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        self.cat_shape.collision_type = 1
        self.space.add(self.cat_body, self.cat_shape)

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, r)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = (
            1.0  # Increased bounce for better collision response
        )
        self.car_body.angle = 1.4

        # Set collision type for car
        self.car_shape.collision_type = 2

        # Initial impulse
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse_at_local_point(driving_direction)

        self.space.add(self.car_body, self.car_shape)

    def create_pushable_object(self, x, y, width, height, mass=2.0):
        """Create a pushable object in the environment"""
        # Double the size
        width *= 2
        height *= 2
        
        # Create body with damping
        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        body.position = x, y
        body.angle = 0
        body.damping = 0.8  # Add damping to make the object slow down over time
        
        # Create shape with increased friction
        shape = pymunk.Poly.create_box(body, (width, height))
        shape.friction = 0.0  # Remove friction
        shape.elasticity = 0.1  # Reduced elasticity
        shape.collision_type = 3  # Set collision type for pushable object
        
        # Add to space
        self.space.add(body, shape)
        self.pushable_object = body
        self.pushable_shape = shape

    def create_goal_zone(self, x, y, width, height):
        """Create a goal zone in the physics space"""
        # Create static body for goal zone
        self.goal_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.goal_body.position = x, y

        # Create the shape
        self.goal_shape = pymunk.Poly.create_box(self.goal_body, (width, height))
        self.goal_shape.friction = 0.0
        self.goal_shape.elasticity = 0.0
        self.goal_shape.color = THECOLORS["green"]
        self.goal_shape.collision_type = 4  # Unique collision type for goal zone
        self.goal_shape.sensor = True  # Make it a sensor so objects pass through

        # Add to space
        self.space.add(self.goal_body, self.goal_shape)

    def check_goal(self):
        """Check if the pushable object is in the goal zone"""
        # The goal state is now managed by the collision handler
        return self.is_in_goal

    def frame_step(self, action):
        # Reset goal state at start of each frame
        self.is_in_goal = False
        
        # Update space damping
        self.space.damping = 0.1  # Set to lower value for more damping

        # Update car's turning angle and velocity based on action
        if action == 0:  # Turn left
            self.car_body.angle += 0.1
        elif action == 1:  # Turn right
            self.car_body.angle -= 0.1
        
        # Move forward with constant velocity
        self.car_body.velocity = (
            math.cos(self.car_body.angle) * 100,
            math.sin(self.car_body.angle) * 100
        )
        
        # Step the physics simulation
        self.space.step(1.0/60.0)
        
        # Check bounds
        self.check_bounds()
        
        # Draw the screen
        if draw_screen:
            screen.fill(THECOLORS["black"])  # Clear screen
            self.space.debug_draw(draw_options)  # Draw physics objects
            pygame.display.flip()  # Update display
        
        # Get state and reward
        state = self.get_state()
        reward = self.calculate_reward()
        features = self.get_features()
        
        # Check if goal is reached and reset if needed
        if self.is_in_goal:
            print("Goal reached! Resetting environment...")
            self.reset()
            return reward, state, features, self.collision_count
        
        return reward, state, features, self.collision_count

    def check_bounds(self):
        """Check if car has gone out of bounds and correct if needed"""
        x, y = self.car_body.position
        r = self.car_shape.radius

        # Apply corrections if out of bounds
        if x < r:
            self.car_body.position = (r, y)
            self.car_body.velocity = Vec2d(0, self.car_body.velocity.y)
            self.crashed = True
        elif x > width - r:
            self.car_body.position = (width - r, y)
            self.car_body.velocity = Vec2d(0, self.car_body.velocity.y)
            self.crashed = True

        if y < r:
            self.car_body.position = (x, r)
            self.car_body.velocity = Vec2d(self.car_body.velocity.x, 0)
            self.crashed = True
        elif y > height - r:
            self.car_body.position = (x, height - r)
            self.car_body.velocity = Vec2d(self.car_body.velocity.x, 0)
            self.crashed = True

    def move_obstacles(self):
        # Obstacles are now static, so this method does nothing
        pass

    def move_cat(self):
        speed = random.randint(20, 200)
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction

    def car_is_crashed(self, readings):
        if readings[0] >= 0.93 or readings[1] >= 0.93 or readings[2] >= 0.93:
            return True
        else:
            return False

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left = self.make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left

        obstacleType = []
        obstacleType.append(self.get_arm_distance(arm_left, x, y, angle, 0.75)[1])
        obstacleType.append(self.get_arm_distance(arm_middle, x, y, angle, 0)[1])
        obstacleType.append(self.get_arm_distance(arm_right, x, y, angle, -0.75)[1])

        ObstacleNumber = np.zeros(self.num_obstacles_type)

        for i in obstacleType:
            if i == 0:
                ObstacleNumber[0] += 1  # Black space
            elif i == 1:
                ObstacleNumber[1] += 1  # Yellow obstacle
            elif i == 2:
                ObstacleNumber[2] += 1  # Brown obstacle
            elif i == 3:
                ObstacleNumber[3] += 1  # Out of bounds
            elif i == 4:
                ObstacleNumber[4] += 1  # Red boundary walls

        # Rotate them and get readings.
        readings.append(
            1.0 - float(self.get_arm_distance(arm_left, x, y, angle, 0.75)[0] / 39.0)
        )  # 39 = max distance
        readings.append(
            1.0 - float(self.get_arm_distance(arm_middle, x, y, angle, 0)[0] / 39.0)
        )
        readings.append(
            1.0 - float(self.get_arm_distance(arm_right, x, y, angle, -0.75)[0] / 39.0)
        )
        readings.append(float(ObstacleNumber[0] / 3.0))  # Black space
        readings.append(float(ObstacleNumber[1] / 3.0))  # Yellow obstacles
        readings.append(float(ObstacleNumber[2] / 3.0))  # Brown obstacles
        readings.append(float(ObstacleNumber[3] / 3.0))  # Out of bounds
        readings.append(float(ObstacleNumber[4] / 3.0))  # Red boundary walls

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(x, y, point[0], point[1], angle + offset)

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if (
                rotated_p[0] <= 0
                or rotated_p[1] <= 0
                or rotated_p[0] >= width
                or rotated_p[1] >= height
            ):
                return [i, 3]  # Sensor is off the screen, return 3 for out of bounds
            else:
                obs = screen.get_at(rotated_p)
                temp = self.get_track_or_not(obs)
                if temp != 0:
                    return [
                        i,
                        temp,
                    ]  # sensor hit an obstacle, return the type of obstacle

        # Return the distance for the arm.
        return [i, 0]  # sensor did not hit anything return 0 for black space

    def make_sonar_arm(self, x, y):
        spread = 8  # Default spread.
        distance = 7  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = y_change + y_1
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        """Differentiate between the objects the car views."""
        if reading == THECOLORS["red"]:
            return 4  # Sensor is on a red boundary wall - NEW!
        elif reading == THECOLORS["yellow"]:
            return 1  # Sensor is on a yellow obstacle
        elif reading == THECOLORS["brown"]:
            return 2  # Sensor is on brown obstacle
        else:
            return 0  # for black space

    def get_state(self):
        """Get the current state vector"""
        # Get car position and velocity
        car_pos = self.car_body.position
        car_vel = self.car_body.velocity
        car_angle = self.car_body.angle
        
        # Get object position and velocity if it exists
        obj_pos = (0, 0)
        obj_vel = (0, 0)
        if self.pushable_object:
            obj_pos = self.pushable_object.position
            obj_vel = self.pushable_object.velocity
            
        # Get goal position
        goal_pos = (0, 0)
        if self.goal_body:
            goal_pos = self.goal_body.position
            
        # Calculate distances and angles
        dist_to_obj = math.sqrt((car_pos.x - obj_pos.x)**2 + (car_pos.y - obj_pos.y)**2)
        angle_to_obj = math.atan2(obj_pos.y - car_pos.y, obj_pos.x - car_pos.x) - car_angle
        dist_to_goal = math.sqrt((obj_pos.x - goal_pos.x)**2 + (obj_pos.y - goal_pos.y)**2)
        
        # Normalize angles to [-pi, pi]
        angle_to_obj = (angle_to_obj + math.pi) % (2 * math.pi) - math.pi
        
        # Create state vector
        state = np.array([
            # Car state
            car_pos.x / 1000.0,  # Normalize by environment width
            car_pos.y / 700.0,   # Normalize by environment height
            car_vel.x / 10.0,    # Normalize velocity
            car_vel.y / 10.0,
            car_angle / math.pi, # Normalize angle to [-1, 1]
            
            # Object state
            obj_pos.x / 1000.0,
            obj_pos.y / 700.0,
            obj_vel.x / 10.0,
            obj_vel.y / 10.0,
            
            # Goal state
            goal_pos.x / 1000.0,
            goal_pos.y / 700.0,
            
            # Relative states
            dist_to_obj / 1000.0,
            math.sin(angle_to_obj),
            math.cos(angle_to_obj),
            dist_to_goal / 1000.0,
            
            # Contact state
            1.0 if self.is_pushing else 0.0,
            
            # Collision count
            self.collision_count / 10.0
        ])
        
        return state

    def calculate_reward(self):
        """Calculate reward based on current state"""
        # Get current positions
        car_pos = self.car_body.position
        obj_pos = self.pushable_object.position if self.pushable_object else (0, 0)
        goal_pos = self.goal_body.position if self.goal_body else (0, 0)
        
        # Calculate distances
        dist_to_obj = math.sqrt((car_pos.x - obj_pos.x)**2 + (car_pos.y - obj_pos.y)**2)
        dist_to_goal = math.sqrt((obj_pos.x - goal_pos.x)**2 + (obj_pos.y - goal_pos.y)**2)
        
        # Get features
        features = self.get_features()
        
        # Calculate reward components
        reward = 0.0
        
        # Distance reduction reward
        if hasattr(self, 'prev_dist_to_goal'):
            dist_reduction = self.prev_dist_to_goal - dist_to_goal
            reward += dist_reduction * 10.0  # Scale up distance reduction
        
        # Contact maintenance reward
        if self.is_pushing:
            reward += 0.1  # Small reward for maintaining contact
        
        # Goal reached bonus
        if self.is_in_goal:
            reward += 100.0
        
        # Collision penalty
        if self.collision_count > 0:
            reward -= 5.0 * self.collision_count
        
        # Time penalty
        reward -= 0.01  # Small penalty per step
        
        # Update previous distance
        self.prev_dist_to_goal = dist_to_goal
        
        return reward

    def get_features(self):
        """Get feature vector for reward calculation"""
        # Get current positions
        car_pos = self.car_body.position
        obj_pos = self.pushable_object.position if self.pushable_object else (0, 0)
        goal_pos = self.goal_body.position if self.goal_body else (0, 0)
        
        # Calculate distances and angles
        dist_to_obj = math.sqrt((car_pos.x - obj_pos.x)**2 + (car_pos.y - obj_pos.y)**2)
        angle_to_obj = math.atan2(obj_pos.y - car_pos.y, obj_pos.x - car_pos.x) - self.car_body.angle
        dist_to_goal = math.sqrt((obj_pos.x - goal_pos.x)**2 + (obj_pos.y - goal_pos.y)**2)
        
        # Normalize angles to [-pi, pi]
        angle_to_obj = (angle_to_obj + math.pi) % (2 * math.pi) - math.pi
        
        # Get sonar readings (8 features)
        sonar_readings = self.get_sonar_readings(car_pos.x, car_pos.y, self.car_body.angle)
        
        # Create feature vector (16 features total)
        features = [
            # Sonar readings (8)
            *sonar_readings,
            
            # Object features (3)
            dist_to_obj / 1000.0,  # Normalize by environment width
            math.sin(angle_to_obj),
            math.cos(angle_to_obj),
            
            # Goal features (1)
            dist_to_goal / 1000.0,
            
            # Contact state (1)
            1.0 if self.is_pushing else 0.0,
            
            # Collision count (1)
            self.collision_count / 10.0,
            
            # Goal state (1)
            1.0 if self.is_in_goal else 0.0,
            
            # Crashed state (1)
            1.0 if self.crashed else 0.0
        ]
        
        return features


if __name__ == "__main__":
    # Can work with either weights format
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    game_state = GameState(weights)
    while True:
        game_state.frame_step((random.randint(0, 3)))
