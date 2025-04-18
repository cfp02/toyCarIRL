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
show_sensors = flag
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
        self.W = weights  # weights for the reward function

        # Record steps.
        self.num_steps = 0
        self.num_obstacles_type = 5  # Track 5 types consistently

        # Set up collision handler
        self.collision_handler = self.space.add_collision_handler(1, 2)
        self.collision_handler.begin = self.handle_collision

        self.load_environment(obstacle_file)

    def handle_collision(self, arbiter, space, data):
        """Handle collision between car and obstacles"""
        self.crashed = True
        # Increment collision counter
        self.collision_count += 1
        # Allow the collision to occur and generate response force
        return True

    def load_environment(self, obstacle_file="tracks/default.json"):
        """Load obstacles and car position from a file if provided, otherwise use defaults."""
        self.obstacles = []

        if obstacle_file and os.path.exists(obstacle_file):
            try:
                with open(obstacle_file, "r") as f:
                    config = json.load(f)

                # Load car configuration
                car_config = config.get(
                    "car_start", {"x": 150, "y": 20, "r": 15, "angle": 1.4}
                )
                self.create_car(car_config["x"], car_config["y"], car_config["r"])
                self.car_body.angle = car_config.get("angle", 1.4)

                # Load walls
                self.create_boundary_walls()

                # Load obstacles
                for wall in config.get("walls", []):
                    self.obstacles.append(
                        self.create_obstacle(
                            wall["xy1"],
                            wall["xy2"],
                            wall.get("radius", 7),
                            wall.get("color", "yellow"),
                        )
                    )

                print(f"Loaded environment from {obstacle_file}")

            except Exception as e:
                print(f"Error loading obstacle file: {e}")
        else:
            # If no file, create default setup
            self.create_car(150, 20, 15)
            self.create_boundary_walls()

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

    def frame_step(self, action):
        if action == 0:  # Turn left.
            self.car_body.angle -= 0.3
        elif action == 1:  # Turn right.
            self.car_body.angle += 0.3

        # Reset crash status at the start of each frame
        self.crashed = False

        # Get driving direction and apply velocity
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)

        # Set velocity without overriding physics response from collisions
        if not self.crashed:
            self.car_body.velocity = 100 * driving_direction

        # Take multiple smaller physics steps for better collision detection
        for _ in range(5):  # 5 substeps for better accuracy
            self.space.step(1.0 / 50)  # 5 substeps per frame = 1/10 sec total

        # Update the screen
        screen.fill(THECOLORS["black"])
        self.space.debug_draw(draw_options)

        if draw_screen:
            pygame.display.flip()
        clock.tick()

        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)

        # Check if car crashed based on sonar readings
        if self.car_is_crashed(readings) or self.crashed:
            self.crashed = True
            readings.append(1)
            # Slow down the car when crashed
            self.car_body.velocity *= 0.5
        else:
            readings.append(0)

        # Ensure car doesn't go out of bounds
        self.check_bounds()

        # Calculate base reward from features
        base_reward = np.dot(self.W, readings)

        # Add collision penalty to reward
        collision_penalty = 0
        if self.crashed and not self.was_crashed_last_frame:
            # Only apply penalty on new crashes
            collision_penalty = -10.0  # Significant penalty for each collision

        # Store crash state for next frame
        self.was_crashed_last_frame = self.crashed

        # Combined reward
        reward = base_reward + collision_penalty
        state = np.array([readings])

        self.num_steps += 1

        # Return collision count along with other information
        return reward, state, readings, self.collision_count

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

        # Always include red boundary walls as a standard feature
        readings.append(float(ObstacleNumber[4] / 3.0))  # Red boundary walls

        # Add normalized collision count as a feature
        # Normalize by dividing by a reasonable maximum (e.g., 10)
        readings.append(min(1.0, self.collision_count / 10.0))

        if show_sensors:
            pygame.display.update()

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

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

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


if __name__ == "__main__":
    # Can work with either weights format
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    game_state = GameState(weights)
    while True:
        game_state.frame_step((random.randint(0, 2)))
