from typing import List, Optional, Tuple, Union, Any

import numpy as np
import pygame
import pymunk

class GameState:
    space: pymunk.Space
    car_body: pymunk.Body
    car_shape: pymunk.Circle
    obstacles: List[pymunk.Body]
    W: List[float]
    crashed: bool
    num_steps: int
    num_obstacles_type: int
    collision_count: int
    was_crashed_last_frame: bool

    def __init__(
        self, weights: List[float], obstacle_file: Optional[str] = None
    ) -> None: ...
    
    def handle_collision(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: Any) -> bool: ...
    
    def load_environment(self, obstacle_file: Optional[str] = None) -> None: ...
    
    def create_boundary_walls(self) -> None: ...
    
    def create_obstacle(
        self, xy1: Tuple[float, float], xy2: Tuple[float, float], r: float, color: str
    ) -> pymunk.Body: ...
    
    def create_cat(self) -> None: ...
    
    def create_car(self, x: float, y: float, r: float) -> None: ...
    
    def frame_step(self, action: int) -> Tuple[float, np.ndarray, List[float], int]: ...
    
    def check_bounds(self) -> None: ...
    
    def move_obstacles(self) -> None: ...
    
    def move_cat(self) -> None: ...
    
    def car_is_crashed(self, readings: List[float]) -> bool: ...
    
    def recover_from_crash(self, driving_direction: pymunk.Vec2d) -> None: ...
    
    def get_sonar_readings(self, x: float, y: float, angle: float) -> List[float]: ...
    
    def get_arm_distance(
        self,
        arm: List[Tuple[float, float]],
        x: float,
        y: float,
        angle: float,
        offset: float,
    ) -> List[Union[int, float]]: ...
    
    def make_sonar_arm(self, x: float, y: float) -> List[Tuple[float, float]]: ...
    
    def get_rotated_point(
        self, x_1: float, y_1: float, x_2: float, y_2: float, radians: float
    ) -> Tuple[int, int]: ...
    
    def get_track_or_not(self, reading: pygame.Color) -> int: ...