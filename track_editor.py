import argparse
import json
import os

import pygame
from pygame.color import THECOLORS

# Initialize Pygame
pygame.init()
width, height = 1000, 700
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Track Editor")

# Colors and settings
COLORS = {
    "yellow": THECOLORS["yellow"],
    "brown": THECOLORS["brown"],
    "red": THECOLORS["red"],
    "black": THECOLORS["black"],
    "white": THECOLORS["white"],
    "green": THECOLORS["green"],
}

current_color = "yellow"
wall_radius = 7
car_pos = [150, 20]
car_radius = 15


def save_track(filename, walls, car_pos, car_radius):
    """Save track configuration to a file."""
    data = {
        "name": os.path.basename(filename).split(".")[0],
        "walls": walls,
        "car_start": {"x": car_pos[0], "y": car_pos[1], "r": car_radius, "angle": 1.4},
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Track saved to {filename}")


def load_track(filename):
    """Load track configuration from a file."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        walls = data.get("walls", [])
        car_config = data.get("car_start", {"x": 150, "y": 20, "r": 15})
        return walls, [car_config["x"], car_config["y"]], car_config["r"]
    except Exception as e:
        print(f"Error loading track: {e}")
        return [], [150, 20], 15


def main():
    parser = argparse.ArgumentParser(description="Track Editor")
    parser.add_argument("--load", type=str, help="Load track from file")
    parser.add_argument(
        "--save", type=str, default="track.json", help="Save track to file"
    )
    args = parser.parse_args()

    # Initialize variables
    global current_color, car_pos, car_radius
    walls = []
    start_pos = None
    current_action = None

    # Load track if specified
    if args.load:
        walls, car_pos, car_radius = load_track(args.load)

    # Main loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle mouse events for drawing walls
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:  # Move car position
                        car_pos = list(event.pos)
                    else:  # Start drawing a wall
                        start_pos = event.pos
                        current_action = "drawing"

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and current_action == "drawing" and start_pos:
                    end_pos = event.pos
                    # Add wall to list if it has some length
                    if start_pos[0] != end_pos[0] or start_pos[1] != end_pos[1]:
                        walls.append(
                            {
                                "xy1": list(start_pos),
                                "xy2": list(end_pos),
                                "radius": wall_radius,
                                "color": current_color,
                            }
                        )
                    start_pos = None
                    current_action = None

            # Handle keyboard events
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    save_track(args.save, walls, car_pos, car_radius)
                elif (
                    event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL
                ):
                    if walls:  # Undo last wall
                        walls.pop()
                elif event.key == pygame.K_1:
                    current_color = "yellow"
                elif event.key == pygame.K_2:
                    current_color = "brown"
                elif event.key == pygame.K_3:
                    current_color = "red"

        # Draw background
        screen.fill(COLORS["black"])

        # Draw walls
        for wall in walls:
            pygame.draw.line(
                screen, COLORS[wall["color"]], wall["xy1"], wall["xy2"], wall["radius"]
            )

        # Draw car position
        pygame.draw.circle(screen, COLORS["green"], car_pos, car_radius)

        # Draw currently drawing line
        if current_action == "drawing" and start_pos:
            pygame.draw.line(
                screen,
                COLORS[current_color],
                start_pos,
                pygame.mouse.get_pos(),
                wall_radius,
            )

        # Update display
        pygame.display.flip()

    pygame.quit()
    print("Editor closed")


if __name__ == "__main__":
    main()
