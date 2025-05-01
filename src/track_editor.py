import argparse
import json
import os

import pygame
from pygame.color import THECOLORS

pygame.init()
width, height = 1000, 700
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Track Editor")

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
goal_radius = 20


def save_track(filename, walls, goals, car_pos, car_radius):
    """Save track configuration to a file."""
    # Create tracks directory if it doesn't exist
    os.makedirs("tracks", exist_ok=True)

    # Prepare path in tracks directory
    filepath = os.path.join("tracks", os.path.basename(filename))

    data = {
        "name": os.path.basename(filename).split(".")[0],
        "walls": walls,
        "goals": goals,
        "car_start": {"x": car_pos[0], "y": car_pos[1], "r": car_radius, "angle": 1.4},
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Track saved to {filepath}")


def load_track(filename):
    """Load track configuration from a file."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        walls = data.get("walls", [])
        goals = data.get("goals", [])
        car_config = data.get("car_start", {"x": 150, "y": 20, "r": 15})
        return walls, goals, [car_config["x"], car_config["y"]], car_config["r"]
    except Exception as e:
        print(f"Error loading track: {e}")
        return [], [], [150, 20], 15


def main():
    parser = argparse.ArgumentParser(description="Track Editor")
    parser.add_argument("--load", type=str, help="Load track from file")
    parser.add_argument(
        "--save", type=str, default="track.json", help="Save track to file"
    )
    args = parser.parse_args()

    # Initialize variables
    global current_color, car_pos, car_radius, goal_radius, current_mode
    walls = []
    goals = []
    start_pos = None
    current_action = None
    current_mode = "car"
    resizing_goal = None

    # Load track if specified
    if args.load:
        walls, goals, car_pos, car_radius = load_track(args.load)

    # Draw info text
    font = pygame.font.SysFont("Arial", 16)

    # Main loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle mouse events
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = event.pos

                    if current_mode == "car":
                        car_pos = list(mouse_pos)
                    elif current_mode == "goal":
                        # Check if we're clicking on an existing goal to resize
                        clicked_goal = None
                        for i, goal in enumerate(goals):
                            distance = (
                                (goal["x"] - mouse_pos[0]) ** 2
                                + (goal["y"] - mouse_pos[1]) ** 2
                            ) ** 0.5
                            if distance <= goal["radius"]:
                                clicked_goal = i
                                break

                        if clicked_goal is not None:
                            # Start resizing the goal
                            resizing_goal = clicked_goal
                            current_action = "resizing_goal"
                        else:
                            # Add a new goal
                            goals.append(
                                {
                                    "x": mouse_pos[0],
                                    "y": mouse_pos[1],
                                    "radius": goal_radius,
                                }
                            )
                    elif current_mode == "wall":
                        # Start drawing a wall
                        start_pos = mouse_pos
                        current_action = "drawing"

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if current_action == "drawing" and start_pos:
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

                    # Stop resizing goals
                    if current_action == "resizing_goal":
                        resizing_goal = None

                    current_action = None

            elif event.type == pygame.MOUSEMOTION:
                # Handle resizing goals
                if current_action == "resizing_goal" and resizing_goal is not None:
                    goal_center = (goals[resizing_goal]["x"], goals[resizing_goal]["y"])
                    mouse_pos = event.pos
                    # Calculate new radius based on distance to mouse
                    new_radius = max(
                        10,
                        (
                            (goal_center[0] - mouse_pos[0]) ** 2
                            + (goal_center[1] - mouse_pos[1]) ** 2
                        )
                        ** 0.5,
                    )
                    goals[resizing_goal]["radius"] = new_radius

            # Handle keyboard events
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    save_track(args.save, walls, goals, car_pos, car_radius)
                elif (
                    event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL
                ):
                    if current_mode == "wall" and walls:
                        walls.pop()  # Undo last wall
                    elif current_mode == "goal" and goals:
                        goals.pop()  # Undo last goal
                elif event.key == pygame.K_1:
                    current_color = "yellow"
                elif event.key == pygame.K_2:
                    current_color = "brown"
                elif event.key == pygame.K_3:
                    current_color = "red"
                elif event.key == pygame.K_w:
                    current_mode = "wall"
                    print("Mode: Wall drawing")
                elif event.key == pygame.K_g:
                    current_mode = "goal"
                    print("Mode: Goal placement")
                elif event.key == pygame.K_c:
                    current_mode = "car"
                    print("Mode: Car placement")
                elif event.key == pygame.K_DELETE:
                    # Delete goals if in goal mode and mouse over a goal
                    if current_mode == "goal":
                        mouse_pos = pygame.mouse.get_pos()
                        for i, goal in enumerate(goals):
                            distance = (
                                (goal["x"] - mouse_pos[0]) ** 2
                                + (goal["y"] - mouse_pos[1]) ** 2
                            ) ** 0.5
                            if distance <= goal["radius"]:
                                del goals[i]
                                print(f"Deleted goal #{i}")
                                break

        # Draw background
        screen.fill(COLORS["black"])

        # Draw walls
        for wall in walls:
            pygame.draw.line(
                screen, COLORS[wall["color"]], wall["xy1"], wall["xy2"], wall["radius"]
            )

        # Draw goals
        for goal in goals:
            pygame.draw.circle(
                screen, COLORS["green"], (goal["x"], goal["y"]), goal["radius"], 2
            )
            # Draw a dot at the center for clarity
            pygame.draw.circle(screen, COLORS["green"], (goal["x"], goal["y"]), 3)

        # Draw car position
        car_color = COLORS["green"] if current_mode != "car" else THECOLORS["cyan"]
        pygame.draw.circle(screen, car_color, car_pos, car_radius)

        # Draw currently drawing line
        if current_action == "drawing" and start_pos:
            pygame.draw.line(
                screen,
                COLORS[current_color],
                start_pos,
                pygame.mouse.get_pos(),
                wall_radius,
            )

        # Draw UI info text
        mode_text = f"Mode: {'Wall' if current_mode == 'wall' else 'Goal' if current_mode == 'goal' else 'Car'}"
        controls_text = "Controls: W=Wall mode, G=Goal mode, C=Car mode, Del=Delete, 1-3=Wall colors"
        save_text = "Ctrl+S to save, Ctrl+Z to undo"

        mode_surface = font.render(mode_text, True, THECOLORS["white"])
        controls_surface = font.render(controls_text, True, THECOLORS["white"])
        save_surface = font.render(save_text, True, THECOLORS["white"])

        screen.blit(mode_surface, (10, 10))
        screen.blit(controls_surface, (10, 30))
        screen.blit(save_surface, (10, 50))

        # Update display
        pygame.display.flip()

    pygame.quit()
    print("Editor closed")


if __name__ == "__main__":
    main()
