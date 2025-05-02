import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch
from torch import nn, optim

from flat_game import carmunk
from nn import DQNAgent
from metrics import IRLTracker
from utils import TrajectoryRecorder, NUM_STATES, GAMMA


def train(
    agent: DQNAgent,
    env_weights: List[float],
    env_path: str,
    num_episodes: int,
    max_steps_per_episode: int,
    log_interval: int = 10,
    save_interval: int = 100,
    checkpoint_dir: str = "saved-models/checkpoints",
    log_dir: str = "logs",
    resume: bool = False,
    tracker=None,
    iteration=None,
    game_state=None,
    recorder=None,
) -> Tuple[List[float], List[float]]:
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    writer = SummaryWriter(log_dir)

    # Resume from checkpoint if requested
    start_episode = 0
    if resume:
        checkpoint_path = os.path.join(checkpoint_dir, "latest.pth")
        if agent.load(checkpoint_path):
            # Find the latest episode number
            with open(os.path.join(log_dir, "progress.json"), "r") as f:
                progress = json.load(f)
                start_episode = progress.get("last_episode", 0) + 1

    total_frames = 0
    all_rewards = []
    all_lengths = []
    all_losses = []  # Track losses

    # Track best performance
    best_reward = float("-inf")

    print(f"Training with track file: {env_path}")

    # Initialize environment if not provided
    if game_state is None:
        game_state = carmunk.GameState(env_weights, env_path)
        game_state.load_environment(env_path)
        game_state.reset()

    # Initialize trajectory recorder if not provided
    if recorder is None:
        recorder = TrajectoryRecorder(env_weights, "training")

    progress_bar = tqdm(range(start_episode, num_episodes), desc="Training")
    from collections import deque

    # Conditons for early stopping
    early_stop_window = 40
    recent_rewards = deque(maxlen=early_stop_window)

    # Define a frequency for updating the tracker during training
    training_update_interval = max(
        1, num_episodes // 20
    )  # Update ~20 times during training

    # Initialize metrics
    episode_rewards = []  # Track rewards for each episode
    episode_losses = []   # Track losses for each episode

    for episode in progress_bar:
        # Reset environment and recorder
        _, state, _, _ = game_state.frame_step(2)  # Start with forward motion
        state = state.reshape(-1)  # Flatten the state
        recorder.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0

        while steps < max_steps_per_episode:
            # Choose action
            action = agent.act(state)

            # Take action
            reward, next_state, features, collision_count = game_state.frame_step(action)
            next_state = next_state.reshape(-1)  # Flatten the next state
            done = game_state.is_in_goal or collision_count > 5

            # Record step
            recorder.record_step(
                state, action, features, collision_count, game_state.is_in_goal
            )

            # Store experience
            agent.memorize(state, action, reward, next_state, done)

            # Learn from experience
            loss = agent.learn()
            if loss is not None:
                episode_loss += loss

            # Update state and metrics
            state = next_state
            episode_reward += reward
            steps += 1

            # Check if episode is done
            if done:
                break

        # Update metrics
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss / steps if steps > 0 else 0)

        # Update tracker if provided
        if tracker:
            tracker.add_iteration_data(
                iteration=iteration,
                t_value=None,  # This will be updated by the IRL process
                weights=env_weights,
                fe_distances=[],  # This will be updated by the IRL process
                best_model_path=None,  # This will be updated by the IRL process
                avg_reward=float(episode_reward),
                avg_loss=float(episode_loss / steps if steps > 0 else 0)
            )

        # Save checkpoint if this is the best episode so far
        if episode_reward > best_reward and checkpoint_dir:
            best_reward = episode_reward
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir, f"episode_{episode}_reward_{episode_reward:.2f}.pt"
            )
            agent.save(checkpoint_path)

        # Track rewards and losses
        all_rewards.append(episode_reward)
        all_lengths.append(steps)
        avg_loss = episode_loss / steps if steps > 0 else 0
        all_losses.append(avg_loss)  # Store average loss for the episode

        # Update progress bar
        progress_bar.set_postfix(
            {
                "reward": f"{episode_reward:.2f}",
                "steps": steps,
                "epsilon": f"{agent.epsilon:.2f}",
                "avg_reward": f"{np.mean(all_rewards[-100:]):.2f}"
                if all_rewards
                else "N/A",
            }
        )

        # Logging
        writer.add_scalar("Train/Reward", episode_reward, episode)
        if steps > 0:
            writer.add_scalar("Train/AvgLoss", avg_loss, episode)

        # Update tracker during training (real-time updates)
        if tracker and (episode + 1) % training_update_interval == 0:
            # Calculate current metrics
            avg_reward_window = (
                np.mean(all_rewards[-100:])
                if len(all_rewards) >= 100
                else np.mean(all_rewards)
            )
            avg_loss_window = (
                np.mean(all_losses[-100:])
                if len(all_losses) >= 100
                else np.mean(all_losses)
            )

            # Add training progress data point
            tracker.add_training_progress(
                iteration=iteration,  # Current IRL iteration
                episode=episode,  # Current training episode
                avg_reward=avg_reward_window,
                avg_loss=avg_loss_window,
            )

            # Generate updated plots
            tracker.plot_training_progress()

        # Log to console
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(all_rewards[-log_interval:])
            avg_length = np.mean(all_lengths[-log_interval:])
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Length: {avg_length:.2f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

            # Save progress
            progress = {
                "last_episode": episode,
                "total_frames": total_frames,
                "best_reward": best_reward,
                "current_avg_reward": avg_reward,
            }
            with open(os.path.join(log_dir, "progress.json"), "w") as f:
                json.dump(progress, f)

        # Save checkpoints
        if (episode + 1) % save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_ep{episode + 1}.pth"))
            agent.save(os.path.join(checkpoint_dir, "latest.pth"))

        # Save best model
        if len(all_rewards) >= 100:
            avg_reward = np.mean(all_rewards[-100:])
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(os.path.join(checkpoint_dir, "best_model.pth"))
                print(f"New best model saved with average reward: {best_reward:.2f}")

    # Final save
    agent.save(os.path.join(checkpoint_dir, "final_model.pth"))

    # Close logger
    writer.close()

    print("Training complete!")
    return episode_rewards, episode_losses  # Return both rewards and losses


def evaluate(
    agent: DQNAgent,
    env_weights: List[float],
    env_path: str,
    num_episodes: int = 10,
    render: bool = True,
):
    # Create environment with specified track file
    print(f"Evaluating with track file: {env_path}")
    env = carmunk.GameState(env_weights, env_path)

    rewards = []
    lengths = []

    for episode in range(num_episodes):
        # Reset environment
        action = 2
        _, state, _, collisions = env.frame_step(action)
        state = state.reshape(-1)

        episode_reward = 0
        steps = 0

        while collisions <= 5:  # allow some collisons to learn recovery behavior
            # Select action without exploration
            action = agent.act(state, evaluate=True)
            reward, next_state, _, next_collisions = env.frame_step(action)
            next_state = next_state.reshape(-1)  # Flatten

            collisions = next_collisions

            # Move to the next state
            state = next_state

            episode_reward += reward
            steps += 1

        rewards.append(episode_reward)
        lengths.append(steps)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

    avg_reward = np.mean(rewards)
    avg_length = np.mean(lengths)

    print("\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Length: {avg_length:.2f}")

    return rewards, lengths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate a DQN agent on the car environment"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode to run in",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[164, 150],
        help="Hidden layer sizes",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--buffer-size", type=int, default=100000, help="Replay buffer size"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=1000,
        help="Target network update frequency",
    )
    parser.add_argument(
        "--no-prioritized",
        action="store_true",
        help="Disable prioritized experience replay",
    )
    parser.add_argument("--no-double", action="store_true", help="Disable double DQN")
    parser.add_argument(
        "--reward-scaling", type=float, default=1.0, help="Scaling factor for rewards"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to load model from"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--env-path", type=str, default="tracks/default.json", help="Path to track file"
    )
    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the car agent")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Mode to run in")
    parser.add_argument("--env-path", type=str, default="tracks/default.json", help="Path to environment configuration")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[164, 150], help="Hidden layer sizes")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()

    # Initialize weights for the reward function
    # Original weights: 10 dimensions
    # New weights for pushing task: 16 dimensions
    # [sonar1, sonar2, sonar3, black_space, yellow_obstacles, brown_obstacles, out_of_bounds, red_walls, collision_count,
    #  distance_to_object, sin_angle_to_object, cos_angle_to_object, distance_to_goal, is_pushing, is_in_goal, crashed]
    env_weights = [1.0] * 16  # Equal weights for all features

    # Initialize agent
    state_size = 16  # Updated state size
    action_size = 3  # Left, Right, Straight
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_sizes=args.hidden_sizes,
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        is_pushing_task=True  # Enable pushing task mode
    )

    if args.mode == "train":
        train(
            agent=agent,
            env_weights=env_weights,
            env_path=args.env_path,
            num_episodes=args.episodes,
            max_steps_per_episode=1000,
            resume=args.resume
        )
    else:
        evaluate(
            agent=agent,
            env_weights=env_weights,
            env_path=args.env_path
        )


if __name__ == "__main__":
    main()
