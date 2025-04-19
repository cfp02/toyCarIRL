import argparse
import json
import os
from typing import List

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from flat_game import carmunk
from nn import DQNAgent


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
):
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

    # Track best performance
    best_reward = float("-inf")

    print(f"Training with track file: {env_path}")
    env = carmunk.GameState(env_weights, env_path)

    progress_bar = tqdm(range(start_episode, num_episodes), desc="Training")
    from collections import deque

    # Conditons for early stopping
    early_stop_window = 30
    recent_rewards = deque(maxlen=early_stop_window)

    for episode in progress_bar:
        # Reset environment
        action = 2
        _, state, _, _ = env.frame_step(action)
        state = state.reshape(-1)

        episode_reward = 0
        episode_loss = 0
        steps = 0

        recent_rewards.append(episode_reward)
        if len(recent_rewards) == early_stop_window:
            # Calculate mean and standard deviation of recent rewards
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)

            # Check if the standard deviation is small enough (rewards are stable)
            converged = (
                std_reward <= 0.1 * abs(mean_reward)
                if mean_reward != 0
                else std_reward <= 1.0
            )

            # Log the convergence progress
            if (episode + 1) % log_interval == 0:
                print(
                    f"Convergence check: mean={mean_reward:.2f}, std={std_reward:.2f}, "
                    f"{'stable' if converged else 'not stable yet'}"
                )

            # Check if we've met our convergence criteria
            if converged:
                print(
                    f"\n✅ Early stopping: Agent has converged with stable rewards "
                    f"(mean={mean_reward:.2f}, std={std_reward:.2f})"
                )
                agent.save(os.path.join(checkpoint_dir, "converged_model.pth"))
                break

        while steps < max_steps_per_episode:
            # Select action
            action = agent.act(state)
            reward, next_state, _, _ = env.frame_step(action)
            next_state = next_state.reshape(-1)

            # Store transition in memory
            # Check if we're near the end of the episode
            done = (max_steps_per_episode - steps) <= 2

            agent.memorize(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state

            # Learn
            loss = agent.learn()
            if loss != 0:  # Only count loss if learning happened
                episode_loss += loss

            # Update epsilon
            agent.update_epsilon(total_frames)

            episode_reward += reward
            steps += 1
            total_frames += 1

        # Track rewards
        all_rewards.append(episode_reward)
        all_lengths.append(steps)

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
        writer.add_scalar("Train/Length", steps, episode)
        writer.add_scalar("Train/Epsilon", agent.epsilon, episode)
        if steps > 0:
            writer.add_scalar("Train/AvgLoss", episode_loss / steps, episode)

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
    return all_rewards, all_lengths


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


if __name__ == "__main__":
    args = parse_args()

    weights = [
        3.96957075e-01,
        1.60768375e-01,
        3.85956376e-01,
        2.17107187e-01,
        5.31013934e-01,
        1.89519667e-01,
        0.00000000e00,
        6.23592131e-02,
        5.14246354e-01,
        3.43368382e-02,
    ]

    # Number of inputs (from the environment)
    NUM_INPUT = len(weights)

    # Create agent
    agent = DQNAgent(
        state_size=NUM_INPUT,
        action_size=3,  # Left, Right, No-op
        hidden_sizes=args.hidden_sizes,
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
        reward_scaling=args.reward_scaling,
    )

    # Verify that the environment file exists
    env_path = args.env_path
    if not os.path.exists(env_path):
        print(
            f"Warning: Track file {env_path} not found, checking alternative paths..."
        )
        # Try with tracks/ prefix
        alt_path = os.path.join("tracks", env_path)
        if os.path.exists(alt_path):
            env_path = alt_path
            print(f"Found track at: {env_path}")
        else:
            # Try with .json extension
            alt_path = os.path.join("tracks", f"{env_path}.json")
            if os.path.exists(alt_path):
                env_path = alt_path
                print(f"Found track at: {env_path}")
            else:
                print("⚠️ Could not find track file, using default: tracks/default.json")
                env_path = "tracks/default.json"

    print(f"🚗 Using track file: {os.path.abspath(env_path)}")

    if args.mode == "train":
        # If model path provided and not resuming, load it
        if args.model_path and not args.resume:
            agent.load(args.model_path)

        # Train the agent with the specified environment
        train(
            agent=agent,
            env_weights=weights,
            env_path=env_path,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            resume=args.resume,
        )

    elif args.mode == "eval":
        # Load model for evaluation
        model_path = args.model_path or "saved-models/checkpoints/best_model.pth"
        if not agent.load(model_path):
            print("Cannot evaluate without a trained model.")
            exit(1)

        # Evaluate the agent with the specified environment
        evaluate(
            agent=agent,
            env_weights=weights,
            env_path=env_path,
            num_episodes=args.episodes,
        )
