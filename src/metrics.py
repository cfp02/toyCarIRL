# metrics.py
import json
import os
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


class IRLTracker:
    def __init__(self, behavior: str, track_file: str, num_frames: int, run_id: str = None):
        self.behavior = behavior
        self.track_name = os.path.splitext(os.path.basename(track_file))[0]
        self.num_frames = num_frames
        self.run_id = run_id or f"run_{int(time.time())}"

        # Create output directories
        self.output_dir = os.path.join("metrics", f"{behavior}_{self.track_name}", self.run_id)
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        # Initialize data containers
        self.iterations_data = []
        self.training_progress = []  # For tracking training progress within iterations

        # Save file paths
        self.iterations_file = os.path.join(self.output_dir, "iterations.json")
        self.training_file = os.path.join(self.output_dir, "training_progress.json")

        # Load existing data if available
        self._load_data()

        # Generate a timestamp for this run
        self.timestamp = int(time.time())

    def _load_data(self):
        """Load existing data from files if they exist."""
        try:
            if os.path.exists(self.iterations_file):
                with open(self.iterations_file, "r") as f:
                    self.iterations_data = json.load(f)
                print(f"Loaded {len(self.iterations_data)} iterations from file.")
        except Exception as e:
            print(f"Error loading iterations data: {e}")
            self.iterations_data = []

        try:
            if os.path.exists(self.training_file):
                with open(self.training_file, "r") as f:
                    self.training_progress = json.load(f)
                print(
                    f"Loaded {len(self.training_progress)} training progress points from file."
                )
        except Exception as e:
            print(f"Error loading training progress data: {e}")
            self.training_progress = []

    def _save_data(self):
        """Save current data to files."""
        # Save iterations data
        try:
            with open(self.iterations_file, "w") as f:
                json.dump(self.iterations_data, f, indent=2)
        except Exception as e:
            print(f"Error saving iterations data: {e}")

        # Save training progress data
        try:
            with open(self.training_file, "w") as f:
                json.dump(self.training_progress, f, indent=2)
        except Exception as e:
            print(f"Error saving training progress data: {e}")

    def add_iteration_data(
        self,
        iteration: int,
        t_value: float,
        weights: List[float],
        fe_distances: List[float],
        best_model_path: Optional[str] = None,
        avg_reward: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ):
        """Add data for a complete IRL iteration."""
        data = {
            "iteration": iteration,
            "timestamp": int(time.time()),
            "t_value": t_value,
            "weights": weights.tolist() if isinstance(weights, np.ndarray) else weights,
            "fe_distances": fe_distances,
            "best_model_path": best_model_path,
            "avg_reward": avg_reward,
            "avg_loss": avg_loss,
        }

        # Check if this iteration already exists and update it
        for i, existing in enumerate(self.iterations_data):
            if existing["iteration"] == iteration:
                self.iterations_data[i] = data
                break
        else:
            # Add as a new iteration
            self.iterations_data.append(data)

        # Save to file
        self._save_data()

    def add_training_progress(
        self,
        iteration: int,
        episode: int,
        avg_reward: float,
        avg_loss: float,
    ):
        """Add data point for training progress within an iteration."""
        data = {
            "iteration": iteration,
            "episode": episode,
            "timestamp": int(time.time()),
            "avg_reward": avg_reward,
            "avg_loss": avg_loss,
        }

        self.training_progress.append(data)
        self._save_data()

    def plot_metrics(self):
        """Generate plots for IRL iteration metrics."""
        if not self.iterations_data:
            print("No iteration data available for plotting")
            return

        # Sort data by iteration
        sorted_data = sorted(self.iterations_data, key=lambda x: x["iteration"])

        # Extract data for plotting
        iterations = [d["iteration"] for d in sorted_data]
        t_values = [d["t_value"] for d in sorted_data]
        rewards = [d.get("avg_reward", 0) for d in sorted_data]
        losses = [d.get("avg_loss", 0) for d in sorted_data]

        # Create plots
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))

        # Plot T values over iterations
        axs[0].plot(iterations, t_values, "bo-", linewidth=2)
        axs[0].set_title("IRL Progress: T-Value vs Iteration")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("T-Value (lower is better)")
        axs[0].grid(True)

        # Plot rewards and losses over iterations
        ax1 = axs[1]
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Average Reward", color="tab:blue")
        ax1.plot(iterations, rewards, "b-", label="Reward")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Average Loss", color="tab:red")
        ax2.plot(iterations, losses, "r-", label="Loss")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "irl_metrics.png"))
        plt.close()

        # Plot weight evolution
        self._plot_weights()

    def plot_training_progress(self):
        """Generate plots for real-time training progress."""
        if not self.training_progress:
            print("No training progress data available for plotting")
            return

        # Group data by iteration
        iterations = {}
        for data in self.training_progress:
            iter_num = data["iteration"]
            if iter_num not in iterations:
                iterations[iter_num] = []
            iterations[iter_num].append(data)

        # Plot learning curves for each iteration
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        # Use different colors for different iterations
        colormap = plt.get_cmap("tab10")
        colors = [colormap(i) for i in range(10)]  # Get 10 colors from the colormap

        # Plot rewards
        ax = axs[0, 0]
        for i, iter_num in enumerate(sorted(iterations.keys())):
            iter_data = sorted(iterations[iter_num], key=lambda x: x["episode"])
            episodes = [d["episode"] for d in iter_data]
            rewards = [d["avg_reward"] for d in iter_data]
            color = colors[i % len(colors)]
            ax.plot(episodes, rewards, "-", color=color, label=f"Iter {iter_num}")

        ax.set_title("Average Reward During Training")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Reward")
        ax.grid(True)
        ax.legend()

        # Plot losses
        ax = axs[0, 1]
        for i, iter_num in enumerate(sorted(iterations.keys())):
            iter_data = sorted(iterations[iter_num], key=lambda x: x["episode"])
            episodes = [d["episode"] for d in iter_data]
            losses = [d["avg_loss"] for d in iter_data]
            color = colors[i % len(colors)]
            ax.plot(episodes, losses, "-", color=color, label=f"Iter {iter_num}")

        ax.set_title("Average Loss During Training")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Loss")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "training_progress.png"))
        plt.close()

        # Plot convergence metrics over time
        self._plot_convergence_over_time()

    def _plot_weights(self):
        """Plot the evolution of weights across iterations."""
        if not self.iterations_data:
            return

        # Sort data by iteration
        sorted_data = sorted(self.iterations_data, key=lambda x: x["iteration"])

        # Get the number of weight dimensions
        if "weights" not in sorted_data[0] or not sorted_data[0]["weights"]:
            return

        n_dims = len(sorted_data[0]["weights"])
        iterations = [d["iteration"] for d in sorted_data]

        # Create a plot with one line per weight dimension
        plt.figure(figsize=(12, 6))

        for i in range(n_dims):
            weights = [d["weights"][i] for d in sorted_data]
            plt.plot(iterations, weights, label=f"w{i + 1}")

        plt.title("Weight Evolution Across Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Weight Value")
        plt.grid(True)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "weight_evolution.png"))
        plt.close()

    def _plot_convergence_over_time(self):
        """Plot convergence metrics over wall-clock time."""
        if not self.training_progress:
            return

        # Calculate cumulative time
        start_time = min(d["timestamp"] for d in self.training_progress)

        # Initialize plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by iteration
        iterations = {}
        for data in self.training_progress:
            iter_num = data["iteration"]
            if iter_num not in iterations:
                iterations[iter_num] = []

            # Convert timestamp to hours from start
            hours = (data["timestamp"] - start_time) / 3600

            # Store the data with time info
            item = data.copy()
            item["hours_elapsed"] = hours
            iterations[iter_num].append(item)

        # Plot rewards over time
        colormap = plt.get_cmap("tab10")
        colors = [colormap(i) for i in range(10)]  # Get 10 colors from the colormap
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

        for i, iter_num in enumerate(sorted(iterations.keys())):
            iter_data = sorted(iterations[iter_num], key=lambda x: x["timestamp"])
            hours = [d["hours_elapsed"] for d in iter_data]
            rewards = [d["avg_reward"] for d in iter_data]

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            ax.plot(
                hours,
                rewards,
                "-",
                color=color,
                marker=marker,
                markersize=5,
                label=f"Iteration {iter_num}",
            )

            # Mark the end of each iteration
            if hours:
                ax.plot(hours[-1], rewards[-1], "o", color=color, markersize=10)

        # Add vertical lines between iterations using iteration data if available
        if len(self.iterations_data) > 1:
            iter_timestamps = sorted([d["timestamp"] for d in self.iterations_data])
            for ts in iter_timestamps[:-1]:  # Skip the last one
                hours = (ts - start_time) / 3600
                ax.axvline(x=hours, color="gray", linestyle="--", alpha=0.5)

        ax.set_title("Training Progress Over Time")
        ax.set_xlabel("Hours Elapsed")
        ax.set_ylabel("Average Reward")
        ax.grid(True)
        ax.legend(loc="best")

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "time_convergence.png"))
        plt.close()

    def generate_summary_report(self):
        """Generate a summary report of the IRL training process."""
        if not self.iterations_data:
            print("No data available for summary report")
            return

        report_path = os.path.join(self.output_dir, "summary_report.txt")

        with open(report_path, "w") as f:
            f.write("IRL TRAINING SUMMARY\n")
            f.write("===================\n\n")
            f.write(f"Behavior: {self.behavior}\n")
            f.write(f"Track: {self.track_name}\n")
            f.write(f"Frames: {self.num_frames}\n\n")

            f.write("ITERATION SUMMARY\n")
            f.write("-----------------\n")
            f.write(
                f"{'Iteration':<10} {'T-Value':<15} {'Avg Reward':<15} {'Avg Loss':<15}\n"
            )

            sorted_data = sorted(self.iterations_data, key=lambda x: x["iteration"])
            for data in sorted_data:
                iteration = data["iteration"]
                t_value = data.get("t_value", "N/A")
                reward = data.get("avg_reward", "N/A")
                loss = data.get("avg_loss", "N/A")

                f.write(
                    f"{iteration:<10} {t_value:<15.6f} {reward:<15.6f} {loss:<15.6f}\n"
                )

            f.write("\n")

            if self.iterations_data:
                final_data = sorted_data[-1]
                f.write("FINAL WEIGHTS\n")
                f.write("------------\n")
                for i, w in enumerate(final_data.get("weights", [])):
                    f.write(f"w{i + 1}: {w:.6f}\n")

            f.write("\n")
            f.write(f"Plots saved to: {self.plots_dir}\n")

        print(f"Summary report generated: {report_path}")
