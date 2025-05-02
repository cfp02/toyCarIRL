import argparse
import logging
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from cvxopt import (
    matrix,
    solvers,
)

from flat_game import carmunk
from nn import DQNAgent
from playing import (
    play,
)
from train import train as IRL_helper
from metrics import IRLTracker
from utils import TrajectoryRecorder, NUM_STATES, GAMMA

NUM_STATES = 16  # Updated for pushing task


class irlAgent:
    def __init__(
        self,
        randomFE,
        expertFE,
        epsilon,
        num_states,
        num_frames,
        behavior,
        track_file="tracks/default.json",
        run_id=None
    ):
        self.randomPolicy = randomFE
        self.expertPolicy = expertFE
        self.num_states = num_states
        self.num_frames = num_frames
        self.behavior = behavior
        self.track_file = track_file
        self.epsilon = epsilon  # termination when t<0.1
        self.randomT = np.linalg.norm(
            np.asarray(self.expertPolicy) - np.asarray(self.randomPolicy)
        )  # norm of the diff in expert and random
        self.policiesFE = {self.randomT: self.randomPolicy}

        self.best_model_path = None
        self.best_t_value = float("inf")
        
        # Add run_id for multiple training runs
        self.run_id = run_id or f"run_{int(time.time())}"
        
        # Initialize tracker with run_id
        self.tracker = IRLTracker(behavior, track_file, num_frames, run_id=self.run_id)

        print("Expert - Random at the Start (t) :: ", self.randomT)
        self.currentT = self.randomT
        self.minimumT = self.randomT

    def getRLAgentFE(
        self, W, i
    ):  # get the feature expectations of a new policy using RL agent
        # Create directories for saving models
        track_name = os.path.splitext(os.path.basename(self.track_file))[0]
        ITERATION = i
        save_dir = os.path.join(
            "saved-models", f"{self.behavior}_models", self.run_id, "evaluatedPolicies"
        )
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_dir = f"{save_dir}/checkpoints_{i}"

        # Create and train the agent
        agent = DQNAgent(
            state_size=self.num_states,
            action_size=3,
            hidden_sizes=[164, 150],
            is_pushing_task=True  # Enable pushing task mode
        )

        if self.best_model_path and os.path.exists(self.best_model_path):
            print(f"Loading previous best model: {self.best_model_path}")
            agent.load(self.best_model_path)
            # Initialize with a higher epsilon
            agent.epsilon = 0.3

        # Create game state and ensure environment is loaded
        game_state = carmunk.GameState(W, self.track_file)
        game_state.load_environment(self.track_file)
        game_state.reset()

        # Initialize trajectory recorder
        recorder = TrajectoryRecorder(W, self.behavior)

        # Train the agent with the tracker passed in for real-time updates
        rewards, losses = IRL_helper(
            agent=agent,
            env_weights=W,
            env_path=self.track_file,
            num_episodes=self.num_frames // 100,
            max_steps_per_episode=250,
            checkpoint_dir=checkpoint_dir,
            log_dir=f"{save_dir}/logs_{i}",
            tracker=self.tracker,  # Pass the tracker object
            iteration=i,  # Pass the current iteration number
            game_state=game_state,  # Pass the initialized game state
            recorder=recorder  # Pass the trajectory recorder
        )

        # Save the final model
        saved_model = os.path.join(
            save_dir, f"iter{ITERATION}_track-{track_name}_frame-{self.num_frames}.pt"
        )
        agent.save(saved_model)

        # Save intermediate model
        intermediate_model = os.path.join(
            save_dir, f"intermediate_iter{ITERATION}.pt"
        )
        agent.save(intermediate_model)

        # Get feature expectations from the recorder
        fe = recorder.feature_expectations

        # Calculate average metrics from training
        avg_reward = (
            np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        )
        avg_loss = np.mean(losses) if losses else 0

        return fe, avg_reward, avg_loss, saved_model

    def policyListUpdater(self, W, i):
        tempFE, avg_reward, avg_loss, saved_model = self.getRLAgentFE(
            W, i
        )  # get feature expectations of a new policy respective to the input weights
        hyperDistance = np.abs(
            np.dot(W, np.asarray(self.expertPolicy) - np.asarray(tempFE))
        )  # hyperdistance = t
        self.policiesFE[hyperDistance] = tempFE

        # Update best model path if this policy is better
        if hyperDistance < self.best_t_value:
            self.best_model_path = saved_model
            self.best_t_value = hyperDistance
            print(
                f"New best model: {self.best_model_path} with distance: {hyperDistance}"
            )

        # Track iteration data
        self.tracker.add_iteration_data(
            iteration=i,
            t_value=hyperDistance,
            weights=W,
            fe_distances=[float(k) for k in self.policiesFE.keys()],
            best_model_path=self.best_model_path,
            avg_reward=float(avg_reward),
            avg_loss=float(avg_loss),
        )

        # Generate plots after each iteration
        self.tracker.plot_metrics()

        return hyperDistance

    def optimalWeightFinder(self):
        # Create a directory for this run's weights
        weights_dir = os.path.join("weights", self.behavior, self.run_id)
        os.makedirs(weights_dir, exist_ok=True)
        
        weights_file = os.path.join(weights_dir, "weights.txt")
        f = open(weights_file, "w")
        
        i = 1
        while True:
            W = self.optimization()
            print("weights ::", W)
            f.write(str(W))
            f.write("\n")
            print("the distances  ::", self.policiesFE.keys())
            self.currentT = self.policyListUpdater(W, i)
            print("Current distance (t) is:: ", self.currentT)
            if (
                self.currentT <= self.epsilon
            ):  # terminate if the point reached close enough
                break
            i += 1
        f.close()

        # Generate final summary report
        self.tracker.generate_summary_report()

        return W

    def optimization(self):
        m = len(self.expertPolicy)
        
        # Normalize feature expectations
        expert_norm = np.linalg.norm(self.expertPolicy)
        if expert_norm > 0:
            normalized_expert = self.expertPolicy / expert_norm
        else:
            normalized_expert = self.expertPolicy
            
        normalized_policies = {}
        for k, v in self.policiesFE.items():
            policy_norm = np.linalg.norm(v)
            if policy_norm > 0:
                normalized_policies[k] = v / policy_norm
            else:
                normalized_policies[k] = v

        # Add regularization to prevent numerical instability
        P = matrix(2.0 * np.eye(m) + 0.1 * np.ones((m, m)), tc="d")
        q = matrix(np.zeros(m), tc="d")
        
        policyList = [normalized_expert]
        h_list = [1]
        
        for i in normalized_policies.keys():
            policyList.append(normalized_policies[i])
            h_list.append(1)
            
        policyMat = np.matrix(policyList)
        policyMat[0] = -1 * policyMat[0]
        
        # Add small regularization to constraints
        G = matrix(policyMat, tc="d")
        h = matrix(-np.array(h_list) - 0.01, tc="d")
        
        try:
            sol = solvers.qp(P, q, G, h)
            if sol['status'] == 'optimal':
                weights = np.squeeze(np.asarray(sol["x"]))
                norm = np.linalg.norm(weights)
                if norm > 0:
                    weights = weights / norm
                return weights
            else:
                print(f"Warning: Optimization did not converge. Status: {sol['status']}")
                return np.ones(m) / m  # Return uniform weights as fallback
        except Exception as e:
            print(f"Warning: Optimization failed with error: {e}")
            return np.ones(m) / m  # Return uniform weights as fallback


if __name__ == "__main__":
    import os

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run IRL on toy car environment")
    parser.add_argument(
        "--track",
        "-t",
        type=str,
        default="tracks/pushing.json",  # Updated default track
        help="Path to track file (default: tracks/pushing.json)",
    )
    parser.add_argument(
        "--behavior",
        "-b",
        type=str,
        default="pushing",  # Updated default behavior
        help="Behavior name (default: pushing)",
    )
    parser.add_argument(
        "--frames",
        "-f",
        type=int,
        default=100000,
        help="Number of frames for training (default: 100000)",
    )
    parser.add_argument(
        "--continue",
        dest="continue_training",
        action="store_true",
        help="Continue from previous training run",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=1,
        help="Number of training runs to perform (default: 1)",
    )
    args = parser.parse_args()

    # Update global variables from arguments
    BEHAVIOR = args.behavior
    FRAMES = args.frames
    track_file = args.track
    num_runs = args.runs

    if not os.path.exists(track_file):
        print(
            f"Warning: Track file {track_file} not found, checking alternative paths..."
        )
        alt_path = os.path.join("tracks", track_file)
        if os.path.exists(alt_path):
            track_file = alt_path
            print(f"Using track file: {track_file}")
        else:
            alt_path = os.path.join("tracks", f"{track_file}.json")
            if os.path.exists(alt_path):
                track_file = alt_path
                print(f"Using track file: {track_file}")
            else:
                print("Could not find track file, using default: tracks/pushing.json")
                track_file = "tracks/pushing.json"

    print(f"Using track file: {track_file}")
    print(f"Behavior: {BEHAVIOR}")
    print(f"Training frames: {FRAMES}")
    print(f"Number of runs: {num_runs}")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Initialize random policy feature expectations for 16 dimensions
    randomPolicyFE = [0.1] * NUM_STATES

    # Load demonstrations and calculate expert policy feature expectations
    expertPolicy = [0.0] * NUM_STATES
    demo_dir = os.path.join("demonstrations", BEHAVIOR)
    if os.path.exists(demo_dir):
        print(f"Loading demonstrations from {demo_dir}")
        demo_files = [f for f in os.listdir(demo_dir) if f.endswith('.json')]
        if demo_files:
            print(f"Found {len(demo_files)} demonstration files")
            for demo_file in demo_files:
                with open(os.path.join(demo_dir, demo_file), 'r') as f:
                    demo_data = json.load(f)
                    if 'feature_expectations' in demo_data:
                        expertPolicy = [x + y for x, y in zip(expertPolicy, demo_data['feature_expectations'])]
            # Average the feature expectations
            expertPolicy = [x / len(demo_files) for x in expertPolicy]
            print("Expert policy feature expectations calculated from demonstrations")
        else:
            print("No demonstration files found, using zero initialization")
    else:
        print(f"Demonstration directory {demo_dir} not found, using zero initialization")

    epsilon = 0.1

    # Run multiple training runs
    for run in range(num_runs):
        print(f"\nStarting training run {run + 1}/{num_runs}")
        run_id = f"run_{run + 1}_{int(time.time())}"
        
        best_model_path = None
        if args.continue_training:
            # Try to load convergence data
            try:
                with open(f"convergence-{BEHAVIOR}.json", "r") as f:
                    convergence_data = json.load(f)
                    if convergence_data:
                        best_model = convergence_data[-1].get("best_model")
                        if best_model and os.path.exists(best_model):
                            best_model_path = best_model
                            print(f"Continuing training from model: {best_model_path}")
            except Exception as e:
                print(f"Error loading previous convergence data: {e}")
                print("Starting fresh training run.")

        irlearner = irlAgent(
            randomPolicyFE,
            expertPolicy,
            epsilon,
            NUM_STATES,
            FRAMES,
            BEHAVIOR,
            track_file,
            run_id=run_id
        )

        if best_model_path:
            irlearner.best_model_path = best_model_path

        final_weights = irlearner.optimalWeightFinder()
        print(f"Run {run + 1} complete. Final optimized weights:", final_weights)
