import argparse
import logging

import numpy as np
from cvxopt import (
    matrix,
    solvers,
)

from nn import DQNAgent
from playing import (
    play,
)
from train import train as IRL_helper

NUM_STATES = 10


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
            "saved-models", f"{self.behavior}_models", "evaluatedPolicies"
        )
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_dir = f"{save_dir}/checkpoints_{i}"

        # Create and train the agent
        agent = DQNAgent(
            state_size=self.num_states,
            action_size=3,
            hidden_sizes=[164, 150],
        )

        if self.best_model_path and os.path.exists(self.best_model_path):
            print(f"Loading previous best model: {self.best_model_path}")
            agent.load(self.best_model_path)
            # Initialize with a higher epsilon
            agent.epsilon = 0.3

        # Train the agent
        IRL_helper(
            agent=agent,
            env_weights=W,
            env_path=self.track_file,
            num_episodes=self.num_frames // 100,
            max_steps_per_episode=250,
            checkpoint_dir=checkpoint_dir,
            log_dir=f"{save_dir}/logs_{i}",
        )

        # Save the final model
        saved_model = os.path.join(
            save_dir, f"iter{ITERATION}_track-{track_name}_frame-{self.num_frames}.pt"
        )
        agent.save(saved_model)

        # Use the trained agent to get feature expectations with the same track
        return play(agent, W, self.track_file)

    def policyListUpdater(self, W, i):
        tempFE = self.getRLAgentFE(
            W, i
        )  # get feature expectations of a new policy respective to the input weights
        hyperDistance = np.abs(
            np.dot(W, np.asarray(self.expertPolicy) - np.asarray(tempFE))
        )  # hyperdistance = t
        self.policiesFE[hyperDistance] = tempFE

        # Update best model path if this policy is better
        if hyperDistance < self.best_t_value:
            track_name = os.path.splitext(os.path.basename(self.track_file))[0]
            save_dir = os.path.join(
                "saved-models", f"{self.behavior}_models", "evaluatedPolicies"
            )
            self.best_model_path = os.path.join(
                save_dir, f"iter{i}_track-{track_name}_frame-{self.num_frames}.pt"
            )
            self.best_t_value = hyperDistance
            print(
                f"New best model: {self.best_model_path} with distance: {hyperDistance}"
            )

        return hyperDistance

    def optimalWeightFinder(self):
        f = open("weights-" + BEHAVIOR + ".txt", "w")
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
        return W

    def optimization(
        self,
    ):
        m = len(self.expertPolicy)
        P = matrix(2.0 * np.eye(m), tc="d")
        q = matrix(np.zeros(m), tc="d")
        policyList = [self.expertPolicy]
        h_list = [1]
        for i in self.policiesFE.keys():
            policyList.append(self.policiesFE[i])
            h_list.append(1)
        policyMat = np.matrix(policyList)
        policyMat[0] = -1 * policyMat[0]
        G = matrix(policyMat, tc="d")
        h = matrix(-np.array(h_list), tc="d")
        sol = solvers.qp(P, q, G, h)

        weights = np.squeeze(np.asarray(sol["x"]))
        norm = np.linalg.norm(weights)
        weights = weights / norm
        return weights


if __name__ == "__main__":
    import os

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run IRL on toy car environment")
    parser.add_argument(
        "--track",
        "-t",
        type=str,
        default="tracks/default.json",
        help="Path to track file (default: tracks/default.json)",
    )
    parser.add_argument(
        "--behavior",
        "-b",
        type=str,
        default="custom",
        help="Behavior name (default: custom)",
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
    args = parser.parse_args()

    # Update global variables from arguments
    BEHAVIOR = args.behavior
    FRAMES = args.frames
    track_file = args.track

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
                print("Could not find track file, using default: tracks/default.json")
                track_file = "tracks/default.json"

    print(f"Using track file: {track_file}")
    print(f"Behavior: {BEHAVIOR}")
    print(f"Training frames: {FRAMES}")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    randomPolicyFE = [
        3.90000000e-01,
        1.50000000e-01,
        3.80000000e-01,
        2.00000000e-01,
        2.50000000e-01,
        3.00000000e-01,
        3.50000000e-01,
        4.00000000e-01,
        2.00000000e-01,
        1.00000000e-01,
    ]

    # Change this based on demonstration FE
    expertPolicy = [
        1.72849219e-01,
        7.87486915e-01,
        8.62962472e-01,
        2.39732714e-01,
        7.60214094e-01,
        9.68773805e-08,
        2.64416711e-05,
        2.66536551e-05,
        1.00000000e00,
        8.52654902e-01,
    ]

    epsilon = 0.1

    best_model_path = None
    if args.continue_training:
        # Try to load convergence data
        try:
            import json

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
    )

    if best_model_path:
        irlearner.best_model_path = best_model_path

    final_weights = irlearner.optimalWeightFinder()
    print(irlearner.optimalWeightFinder())
