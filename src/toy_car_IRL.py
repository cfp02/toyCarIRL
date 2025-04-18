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
BEHAVIOR = "custom"  # yellow/brown/red/bumping
FRAMES = 100000  # number of RL training frames per iteration of IRL


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

        # Train the agent
        IRL_helper(
            agent=agent,
            env_weights=W,
            env_path=self.track_file,
            num_episodes=self.num_frames // 100,
            max_steps_per_episode=1000,
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
        P = matrix(2.0 * np.eye(m), tc="d")  # min ||w||
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
        8.15450743e-01,
        3.81642258e-01,
        7.21806767e-01,
        1.42339084e-02,
        9.81664173e-01,
        2.56301855e-21,
        2.83499139e-03,
        1.26692678e-03,
        2.00000000e-01,
        4.65498113e-55,
    ]

    epsilon = 0.1
    irlearner = irlAgent(
        randomPolicyFE,
        expertPolicy,
        epsilon,
        NUM_STATES,
        FRAMES,
        BEHAVIOR,
        track_file,
    )

    print(irlearner.optimalWeightFinder())
