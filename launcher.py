import subprocess
import time

SEEDS = [0, 42, 123]
ALGORITHMS = ["nStepDDQN", "DQN", "DDQN"]

# Choose which environment to train on:
# ENV_NAME = "PongNoFrameskip-v4"
ENV_NAME = "MountainCar-v0"


# N-step value to use when algo == "nStepDDQN"
NSTEP_VALUES = [3, 5, 6]

# Python executable (change to "python3" if needed)
PYTHON = "python"

# Path to training script
TRAIN_SCRIPT = "train_agents.py"


def launch_one(seed: int, algo: str, env_name: str, nstep: int | None):
    """Launch a single (seed, algo, env, nstep) training run as a subprocess."""
    cmd = [
        PYTHON,
        TRAIN_SCRIPT,
        "--seed",
        str(seed),
        "--algo",
        algo,
        "--env",
        env_name,
    ]

    if algo == "nStepDDQN" and nstep is not None:
        cmd += ["--nstep", str(nstep)]

    print("Launching:", " ".join(cmd))
    # Start the process
    return subprocess.Popen(cmd)


def main():
    for seed in SEEDS:
        print(f"\n==============================")
        print(f"  STARTING SEED {seed} FOR {ENV_NAME}")
        print(f"==============================\n")

        processes = []

        # Launch all three algorithms in parallel for this seed
        for algo in ALGORITHMS:

            if algo == "nStepDDQN":
                for n in NSTEP_VALUES:
                    p = launch_one(seed, algo, ENV_NAME, n)
                    processes.append(p)
            else:
                p = launch_one(seed, algo, ENV_NAME, None)
                processes.append(p)

        # Wait for all three to finish
        for p in processes:
            p.wait()

        print(f"\n==============================")
        print(f"  FINISHED SEED {seed} FOR {ENV_NAME}")
        print(f"==============================\n")
        # Optional small pause between seeds
        time.sleep(2)


if __name__ == "__main__":
    main()
