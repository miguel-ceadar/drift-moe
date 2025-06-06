# run_experiments.py

import multiprocessing as mp
from config import get_config
from moe_model import MoEModel

def main():
    config = get_config()

    def run_seed(seed):
        print(f"\n===== Running seed={seed}, mode={config.mode} =====\n")
        model = MoEModel(config, seed)
        if config.mode == "joint_data":
            model.train_joint_data()
        elif config.mode == "joint_task":
            model.train_joint_task()
        elif config.mode == "data":
            model.train_data()
        elif config.mode == "task":
            model.train_task()
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

    # If multiple seeds and parallel flag set, run in parallel
    if len(config.seeds) > 1 and config.parallel:
        with mp.Pool(len(config.seeds)) as pool:
            pool.map(run_seed, config.seeds)
    else:
        for sd in config.seeds:
            run_seed(sd)


if __name__ == "__main__":
    main()
