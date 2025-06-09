import multiprocessing as mp
import time
import os
import copy

from config import get_config, EXPERIMENT_SETS, TRACKING
from moe_model import MoEModel
from experiment_tracker import ExperimentTracker
import torch

def _init_worker(n_threads):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)

def _run_single(config, seed):
    config.seed = seed
    run_id = f"{config.mode}_{config.dataset}_s{seed}_{int(time.time())}"
    # build a temporary empty tracker; we'll fill n_experts after init
    tracker = ExperimentTracker(
        run_id=run_id,
        mode=config.mode,
        n_experts=config.n_experts,
        top_k=config.top_k,
        seed=seed,
        dataset=config.dataset,
        config=TRACKING
    )

    try:
        model = MoEModel(config, seed)
        tracker.n_experts = model.n_experts

        # dispatch to the right train_* method, passing tracker
        if config.mode == "joint_data":
            acc, km, kt = model.train_joint_data(tracker=tracker)
        elif config.mode == "joint_task":
            acc, km, kt = model.train_joint_task(tracker=tracker)
        elif config.mode == "data":
            acc, km, kt = model.train_data(tracker=tracker)
        elif config.mode == "task":
            acc, km, kt = model.train_task(tracker=tracker)
        else:
            raise ValueError(config.mode)

        # finalize
        tracker.save_models(model.router, model.experts)
        tracker.log_run_end(acc, km, kt)

    except Exception as e:
        tracker.log_exception(e)
        raise

    finally:
        tracker.close()

def main():
    base_cfg = get_config()

    experiments = EXPERIMENT_SETS
    for exp in experiments:
        # start from CLI base, then override with exp dict
        cfg = copy.deepcopy(base_cfg)
        for key, val in exp.items():
            setattr(cfg, key, val)

        # ensure seeds is a list
        seeds = cfg.seeds if isinstance(cfg.seeds, list) else [cfg.seeds]

        # parallel or sequential
        if len(seeds) > 1 and getattr(cfg, "parallel", False):
            with mp.Pool(len(seeds)) as pool:
                pool.starmap(_run_single, [(cfg, s) for s in seeds])
        else:
            for s in seeds:
                _run_single(cfg, s)


if __name__ == "__main__":
    main()
