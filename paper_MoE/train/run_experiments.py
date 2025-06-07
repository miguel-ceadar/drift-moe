import multiprocessing as mp
import time

from config import get_config, EXPERIMENT_SETS, TRACKING
from moe_model import MoEModel
from experiment_tracker import ExperimentTracker

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
    cfg = get_config()

    # If EXPERIMENT_SETS is non-empty, override CLI; else run the single CLI run
    experiments = EXPERIMENT_SETS or [
        {
            "mode":    cfg.mode,
            "dataset": cfg.dataset,
            "seeds":   cfg.seeds,
            "top_k":   cfg.top_k
        }
    ]

    for exp in experiments:
        cfg.mode    = exp["mode"]
        cfg.dataset = exp["dataset"]
        cfg.seeds   = exp["seeds"]
        cfg.top_k   = exp.get("top_k", cfg.top_k)

        if len(cfg.seeds) > 1 and cfg.parallel:
            with mp.Pool(len(cfg.seeds)) as pool:
                pool.starmap(_run_single, [(cfg, s) for s in cfg.seeds])
        else:
            for s in cfg.seeds:
                _run_single(cfg, s)

if __name__ == "__main__":
    main()
