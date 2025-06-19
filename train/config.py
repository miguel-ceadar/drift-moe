"""
Config file, gets command line arguments and parse them for usage
"""

import argparse
import random
import pandas as pd

def get_config():
    """
    Gets arguments when running single experiment
    """
    parser = argparse.ArgumentParser(
        description="Master configuration for all MoE experiments (joint, co-learn, two-stage)."
    )

    # Mode selection: joint_data, joint_task, data, or task
    parser.add_argument(
        "--mode",
        choices=["joint_data", "joint_task", "data", "task"],
        default="joint_data",
        help="Which variant of MoE to run."
    )
    parser.add_argument(
            "--cv_folds",
            type=int,
            default=1,
            help="k for k-fold distributed CV-prequential (1 = classic prequential)"
            )

    # General model/stream parameters
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name, possible choices: [led_a, led_g, sea_a, sea_g, rbf_m, rbf_f, elec, covt, airl]"
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=1_000_000,
        help="Number of total samples to draw from the stream."
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=24,
        help="Dimensionality of each instance (number of features)."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of target classes."
    )
    parser.add_argument(
        "--n_experts",
        type=int,
        default=12,
        help="Number of experts in the mixture for data experts (task experts always same number as classes)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-K experts to update in joint/co-learn modes."
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of the MLP router."
    )
    parser.add_argument(
        "--lr_router",
        type=float,
        default=2e-3,
        help="Learning rate for the router optimizer."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (where applicable, e.g. two-stage)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=75,
        help="Number of epochs (two-stage router training only)."
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=5_000,
        help="Interval (in samples) at which to print streaming progress."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Single random seed (if not using multiple seeds)."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="List of random seeds to run (loops over)."
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="If set, experiments over multiple seeds will run in parallel."
    )

    # Optional data clustering params (only used when mode=data)
    parser.add_argument(
        "--burn_in_frac",
        type=float,
        default=0.10,
        help="Fraction of stream used for clustering burn-in (data experts only)."
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=0.05,
        help="Fraction of burn-in block to sample for PCA/clustering (data experts only)."
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=15,
        help="Maximum number of clusters to try (data experts only)."
    )
    parser.add_argument(
        "--cluster_type",
        choices=["kmeans", "gmm"],
        default="kmeans",
        help="Clustering algorithm to use during data‐expert burn‐in (data mode)."
    )
    parser.add_argument(
            "--label_delay",
            type=int,
            default=0,
            help="label delay")

    return parser.parse_args()

# ───────────────────────────────────────────────────────────────
# Dataset‐specific configs
# ───────────────────────────────────────────────────────────────
dataset_configs = {
    "led_g": {"input_dim": 24,  "num_classes": 10},
    "led_a": {"input_dim": 24,  "num_classes": 10},
    "sea_a": {"input_dim": 3,  "num_classes": 2},
    "sea_g": {"input_dim": 3,  "num_classes": 2},
    "rbf_m": {"input_dim": 10,  "num_classes": 5},
    "rbf_f": {"input_dim": 10,  "num_classes": 5},
    "covt":  {"input_dim": 54, "num_classes": 7},
    "elec":  {"input_dim": 8,  "num_classes": 2},
    "airl": {"input_dim": 7,  "num_classes": 2},
}


# ───────────────────────────────────────────────────────────────
# num experts and top k ablation configuration
# ───────────────────────────────────────────────────────────────

"""
num_experts_options = [6, 8, 10, 12, 15, 20]
top_k_options       = [1, 2, 3, 4, 5, 6]

combo_grid = [
    (n_exp, k)
    for n_exp in num_experts_options
    for k      in top_k_options
    if k <= n_exp // 2
]

seeds_for_each = [42]

EXPERIMENT_SETS = [
    {
        "mode":     "joint_data",
        "dataset":  "led_g",
        "n_experts": n_exp,
        "top_k":    k,
        "seeds":    seeds_for_each,
        **dataset_configs["led_g"],   # adds input_dim, num_classes
    }
    for n_exp, k in combo_grid
]

TRACKING = {
    "use_tensorboard": True,
    # runs_root stays the same; sub-folders are added per run
    "runs_root": "/home/miguel/drift_moe/ablation_runs",
    "global_csv": "/home/miguel/drift_moe/results/ablation_results.csv",
    "save_models": False,   # stop saving 30× checkpoints if disk is tight
}


"""



# ────────────────────────────────────────────────
# joint_data & joint_task on led_g & led_a with label delay
# ────────────────────────────────────────────────
"""
random.seed(51)   # for reproducibility

modes    = ["joint_data", "joint_task"]
datasets = ["led_g", "led_a"]

EXPERIMENT_SETS = [
    {
        "mode":        mode,
        "dataset":     ds,
        "seeds":       random.sample(range(1, 1_000_001), 10),
        "label_delay": 1000,       # ← your new delayed‐labels param
        **dataset_configs[ds],     # pulls in input_dim & num_classes
    }
    for mode in modes
    for ds   in datasets
]

TRACKING = {
    "use_tensorboard": True,
    "runs_root":       "/home/miguel/drift_moe/delayed_label_runs",
    "global_csv":      "/home/miguel/drift_moe/results/delayed_label_results.csv",
    "save_models":     True,
    }
"""


# ───────────────────────────────────────────────────────────────
# K-fold cross validation
# ───────────────────────────────────────────────────────────────
"""
CV_FOLDS   = 10
SEED       = 42          # single seed for every run
modes      = ["joint_task"]
datasets   = [
    "led_a", "led_g",
    "sea_a", "sea_g",
    "rbf_m", "rbf_f",
    "elec",  "covt", "airl",
]


EXPERIMENT_SETS = [
    {
        "mode":        mode,
        "dataset":     ds,
        "seeds":       [SEED],                 # one-seed list
        "cv_folds":    CV_FOLDS,               # turn on 10-fold CV
        **dataset_configs[ds],                 # merges dataset-specific info
    }
    for mode in modes
    for ds   in datasets
]
EXPERIMENT_SETS[0:0] = [
        {
            "mode":        "joint_data",
            "dataset":     "airl",
            "seeds":       [SEED],                 # one-seed list
            "cv_folds":    CV_FOLDS,               # turn on 10-fold CV
            **dataset_configs["airl"]
            },
        {
            "mode":        "joint_data",
            "dataset":     "elec",
            "seeds":       [SEED],                 # one-seed list
            "cv_folds":    CV_FOLDS,               # turn on 10-fold CV
            **dataset_configs["elec"]
            },
        {
            "mode":        "joint_data",
            "dataset":     "covt",
            "seeds":       [SEED],                 # one-seed list
            "cv_folds":    CV_FOLDS,               # turn on 10-fold CV
            **dataset_configs["covt"]
            }
        ]
TRACKING = {
    "use_tensorboard": True,
    "runs_root":       "/home/miguel/drift_moe/k_fold_runs",
    "global_csv":      "/home/miguel/drift_moe/results/k_fold_results.csv",
    "save_models":     False,
    }
    """
# ───────────────────────────────────────────────────────────────
# All different Modes
# ───────────────────────────────────────────────────────────────
modes = ["joint_data", "joint_task", "data", "task"]

# ───────────────────────────────────────────────────────────────
# Full training routine config
# each with 10 random seeds and, contains functionality of restarting from point where it stopped by checking global csv
# ───────────────────────────────────────────────────────────────
"""
TRACKING = {
    "use_tensorboard": True,
    "runs_root":       "/home/miguel/drift_moe/runs",
    "global_csv":      "/home/miguel/drift_moe/results/global_results.csv",
    "save_models":     True,
    }

Loop for full experiment list
try:
    done_df = pd.read_csv(TRACKING["global_csv"])
except FileNotFoundError:
    # no runs yet
    done_df = pd.DataFrame(columns=["run_id","pipeline","dataset","seed"])

random.seed(42)   # for reproducibility

EXPERIMENT_SETS = []
for mode in modes:
    for ds, ds_cfg in dataset_configs.items():
        # get seeds already run for this (mode, ds)
        done_seeds = set(
            done_df.loc[
                (done_df["mode"] == mode) &
                (done_df["dataset"]  == ds),
                "seed"
            ].astype(int)
        )
        desired = 10
        remaining = desired - len(done_seeds)
        if remaining <= 0:
            print(f"[SKIP] {mode},{ds} already has {len(done_seeds)} seeds")
            continue
        # build a pool of candidate seeds
        candidate_pool = set(range(1, 100_001))
        available = list(candidate_pool - done_seeds)

        print(f"[DEBUG] Completed seeds for ({mode},{ds}): {sorted(done_seeds)}")
        # sample up to 10 new seeds
        new_seeds = random.sample(available, remaining)


        EXPERIMENT_SETS.append({
            "mode":    mode,
            "dataset": ds,
            "seeds":   new_seeds,
            "top_k":   3,
            **ds_cfg
        })
"""
EXPERIMENT_SETS = [
    {
        "mode":        "joint_data",
        "dataset":     "led_g",
        "seeds":       random.sample(range(1, 1_000_001), 10),
        **dataset_configs["led_g"],     # pulls in input_dim & num_classes
    }
]
TRACKING = {
    "use_tensorboard": True,
    "runs_root":       "/home/miguel/drift_moe/test_runs",
    "global_csv":      "/home/miguel/drift_moe/results/test_results.csv",
    "save_models":     False,
    }

print("=== EXPERIMENT_SETS ===")
for exp in EXPERIMENT_SETS:
    print(f"Mode: {exp['mode']:<12}  Dataset: {exp['dataset']:<6}  Seeds: {exp['seeds']}")
print("=======================")

