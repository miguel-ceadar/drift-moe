"""
Config file, gets command line arguments and parse them for usage
"""

import argparse
import random
import pandas as pd

def get_config():
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

    return parser.parse_args()



# ─────────────────────────────────────────────────────────
# EXPERIMENT_SETS: edit here to choose one or many runs.
# If empty, we just run the single config from CLI.
# Example single-run:
# EXPERIMENT_SETS = [
#   {"mode":"data", "dataset":"elec", "seeds":[42],      "top_k":3},
# ]
# Example multi-run:
# EXPERIMENT_SETS = [
#   {"mode":"joint_data","dataset":"elec", "seeds":[1,2,3], "top_k":5},
#   {"mode":"task",      "dataset":"covt","seeds":[42],    "top_k":None},
# ]

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
# Modes in the order you want
# ───────────────────────────────────────────────────────────────
modes = ["joint_data", "joint_task", "data", "task"]

# ───────────────────────────────────────────────────────────────
# Now automatically build every (mode,dataset) combo,
# each with 10 random seeds and top_k=3
# ───────────────────────────────────────────────────────────────
TRACKING = {
    "use_tensorboard": True,
    "runs_root":       "/home/miguel/drift_moe/runs",
    "global_csv":      "/home/miguel/drift_moe/results/global_results.csv",
    "save_models":     True,
}
"""
EXPERIMENT_SETS = [
    {
        # core fields
        "mode":    mode,
        "dataset": dataset,
        "seeds":   [random.randint(1, 1000000) for _ in range(10)],
        "top_k":   3,
        # merge in dataset‐specific fields
        **dataset_configs[dataset],
    }
    for mode in modes
    for dataset in dataset_configs
]
print(len(EXPERIMENT_SETS))
print(EXPERIMENT_SETS[0])
"""
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
print("=== EXPERIMENT_SETS ===")
for exp in EXPERIMENT_SETS:
    print(f"Mode: {exp['mode']:<12}  Dataset: {exp['dataset']:<6}  Seeds: {exp['seeds']}")
print("=======================")
