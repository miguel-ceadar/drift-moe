"""
Config file, gets command line arguments and parse them for usage
"""

import argparse

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
        required=True,
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
        default=10_000,
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
