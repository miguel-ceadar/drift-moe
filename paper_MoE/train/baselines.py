# arf_baseline.py
"""
Adaptive Random Forest (ARF) baseline runner for the drift‑MoE code‑base.

• Uses **CapyMOA**'s `AdaptiveRandomForestClassifier` on the *exact* same MOA streams
  produced by `data_loader.DataLoader`.
• Evaluates *prequentially* (test‑then‑train) just like the `train_joint_*` routines
  in `moe_model.py`.
• Integrates seamlessly with the existing **EXPERIMENT_SETS / ExperimentTracker**
  infrastructure so that ARF baselines can be launched in bulk or via the CLI.

Run a *single* ARF baseline:
    python arf_baseline.py --dataset led_g --total_samples 1000000 --print_every 10000

Run the batch experiments defined in `config.EXPERIMENT_SETS` (recommended):
    python arf_baseline.py
"""
from __future__ import annotations

import argparse
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from river import metrics
from capymoa.classifier import (
    AdaptiveRandomForestClassifier,   # ARF
    StreamingRandomPatches,          # SRP
    OnlineSmoothBoost,               # OnlineSmoothBoost
    OzaBoost,                        # OzaBoost
    LeveragingBagging,               # LevBag
    OnlineBagging,                   # OzaBag
)
from capymoa.evaluation import ClassificationEvaluator

from data_loader import DataLoader
from experiment_tracker import ExperimentTracker
from config import dataset_configs, get_config as _base_get_config

# ──────────────────────────────────────────────────────────────
# 1. CLI parser — extend the project‑wide parser with ARF flags
# ──────────────────────────────────────────────────────────────

def get_config() -> argparse.Namespace:
    """Reuse the main project parser but add a handful of ARF‑specific flags."""
    cfg = _base_get_config()

    # Parse *only* the ARF‑specific flags that weren't consumed yet
    arf_parser = argparse.ArgumentParser(add_help=False)
    arf_parser.add_argument("--n_models", type=int, default=100,
                            help="Number of trees in the Adaptive Random Forest.")
    remaining, _ = arf_parser.parse_known_args()

    for k, v in vars(remaining).items():
        setattr(cfg, k, v)

    # Force‑set a mode identifier so downstream scripts can branch if needed
    return cfg

# ──────────────────────────────────────────────────────────────
# 2. Baseline model
# ──────────────────────────────────────────────────────────────
class CapyModel:
    """Simple wrapper around CapyMOA's baselines"""

    _CLF_MAP = {
        "arf":        AdaptiveRandomForestClassifier,
        "srp":        StreamingRandomPatches,
        "smoothboost": OnlineSmoothBoost,
        "ozaboost":   OzaBoost,
        "levbag":     LeveragingBagging,
        "ozabag":     OnlineBagging,        # “OzaBag” in MOA literature
    }

    def __init__(self, config: argparse.Namespace, seed: int):
        self.cfg = config
        self.seed = seed

        # ── Reproducibility ─────────────────────────────
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # ── Data stream ────────────────────────────────
        self.stream = DataLoader(config.dataset, seed=self.seed)
        self.stream.restart()
        self.schema = self.stream.get_schema()

        # ── Classifier ─────────────────────────────────
        Clf = self._CLF_MAP[config.mode]
        self.clf = Clf(
                schema=self.schema,
                random_seed=seed)

        self.evaluator = ClassificationEvaluator(schema=self.schema)

    # ──────────────────────────────────────────────────
    # Core prequential loop
    # ──────────────────────────────────────────────────
    def train_prequential(self, tracker: ExperimentTracker | None = None):
        total = self.cfg.total_samples
        print_every = self.cfg.print_every

        start = time.time()
        for t in range(1, total + 1):
            if not self.stream.has_more_instances():
                break

            inst = self.stream.next_instance()
            y_true = inst.y_index

            # 1) ── TEST
            y_pred = self.clf.predict(inst)
            # CapyMOA returns None until the first label is seen
            if y_pred is None:
                y_pred = random.randrange(self.cfg.num_classes)

            self.evaluator.update(y_true, y_pred)

            # 2) ── TRAIN
            self.clf.train(inst)

            # 3) ── LOGGING
            if t % print_every == 0:
                elapsed = time.time() - start
                a, km, kt = self.evaluator.accuracy(), self.evaluator.kappa_m(), self.evaluator.kappa_t()
                print(f"[{t:,} / {total:,}] Acc={a:.4f}  kappa_m={km:.4f}  kappa_t={kt:.4f}  ({elapsed:.1f}s)")

                if tracker:
                    tracker.log_step(
                        step=t,
                        loss=0.0,               # ARF has no explicit loss
                        accuracy=a,
                        kappa_m=km,
                        kappa_temp=kt,
                    )

        return self.evaluator.accuracy(), self.evaluator.kappa_m(), self.evaluator.kappa_t()

# ──────────────────────────────────────────────────────────────
# 3. Batch‑experiment harness (mirrors run_experiments.py)
# ──────────────────────────────────────────────────────────────

TRACKING = {
    "use_tensorboard": True,
    "runs_root":       "/home/miguel/drift_moe/arf_baseline_runs",
    "global_csv":      "/home/miguel/drift_moe/results/arf_results.csv",
    "save_models":     False,
    } 

modes    = ["srp", "smoothboost", "ozaboost", "levbag", "ozabag"]
datasets = ["led_a","led_g","sea_a","sea_g","rbf_m","rbf_f","elec","covt","airl"]


EXPERIMENT_SETS = [
    {
        "mode":    m,
        "dataset": ds,
        "seeds":   random.sample(range(1, 1_000_001), 10),   # always ten seeds
        **dataset_configs[ds],
    }
    for m  in modes
    for ds in datasets
]

def _run_single(cfg: argparse.Namespace, seed: int):
    cfg.seed = seed
    
    run_id = f"{cfg.mode}_{cfg.dataset}_s{seed}_{int(time.time())}"
    tracker = ExperimentTracker(
        run_id=run_id,
        mode=cfg.mode,
        n_experts=0,
        top_k=0,
        seed=seed,
        dataset=cfg.dataset,
        config=TRACKING,
    )

    try:
        model = CapyModel(cfg, seed)
        acc, km, kt = model.train_prequential(tracker=tracker)
        tracker.log_run_end(acc, km, kt)
    except Exception as exc:
        tracker.log_exception(exc)
        raise
    finally:
        tracker.close()

# Entry point — handles EXPERIMENT_SETS just like run_experiments.py

def main():
    cfg = get_config()

    experiments = EXPERIMENT_SETS or [{"dataset": cfg.dataset, "seeds": cfg.seeds, "n_models": cfg.n_models}]
    
    print("=== EXPERIMENT_SETS ===")
    for exp in EXPERIMENT_SETS:
        print(f"Mode: {exp['mode']:<12}  Dataset: {exp['dataset']:<6}  Seeds: {exp['seeds']}")
    print("=======================")
    for exp in experiments:
        cfg.mode = exp["mode"]
        cfg.dataset = exp["dataset"]
        cfg.seeds = exp["seeds"]
        cfg.n_models = exp.get("n_models", cfg.n_models)

        for s in cfg.seeds:
            _run_single(cfg, s)


if __name__ == "__main__":
    main()

