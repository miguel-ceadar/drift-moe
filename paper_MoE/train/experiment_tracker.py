import os, time, csv, psutil
import torch
from torch.utils.tensorboard import SummaryWriter

class ExperimentTracker:
    def __init__(self, run_id, mode, n_experts, top_k, seed, dataset, config):
        self.run_id    = run_id
        self.mode      = mode
        self.n_experts = n_experts
        self.top_k     = top_k or 0
        self.seed      = seed
        self.dataset   = dataset
        self.config    = config

        # ── Directories
        self.run_dir    = os.path.join(config["runs_root"], run_id)
        self.tb_dir     = os.path.join(self.run_dir, "tb")
        self.models_dir = os.path.join(self.run_dir, "models")
        os.makedirs(self.tb_dir,    exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # ── TensorBoard
        if config.get("use_tensorboard", False):
            self.writer = SummaryWriter(log_dir=self.tb_dir)
        else:
            self.writer = None

        # ── Intra-run CSV
        self.csv_path = os.path.join(self.run_dir, "results.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "step","loss","accuracy","kappa_m","kappa_temporal","elapsed_s"
            ])

        # ── Global summary CSV
        os.makedirs(os.path.dirname(config["global_csv"]), exist_ok=True)
        if not os.path.isfile(config["global_csv"]):
            with open(config["global_csv"], "w", newline="") as f:
                csv.writer(f).writerow([
                    "run_id","mode","n_experts","top_k",
                    "seed","dataset","accuracy",
                    "kappa_m","kappa_temporal",
                    "training_time_s","cpu%","ram%"
                ])

        self.start_time = time.time()

    def log_step(self, step, loss, accuracy, kappa_m, kappa_temp):
        elapsed = time.time() - self.start_time

        if self.writer:
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/accuracy", accuracy, step)
            self.writer.add_scalar("train/kappa_m", kappa_m, step)
            self.writer.add_scalar("train/kappa_temporal", kappa_temp, step)

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                step, f"{loss:.4f}", f"{accuracy:.4f}",
                f"{kappa_m:.4f}", f"{kappa_temp:.4f}", f"{elapsed:.2f}"
            ])

    def save_models(self, router, experts):
        if not self.config.get("save_models", True):
            return
        torch.save(router.state_dict(), os.path.join(self.models_dir, "router.pth"))
        for i, ex in enumerate(experts):
            torch.save(ex.state_dict(), os.path.join(self.models_dir, f"expert_{i}.pth"))

    def log_run_end(self, accuracy, kappa_m, kappa_temp):
        total_time = time.time() - self.start_time
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        with open(self.config["global_csv"], "a", newline="") as f:
            csv.writer(f).writerow([
                self.run_id, self.mode, self.n_experts, self.top_k,
                self.seed, self.dataset,
                f"{accuracy:.4f}", f"{kappa_m:.4f}", f"{kappa_temp:.4f}",
                f"{total_time:.2f}", f"{cpu:.1f}", f"{ram:.1f}"
            ])

    def log_exception(self, exc: Exception):
        with open(os.path.join(self.run_dir, "ERROR.log"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()
