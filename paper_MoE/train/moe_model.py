# moe_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys

from river import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from data_loader import DataLoader
from experts import Expert
from capymoa.instance import LabeledInstance
from capymoa.evaluation import ClassificationEvaluator

class RouterMLP(nn.Module):
    """
    A simple 3-layer MLP router. Input dim → hidden_dim → hidden_dim/2 → output_dim (n_experts).
    Uses Xavier init for Linear layers. Dropout optional.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, drop_prob: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _safe_metric(metric_obj, method_name, *args, **kwargs):
    """
    Calls metric_obj.method_name(*args, **kwargs) but returns 0.0
    on any exception (e.g. no data seen yet).
    """
    try:
        return getattr(metric_obj, method_name)(*args, **kwargs)
    except Exception:
        return 0.


class MoEModel:
    """
    Unified Mixture-of-Experts class. Contains:
      - A CapyMOA DataLoader (stream)
      - CapyMOA HoeffdingTree experts (either multiclass for data mode or binary for task mode)
      - A RouterMLP
      - Methods to run different variants:
          * train_joint_data
          * train_joint_task
          * train_data
          * train_task
    config.mode must be one of ["joint_data", "joint_task", "data", "task"].
    """

    @staticmethod
    def _to_tensor(x):
        return torch.tensor(x, dtype=torch.float32)

    def __init__(self, config, seed: int):
        self.cfg = config
        self.seed = seed

        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Data stream via CapyMOA
        self.stream = DataLoader(self.cfg.dataset)
        print(type(self.stream))
        self.stream.restart()
        self.schema = self.stream.get_schema()
        self.evaluator = ClassificationEvaluator(schema=self.schema)
        
        # Save parameters
        self.input_dim = self.cfg.input_dim
        self.num_classes = self.cfg.num_classes
        self.total_samples = self.cfg.total_samples
        self.print_every = self.cfg.print_every

        
        # Decide number of experts and expert class
        if self.cfg.mode in ["task", "joint_task"]:
            # Task mode: one expert per class, each is a binary-capable CapyMOA HT
            self.n_experts = self.num_classes
            self.task_mode = True

            # Create one binary Expert per class (num_classes=2)
            self.experts = [
                Expert(schema=self.schema, num_classes=2, grace_period=50, confidence=1e-7)
                for _ in range(self.n_experts)
            ]
        else:
            # Data mode: n_experts from config, each Expert is multiclass-capable
            self.n_experts = self.cfg.n_experts
            self.task_mode = False

            self.experts = [
                Expert(schema=self.schema, num_classes=self.n_experts, grace_period=50, confidence=1e-7)
                for _ in range(self.n_experts)
            ]

        # Shared streaming‐accuracy metric for experts (prequential)
        self.exp_metrics = [metrics.Accuracy() for _ in range(self.n_experts)]

        # Router MLP
        self.router = RouterMLP(
            input_dim=self.input_dim,
            hidden_dim=self.cfg.hidden_dim,
            output_dim=self.n_experts,
            drop_prob=0.2
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.router.to(self.device)

        # Router optimizer & losses
        self.opt = torch.optim.Adam(self.router.parameters(), lr=self.cfg.lr_router)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        

    def train_joint_task(self, tracker=None):
        """
        Streaming joint training of router and task-specific experts (binary-capable CapyMOA HT).
        ON‐THE‐FLY:
          1) For each incoming instance:
             a) For each expert_i, call predict(inst) to get 0/1. Build a multi-hot correct_mask
                where correct_mask[i] = 1.0 iff expert_i predicted (orig_y==i).
             b) If no expert is correct, set correct_mask[y_true] = 1.0 to force router to know true class.
             c) Forward x through router → get logits. Argmax(logits) = pred_expert → update pipe_acc.
             d) Compute BCE loss(logits, correct_mask) → one optimizer step on router.
             e) For each expert_i, temporarily set inst.y_index = int(orig_y==i) → call expert.train(inst) → restore inst.y_index.
             f) Update prequential metric for each expert via is_correct vs pred_i.
             g) Every print_every samples, log pipeline accuracy & average expert accuracy.
        """
        
        self.evaluator = ClassificationEvaluator(schema=self.schema)
        BATCH = self.cfg.batch_size
        # Reset metrics
        
        for i in range(self.n_experts):
            self.exp_metrics[i] = metrics.Accuracy()

        total = self.total_samples
        self.stream.restart()
        running_loss = 0.0

        for t in range(1, total + 1):
            if self.stream.has_more_instances():
                inst = self.stream.next_instance()
                sys.stdout.flush()
                x_vec = inst.x  # length=input_dim
                y_true = inst.y_index  # in [0..num_classes-1]

                # Build multi-hot correct_mask exactly as in the original joint loop:
                correct_mask = np.zeros(self.n_experts, dtype=np.float32)
                for cid, expert in enumerate(self.experts):
                    # expert.predict(inst) returns 0/1; we only mark true positives
                    p_i = expert.predict(inst)
                    if p_i == 1 and y_true == cid:
                        correct_mask[cid] = 1.0
                    # Update per-expert metric: y_true_i = (y_true == cid), y_pred_i = p_i
                    self.exp_metrics[cid].update(int(y_true == cid), p_i)
                if correct_mask.sum() == 0.0:
                    correct_mask[y_true] = 1.0

                # Router forward + update
                x_t = MoEModel._to_tensor(x_vec).unsqueeze(0).to(self.device)  # 1×input_dim
                target = torch.tensor(correct_mask, dtype=torch.float32, device=self.device).unsqueeze(0)  # 1×n_experts
                logits = self.router(x_t)  # 1×n_experts

                pred_expert = int(torch.argmax(logits).item())

                self.evaluator.update(y_true, pred_expert)
                loss = self.bce_loss(logits, target)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                running_loss += loss.item() * BATCH

                # Expert updates (binary training)
                orig_label = inst.y_index
                for cid, expert in enumerate(self.experts):
                    binary_label = 1 if (orig_label == cid) else 0
                    #inst.y_index = binary_label
                    inst_copy = LabeledInstance.from_array(schema=self.schema, x=inst.x, y_index=binary_label)
                    expert.train(inst_copy)
                    # Restore original label for next expert / future usage
                    #inst.y_index = orig_label

                # Logging
                if t % self.print_every == 0:
                    curr_acc   = self.evaluator.accuracy()
                    curr_km    = self.evaluator.kappa_m()
                    curr_kt    = self.evaluator.kappa_t()
                    num_batches = max(1, (t // BATCH))
                    avg_bce = running_loss / num_batches
                    print(f"[{t:,} samples] router BCE={avg_bce:.4f}  acc={curr_acc:.4f}  κm={curr_km:.4f}  κt={curr_kt:.4f}")
                    if tracker:
                        tracker.log_step(
                            step=t,
                            loss=avg_bce,
                            accuracy=curr_acc,
                            kappa_m=curr_km,
                            kappa_temp=curr_kt
                        )
                    avg_exp_acc = np.mean([m.get() for m in self.exp_metrics])
                    print(f"[{t:>7}/{total}] PipeAcc={self.evaluator.accuracy():.4f}  AvgExpertAcc={avg_exp_acc:.4f}")
                    running_loss = 0.0

        # Final summary
        print("\n── DONE ─────────────────────────────────────────────")
        final_acc = self.evaluator.accuracy()
        final_km  = self.evaluator.kappa_m()
        final_kt  = self.evaluator.kappa_t()
        print(f"Final pipeline accuracy: {final_acc:.4f}, κm={final_km:.4f}, κt={final_kt:.4f}")
        return final_acc, final_km, final_kt

    def train_joint_data(self, tracker=None):
        """
        Online joint training of router MLP and multiclass data‐experts using BCEWithLogitsLoss.
        ON‐THE‐FLY:
        1) For each incoming instance:
            a) Router forward → get weights via softmax(logits).
            b) For each data‐expert_i, collect predict_proba(inst) → list of length=num_classes.
            c) Build a multi‐hot “correct_mask” indicating which experts would predict y_true correctly.
            d) Accumulate mini‐batch of (logits, correct_mask) for BCE router update every batch_size samples.
            e) Top‐K data‐expert_i.train(inst) updates (prequential metric for each).
            f) Update pipeline metric by picking expert = argmax(weights) → predict(inst).
            g) Every print_every samples, log average BCE loss and pipeline accuracy.
        """
        # Reset metrics
        self.evaluator = ClassificationEvaluator(schema=self.schema)
        for i in range(self.n_experts):
            self.exp_metrics[i] = metrics.Accuracy()

        total = self.total_samples
        BATCH = self.cfg.batch_size
        TOP_K = self.cfg.top_k

        running_loss = 0.0
        micro_logits, micro_multi = [], []

        # Prepare stream and router
        inst = self.stream.next_instance()
        self.stream.restart()
        self.router.train()

        for t in range(1, total + 1):
            if self.stream.has_more_instances():
                inst = self.stream.next_instance()
                x_vec = inst.x
                y_true = inst.y_index

                x_t = MoEModel._to_tensor(x_vec).unsqueeze(0).to(self.device)  # [1×input_dim]

                # Router forward
                logits = self.router(x_t)                  # [1×n_experts]
                weights = torch.softmax(logits, dim=1)     # [1×n_experts]

                # Gather experts’ probability vectors
                exp_probs_list = []
                for eid, expert in enumerate(self.experts):
                    p_list = expert.predict_proba(inst)   # list of length ≤ num_classes or None
                    if p_list is None:
                        padded = [1.0 / self.num_classes] * self.num_classes
                    elif len(p_list) < self.num_classes:
                        padded = list(p_list) + [0.0] * (self.num_classes - len(p_list))
                    else:
                        padded = list(p_list)
                    exp_probs_list.append(padded)

                exp_probs = torch.tensor(
                    exp_probs_list, dtype=torch.float32, device=self.device
                )     # [n_experts × num_classes]
                # Build multi‐hot “which experts predict y_true correctly?”
                correct_mask = torch.zeros(self.n_experts, dtype=torch.float32, device=self.device)
                for eid in range(self.n_experts):
                    pred_cls = int(torch.argmax(exp_probs[eid]).item())
                    if pred_cls == y_true:
                        correct_mask[eid] = 1.0
                if correct_mask.sum() == 0.0:
                    best_e = int(torch.argmax(exp_probs[:, y_true]).item())
                    correct_mask[best_e] = 1.0

                # Accumulate minibatch for BCE router update
                micro_logits.append(logits)                     # [1×n_experts]
                micro_multi.append(correct_mask.unsqueeze(0))   # [1×n_experts]
                if len(micro_logits) == BATCH:
                    batch_logits = torch.cat(micro_logits, dim=0)       # [BATCH×n_experts]
                    batch_multi = torch.cat(micro_multi, dim=0)         # [BATCH×n_experts]
                    loss_b = self.bce_loss(batch_logits, batch_multi)
                    self.opt.zero_grad()
                    loss_b.backward()
                    self.opt.step()
                    running_loss += loss_b.item() * BATCH
                    micro_logits.clear()
                    micro_multi.clear()

                # Top‐K data‐expert updates (train & metric)
                with torch.no_grad():
                    topk_ids = torch.topk(weights, k=TOP_K, dim=1).indices.squeeze(0)
                for eid in topk_ids.tolist():
                    self.experts[eid].train(inst)
                    p_hat = self.experts[eid].predict(inst)
                    self.exp_metrics[eid].update(int(p_hat == y_true), p_hat)

                # Running pipeline metric (choose expert = argmax(weights))
                chosen_eid = int(torch.argmax(weights).item())
                y_hat = self.experts[chosen_eid].predict(inst)
                self.evaluator.update(y_true, y_hat)

                # Logging
                if t % self.print_every == 0:
                    num_batches = max(1, (t // BATCH))
                    avg_bce = running_loss / num_batches
                    curr_acc   = self.evaluator.accuracy()
                    curr_km    = self.evaluator.kappa_m()
                    curr_kt    = self.evaluator.kappa_t()
                    print(f"[{t:,} samples] router BCE={avg_bce:.4f}  acc={curr_acc:.4f}  κm={curr_km:.4f}  κt={curr_kt:.4f}")
                    if tracker:
                        tracker.log_step(
                            step=t,
                            loss=avg_bce,
                            accuracy=curr_acc,
                            kappa_m=curr_km,
                            kappa_temp=curr_kt
                        )
                    running_loss = 0.0

        # Final summary
        print("\n── DONE ─────────────────────────────────────────────")
        print("\nPer-data-expert accuracies (prequential):")
        for eid in range(self.n_experts):
            print(f"  Data Expert {eid}: {self.exp_metrics[eid].get():.4f}")
        
        final_acc   = self.evaluator.accuracy()
        final_km    = self.evaluator.kappa_m()
        final_kt    = self.evaluator.kappa_t()
        print(f"Final pipeline accuracy: {final_acc:.4f}, κm={final_km:.4f}, κt={final_kt:.4f}")
        return final_acc, final_km, final_kt

    def _cluster_burn_in(self, burn_in_X: np.ndarray):
        """
        Perform clustering on `burn_in_X` using either KMeans or GMM (as per config).
        Returns: (best_model, scaler, pca, best_k)
        - best_model has a `.predict()` method on whitened data
        """
        # Sample a fraction for PCA/clustering
        burn_in_end = burn_in_X.shape[0]
        samp_size = max(1, int(burn_in_end * self.cfg.sample_frac))
        samp_idx = np.random.choice(burn_in_end, samp_size, replace=False)
        X_samp = burn_in_X[samp_idx]

        # Standardize + PCA (whiten)
        scaler = StandardScaler().fit(X_samp)
        X_samp_std = scaler.transform(X_samp)
        pca = PCA(n_components=self.input_dim, random_state=self.seed, whiten=True)
        X_samp_whiten = pca.fit_transform(X_samp_std)

        best_k = 2
        best_score = -1.0
        best_model = None

        for k in range(2, self.cfg.max_k + 1):
            if self.cfg.cluster_type == "kmeans":
                model = KMeans(n_clusters=k, random_state=self.seed, n_init="auto").fit(X_samp_whiten)
                labels = model.labels_
            else:  # GMM
                gm = GaussianMixture(n_components=k, random_state=self.seed, covariance_type="full", n_init=2)
                labels = gm.fit_predict(X_samp_whiten)
                model = gm

            try:
                score = silhouette_score(X_samp_whiten, labels)
            except ValueError:
                score = -1.0

            if score > best_score:
                best_score = score
                best_k = k
                best_model = model

        if best_model is None:
            # fallback: single cluster
            best_k = 1
            if self.cfg.cluster_type == "kmeans":
                best_model = KMeans(n_clusters=1, random_state=self.seed, n_init="auto").fit(X_samp_whiten)
            else:
                best_model = GaussianMixture(n_components=1, random_state=self.seed, covariance_type="full", n_init=2).fit(X_samp_whiten)

        return best_model, scaler, pca, best_k


    def train_data(self, tracker=None):
        """
        Two‐stage “half/prequential” data mode:
          • Stage 1 (first 50% of the stream):
              – Do a small burn‐in for clustering.
              – For each remaining instance in first half:
                  * cluster → test that cluster’s expert → update expert metric
                    → train that expert on this instance (prequential expert training).
              – At the end, report “Expert prequential accuracy on first half.”
          • Stage 2 (second 50% of the stream):
              – Keep experts frozen.
              – For each instance in second half:
                  1) Build a multi‐hot mask: which experts (if any) would correctly classify this instance?
                  2) Use the *current* router to predict an expert: run `eid = argmax(router(x))`, then `y_pred = expert[eid].predict(inst)`.
                     Update pipeline‐prequential accuracy (test→ no expert update).
                  3) Train the router on the multi‐hot mask from (1).
              – At the end, report “Pipeline prequential accuracy on second half.”
        """

        total = self.total_samples
        half = total // 2
        router_eval = ClassificationEvaluator(schema=self.schema)
        expert_evals = [ClassificationEvaluator(schema=self.schema)
                for _ in range(self.n_experts)]
        # ─────────────────────────────────────────────────────────
        # Stage 1: First 50%  →  cluster + expert prequential train
        # ─────────────────────────────────────────────────────────
        # 1a) Compute how many belong to “burn_in” (we can choose a small fraction, e.g. 10% of the first half)
        burn_in_end = int(half * self.cfg.burn_in_frac)
        if burn_in_end < 1:
            burn_in_end = 1
        print(f"[INFO] ==== STAGE 1: First {half:,} samples  (burn-in = {burn_in_end:,})")

        # 1b) Collect burn-in samples for clustering
        burn_in_X = []
        self.stream.restart()
        for i in range(1, burn_in_end + 1):
            inst = self.stream.next_instance()
            burn_in_X.append(inst.x)
            if i % self.print_every == 0:
                print(f"[DEBUG] Collected {i:,}/{burn_in_end:,} burn-in samples")
        burn_in_X = np.stack(burn_in_X)

        # 1c) Run clustering on burn-in_X
        print(f"[INFO] Running clustering on {burn_in_end:,} burn-in samples…")
        best_model, scaler, pca, best_k = self._cluster_burn_in(burn_in_X)
        print(f"[INFO] → Selected n_experts = {best_k} (via {self.cfg.cluster_type})")

        # Print how many burn-in points fell into each cluster
        X_std = scaler.transform(burn_in_X)
        X_wht = pca.transform(X_std)
        labels = best_model.predict(X_wht)
        uniq, counts = np.unique(labels, return_counts=True)
        print("[DEBUG] Burn-in cluster counts:")
        for cid, cnt in zip(uniq, counts):
            print(f"    Cluster {cid:2d}: {cnt:,} samples")

        # 1d) Reinitialize experts & their metrics to size = best_k
        self.n_experts = best_k
        self.experts = [
            Expert(schema=self.schema, num_classes=self.num_classes,
                   grace_period=50, confidence=1e-7)
            for _ in range(best_k)
        ]
        self.exp_metrics = [metrics.Accuracy() for _ in range(best_k)]
        print(f"[INFO] Reinitialized {best_k} data‐experts.")

        # Rebuild the router to match the new number of experts:
        self.router = RouterMLP(
                input_dim=self.input_dim,
                hidden_dim=self.cfg.hidden_dim,
                output_dim=self.n_experts,    # now equals best_k
                drop_prob=0.2
            )
        self.router.to(self.device)

        # Re‐create the router optimizer (so it optimizes the new parameters)
        self.opt = torch.optim.Adam(self.router.parameters(), lr=self.cfg.lr_router)
        # 1e) Now prequentially train + evaluate experts on the *rest* of the first half
        first_half_end = half
        print(f"[INFO] → Running prequential expert train/eval on samples {burn_in_end+1:,}–{half:,}")
        # Already consumed burn_in_end samples, so the next stream.next_instance() is sample (burn_in_end+1)

        for t in range(burn_in_end + 1, first_half_end + 1):
            inst = self.stream.next_instance()
            x_vec = inst.x
            y_true = inst.y_index

            # 1e.i) Find this instance’s cluster_id
            x_std = scaler.transform([x_vec])
            x_wht = pca.transform(x_std)
            cid = int(best_model.predict(x_wht)[0])

            # 1e.ii) Test this expert, then update that expert metric
            y_pred_ex = self.experts[cid].predict(inst)
            expert_evals[cid].update(y_true, y_pred_ex)

            # 1e.iii) Train expert cid on this instance
            self.experts[cid].train(inst)

            if (t - burn_in_end) % self.print_every == 0:
                done = t - burn_in_end
                total_slice = first_half_end - burn_in_end
                accs = [_safe_metric(ev, "accuracy") for ev in expert_evals]
                avg_acc = sum(accs) / len(accs)
                kappa_ms = [_safe_metric(ev, "kappa_m") for ev in expert_evals]
                avg_kappa_m = sum(kappa_ms) / len(kappa_ms)
                kappa_ts = [_safe_metric(ev, "kappa_t") for ev in expert_evals]
                avg_kappa_t = sum(kappa_ts) / len(kappa_ts)
                print(f"[{t:,}] Stage 1 expert preq: processed {done:,}/{total_slice:,}, "
                      f"AvgExpertAcc={avg_acc:.4f}")

                # log to tracker (no meaningful loss yet)
                if tracker:
                    tracker.log_step(
                        step=t,
                        loss=0.0,
                        accuracy=avg_acc,
                        kappa_m=avg_kappa_m,
                        kappa_temp=avg_kappa_t
                    )
                    print(f"[DEBUG] Expert preq stage 1: processed {done:,}/{total_slice:,}")

        print(f"\n── Expert prequential accuracy on first 50%: {avg_acc:.4f}\n")

        # ─────────────────────────────────────────────────────────
        # Stage 2: Last 50%  →  train router & evaluate pipeline prequentially
        # ─────────────────────────────────────────────────────────
        second_half_start = half + 1
        second_half_end   = total
        print(f"[INFO] ==== STAGE 2: Last {total-half:,} samples  (samples {second_half_start:,}–{second_half_end:,})")
        print("[INFO] Experts are now frozen; we will only train/evaluate the router.")

        # Create a router‐prequential accuracy metric (pipeline accuracy)

        # For every instance in the second half:
        #   2a) Build multi‐hot “which experts are correct” mask,
        #   2b) Evaluate pipeline (router->expert) → update pipe_preq_acc,
        #   2c) Train router on that same instance’s mask.
        #
        # Note: At index = (half+1), the stream is already at that position, because
        #       we consumed exactly 'half' calls to next_instance() in Stage 1.

        for t in range(second_half_start, second_half_end + 1):
            inst = self.stream.next_instance()
            x_vec = inst.x
            y_true = inst.y_index

            # — Step 2a: multi-hot mask from frozen experts —
            multi = np.zeros(self.n_experts, dtype=np.float32)
            for cid in range(self.n_experts):
                p_list = self.experts[cid].predict_proba(inst)
                if p_list is None:
                    padded = [1.0 / self.num_classes] * self.num_classes
                elif len(p_list) < self.num_classes:
                    padded = list(p_list) + [0.0] * (self.num_classes - len(p_list))
                else:
                    padded = list(p_list)
                pred_c = int(np.argmax(padded))
                if pred_c == y_true:
                    multi[cid] = 1.0

            # — Step 2b: pipeline evaluation (router → chosen expert → expert.predict) —
            x_t = MoEModel._to_tensor(x_vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scores = self.router(x_t).sigmoid().squeeze(0)  # [n_experts]
                chosen_e = int(torch.argmax(scores).item())
            y_pred_pipeline = self.experts[chosen_e].predict(inst)
            router_eval.update(y_true, y_pred_pipeline)

            # — Step 2c: train the router on multi-hot mask —
            self.router.train()
            xb = x_t  # shape = [1, input_dim]
            yb = torch.tensor(multi, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, n_experts]
            bce = nn.BCEWithLogitsLoss()
            opt = torch.optim.Adam(self.router.parameters(), lr=self.cfg.lr_router)
            opt.zero_grad()
            logits = self.router(xb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()

            if (t - second_half_start) % self.print_every == 0:
                done2 = t - second_half_start
                tot2  = second_half_end - second_half_start + 1
                curr_acc   = router_eval.accuracy()
                curr_km    = router_eval.kappa_m()
                curr_kt    = router_eval.kappa_t()
                print(f"[{t:,} samples] router BCE={loss.item():.4f}  acc={curr_acc:.4f}  κm={curr_km:.4f}  κt={curr_kt:.4f}")
                if tracker:
                    tracker.log_step(
                        step=t,
                        loss=loss.item(),
                        accuracy=curr_acc,
                        kappa_m=curr_km,
                        kappa_temp=curr_kt
                    )

        print(f"\n── Pipeline prequential accuracy on second 50%: {router_eval.accuracy():.4f}\n")

        print("[INFO] ==== TRAIN_DATA COMPLETE ====")
        return router_eval.accuracy(), router_eval.kappa_m(), router_eval.kappa_t()
    def train_task(self, tracker=None):
        """
        Task‐mode with prequential evaluation, aligned with 0_Moe.py:
        • Stage 1 (first 50%): prequential test-then-train each binary expert.
        • Stage 2 (last 50%): prequential test-then-train router on one-hot ground truth.
        """
        from capymoa.instance import LabeledInstance

        total = self.total_samples
        half  = total // 2

        router_eval = ClassificationEvaluator(schema=self.schema)
        expert_evals = [ClassificationEvaluator(schema=self.schema)
                for _ in range(self.n_experts)]

        # ─── Stage 1: Expert prequential on first half ───
        print(f"[INFO] Stage 1: Expert prequential on first {half:,} samples")
        self.stream.restart()

        for t in range(1, half+1):
            inst   = self.stream.next_instance()
            y_true = inst.y_index

            for cid, expert in enumerate(self.experts):
                # build a fresh binary‐labeled instance
                bin_lbl   = 1 if (y_true == cid) else 0
                inst_copy = LabeledInstance.from_array(self.schema, inst.x, bin_lbl)

                # test
                y_pred = expert.predict(inst_copy)
                expert_evals[cid].update(bin_lbl, y_pred)

                # train
                expert.train(inst_copy)

            if t % self.print_every == 0:

                accs = [_safe_metric(ev, "accuracy") for ev in expert_evals]
                avg_acc = sum(accs) / len(accs)
                kappa_ms = [_safe_metric(ev, "kappa_m") for ev in expert_evals]
                avg_kappa_m = sum(kappa_ms) / len(kappa_ms)
                kappa_ts = [_safe_metric(ev, "kappa_t") for ev in expert_evals]
                avg_kappa_t = sum(kappa_ts) / len(kappa_ts)
                print(f"[{t:,}] Expert Stage 1: {t:,}/{half:,}  AvgExpertAcc={avg_acc:.4f}")

                # tracker log
                if tracker:
                    tracker.log_step(
                        step=t,
                        loss=0.0,
                        accuracy=avg_acc,
                        kappa_m=avg_kappa_m,
                        kappa_temp=avg_kappa_t
                    )
                print(f"[DEBUG] Expert Stage 1: {t:,}/{half:,} samples")

        print("\n── Expert prequential accuracy (first 50%) ──")
        for cid, m in enumerate(expert_evals):
            print(f" Expert {cid}: {m.accuracy():.4f}")

        # ─── Stage 2: Router prequential on last half ───
        print(f"\n[INFO] Stage 2: Router prequential on last {total-half:,} samples")
        router    = self.router
        opt       = torch.optim.Adam(router.parameters(), lr=self.cfg.lr_router)
        bce       = nn.BCEWithLogitsLoss()

        for t in range(half+1, total+1):
            inst   = self.stream.next_instance()
            y_true = inst.y_index

            # build one-hot ground truth mask for router
            one_hot = np.zeros(self.n_experts, dtype=np.float32)
            one_hot[y_true] = 1.0

            # test pipeline
            x_t    = MoEModel._to_tensor(inst.x).unsqueeze(0).to(self.device)
            router.eval()
            with torch.no_grad():
                scores = router(x_t).sigmoid().squeeze(0)
                chosen = int(torch.argmax(scores).item())
            router_eval.update(y_true, chosen)
            # train router on one-hot mask
            router.train()
            yb = torch.tensor(one_hot, dtype=torch.float32, device=self.device).unsqueeze(0)
            opt.zero_grad()
            loss = bce(router(x_t), yb)
            loss.backward()
            opt.step()

            if (t-half) % self.print_every == 0:
                done = t - half
                tot  = total - half
                curr_acc = router_eval.accuracy()
                curr_km  = router_eval.kappa_m()
                curr_kt  = router_eval.kappa_t()
                print(f"[{t:,}] router acc={curr_acc:.4f} κm={curr_km:.4f} κt={curr_kt:.4f} ")

                if tracker:
                    tracker.log_step(
                        step=t,
                        loss=loss.item(),
                        accuracy=curr_acc,
                        kappa_m=curr_km,
                        kappa_temp=curr_kt
                    )
                print(f"[DEBUG] Router Stage 2: {done:,}/{tot:,} samples  PipeAcc={router_eval.accuracy():.4f}")

        print(f"\n🏁 Pipeline prequential accuracy (last 50%): {router_eval.accuracy():.4f}")
        return router_eval.accuracy(), router_eval.kappa_m(), router_eval.kappa_t()

