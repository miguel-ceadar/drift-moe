# moe_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from river import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from data_loader import DataLoader
from experts import Expert


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
        self.stream = DataLoader(self.cfg.cli)
        self.stream.restart()
        self.schema = self.stream.get_schema()

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
                Expert(schema=self.schema, grace_period=50, confidence=1e-7)
                for _ in range(self.n_experts)
            ]
        else:
            # Data mode: n_experts from config, each Expert is multiclass-capable
            self.n_experts = self.cfg.n_experts
            self.task_mode = False

            self.experts = [
                Expert(schema=self.schema, grace_period=50, confidence=1e-7)
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

        # Pipeline accuracy metric
        self.pipe_acc = metrics.Accuracy()

    def train_joint_task(self):
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
        # Reset metrics
        self.pipe_acc = metrics.Accuracy()
        for i in range(self.n_experts):
            self.exp_metrics[i] = metrics.Accuracy()

        total = self.total_samples
        self.stream.restart()

        for idx in range(1, total + 1):
            inst = self.stream.next_instance()
            x_vec = inst.x  # length=input_dim
            y_true = inst.y_index  # in [0..num_classes-1]

            # Build multi-hot correct_mask
            correct_mask = np.zeros(self.n_experts, dtype=np.float32)
            # For each expert_i, ask expert_i.predict(inst) which returns 0 or 1
            for cid, expert in enumerate(self.experts):
                # expert.predict(inst) uses features only
                p_i = expert.predict(inst)  # 0 or 1
                is_true_i = int(y_true == cid)
                if p_i == is_true_i:
                    correct_mask[cid] = 1.0
                # update per-expert metric
                self.exp_metrics[cid].update(int(p_i == is_true_i), p_i)

            # Fallback if no expert was correct
            if correct_mask.sum() == 0:
                correct_mask[y_true] = 1.0

            # Router forward + update
            x_t = MoEModel._to_tensor(x_vec).unsqueeze(0).to(self.device)  # 1×input_dim
            target = torch.tensor(correct_mask, dtype=torch.float32, device=self.device).unsqueeze(0)  # 1×n_experts
            logits = self.router(x_t)  # 1×n_experts

            pred_expert = int(torch.argmax(logits).item())
            self.pipe_acc.update(y_true, pred_expert)

            loss = self.bce_loss(logits, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Expert updates (binary training)
            orig_label = inst.y_index
            for cid, expert in enumerate(self.experts):
                binary_label = 1 if (orig_label == cid) else 0
                inst.y_index = binary_label
                expert.train(inst)
                # Restore original label for next expert / future usage
                inst.y_index = orig_label

            # Logging
            if idx % self.print_every == 0:
                avg_exp_acc = np.mean([m.get() for m in self.exp_metrics])
                print(f"[{idx:>7}/{total}] PipeAcc={self.pipe_acc.get():.4f}  AvgExpertAcc={avg_exp_acc:.4f}")

        # Final summary
        print("\n── DONE ─────────────────────────────────────────────")
        print(f"Final pipeline accuracy (argmax): {self.pipe_acc.get():.4f}")
        print("\nPer-task-expert accuracies (prequential):")
        for cid in range(self.n_experts):
            print(f"  Task Expert {cid}: {self.exp_metrics[cid].get():.4f}")

    def train_joint_data(self):
        """
        Online joint training of router MLP and multiclass data-experts (CapyMOA HT).
        ON‐THE‐FLY:
          1) For each incoming instance:
             a) Router forward → get weights via softmax(logits).
             b) For each data-expert_i, collect predict_proba(inst) → length=num_classes.
             c) mix_prob = weights @ exp_probs → log_mix = log(mix_prob / sum).
             d) Accumulate mini-batch of (log_mix, y_true) for NLL router update every batch_size samples.
             e) Top-K data-expert_i.train(inst) updates (prequential metric for each).
             f) Update pipeline metric via argmax(mix_prob).
             g) Every print_every samples, log router CE and pipeline accuracy.
        """
        # Reset metrics
        self.pipe_acc = metrics.Accuracy()
        for i in range(self.n_experts):
            self.exp_metrics[i] = metrics.Accuracy()

        total = self.total_samples
        BATCH = self.cfg.batch_size
        TOP_K = self.cfg.top_k

        nll = nn.NLLLoss(reduction="mean")
        running_loss = 0.0
        micro_X, micro_y = [], []

        self.stream.restart()
        self.router.train()

        for t in range(1, total + 1):
            inst = self.stream.next_instance()
            x_vec = inst.x
            y_true = inst.y_index

            x_t = MoEModel._to_tensor(x_vec).unsqueeze(0).to(self.device)  # 1×input_dim

            # Router forward
            logits = self.router(x_t)            # 1×n_experts
            weights = torch.softmax(logits, dim=1)  # 1×n_experts

            # Gather experts’ probability vectors
            exp_probs_list = []
            for eid, expert in enumerate(self.experts):
                p_list = expert.predict_proba(inst)  # length ≤ num_classes
                if p_list is None:
                    padded = [1.0 / self.num_classes] * self.num_classes
                elif len(p_list) < self.num_classes:
                    padded = list(p_list) + [0.0] * (self.num_classes - len(p_list))
                else:
                    padded = list(p_list)
                exp_probs_list.append(padded)

            exp_probs = torch.tensor(exp_probs_list, dtype=torch.float32, device=self.device)  # n_experts × num_classes

            # Mix & log
            mix_prob = torch.mm(weights, exp_probs) + 1e-9  # 1×num_classes
            log_mix = (mix_prob / mix_prob.sum()).log()  # 1×num_classes

            # Accumulate mini-batch for router update
            micro_X.append(log_mix)
            micro_y.append(y_true)
            if len(micro_X) == BATCH:
                batch_X = torch.cat(micro_X, dim=0)        # B × num_classes
                batch_y = torch.tensor(micro_y, dtype=torch.long, device=self.device)  # B
                loss = nll(batch_X, batch_y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                running_loss += loss.item() * BATCH
                micro_X.clear()
                micro_y.clear()

            # Top-K data-expert updates (train & metric)
            with torch.no_grad():
                topk_ids = torch.topk(weights, k=TOP_K, dim=1).indices.squeeze(0)
            for eid in topk_ids.tolist():
                self.experts[eid].train(inst)
                p_hat = self.experts[eid].predict(inst)
                self.exp_metrics[eid].update(int(p_hat == y_true), p_hat)

            # Running pipeline metric
            y_hat = int(torch.argmax(mix_prob).item())
            self.pipe_acc.update(y_true, y_hat)

            # Logging
            if t % self.print_every == 0:
                avg_ce = running_loss / max(1, (t // BATCH))
                print(f"[{t:,} samples]  router CE: {avg_ce:.4f}   pipeline acc: {self.pipe_acc.get():.4f}")
                running_loss = 0.0

        # Final summary
        print("\n── DONE ─────────────────────────────────────────────")
        print(f"Final pipeline accuracy (argmax): {self.pipe_acc.get():.4f}")
        print("\nPer-data-expert accuracies (prequential):")
        for eid in range(self.n_experts):
            print(f"  Data Expert {eid}: {self.exp_metrics[eid].get():.4f}")

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

    def train_data(self):
        """
        Two-stage pipeline for 'data' mode:
         - Stage 1 (first half):
             a) Burn-in fraction → collect features for clustering.
             b) _cluster_burn_in(...) → returns best_kmeans/GMM, scaler, pca.
             c) Assign all burn-in points to clusters.
             d) Next portion of first half (80/20 split) → train/val data-experts.
         - Stage 2 (second half):
             e) Build router offline training set (multi-hot correct mask from data experts).
             f) Offline train router with BCE.
             g) Evaluate on router validation.
        """
        total = self.total_samples
        half = total // 2
        burn_in_end = int(total * self.cfg.burn_in_frac)
        burn_in_end = min(burn_in_end, half - 1)

        # Stage 1a: Burn-in → collect x vectors
        burn_in_X = []
        self.stream.restart()
        for i in range(1, burn_in_end + 1):
            inst = self.stream.next_instance()
            burn_in_X.append(inst.x)
        burn_in_X = np.stack(burn_in_X)  # [burn_in_end, input_dim]

        # Stage 1b: clustering
        best_model, scaler, pca, best_k = self._cluster_burn_in(burn_in_X)
        print(f"Selected n_experts via clustering = {best_k} (using {self.cfg.cluster_type})")

        # Stage 1c: assign all burn-in points to clusters
        burn_in_std = scaler.transform(burn_in_X)
        burn_in_whiten = pca.transform(burn_in_std)
        if self.cfg.cluster_type == "kmeans":
            burn_in_labels = best_model.predict(burn_in_whiten)
        else:
            burn_in_labels = best_model.predict(burn_in_whiten)

        # Stage 1d: remainder of first half → train & validate data-experts
        rem_count = half - burn_in_end
        expert_train_end = burn_in_end + int(rem_count * 0.8)
        # expert_val_end = half  (unused explicitly)
        self.stream.restart()
        for _ in range(burn_in_end):
            _ = self.stream.next_instance()

        exp_val_acc = [metrics.Accuracy() for _ in range(best_k)]

        # EXPERT TRAINING slice
        for i in range(burn_in_end + 1, expert_train_end + 1):
            inst = self.stream.next_instance()
            x_vec = inst.x
            y_true = inst.y_index

            x_std = scaler.transform([x_vec])
            x_whiten = pca.transform(x_std)
            cluster_id = int(best_model.predict(x_whiten)[0])
            self.experts[cluster_id].train(inst)

        # EXPERT VALIDATION slice
        for i in range(expert_train_end + 1, half + 1):
            inst = self.stream.next_instance()
            x_vec = inst.x
            y_true = inst.y_index

            x_std = scaler.transform([x_vec])
            x_whiten = pca.transform(x_std)
            cluster_id = int(best_model.predict(x_whiten)[0])

            pred_c = self.experts[cluster_id].predict(inst)
            exp_val_acc[cluster_id].update(y_true, pred_c)

        print("\n── EXPERT VALIDATION ACC (first half) ─────────────")
        for cid in range(best_k):
            print(f"  Data Expert {cid}: {exp_val_acc[cid].get():.4f}")

        # Stage 2: Router offline training (second half)
        rem2_count = total - half
        rtr_train_end = half + int(rem2_count * 0.8)
        # rtr_val_end = total
        self.stream.restart()
        for _ in range(half):
            _ = self.stream.next_instance()

        router_X = []
        router_Y = []

        for i in range(half + 1, rtr_train_end + 1):
            inst = self.stream.next_instance()
            x_vec = inst.x
            y_true = inst.y_index

            x_std = scaler.transform([x_vec])
            x_whiten = pca.transform(x_std)
            cluster_id = int(best_model.predict(x_whiten)[0])

            # Build multi-hot mask: expert_i is “correct” if expert_i.predict_proba(inst) ’s argmax == y_true
            multi = np.zeros(best_k, dtype=np.float32)
            for cid in range(best_k):
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

            router_X.append(x_vec)
            router_Y.append(multi)

        router_X = np.stack(router_X)  # [N_rtr_train, input_dim]
        router_Y = np.stack(router_Y)  # [N_rtr_train, best_k]

        print(f"\nRouter-train samples: {len(router_Y):,}")
        print(f"Positive-label density: {router_Y.sum() / (router_Y.size):.4f}")

        # Create Torch Dataset for router (multi-label)
        from torch.utils.data import Dataset, DataLoader

        class TorchDS_Multi(Dataset):
            def __init__(self, X, Y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.Y = torch.tensor(Y, dtype=torch.float32)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.Y[idx]

        train_ds = TorchDS_Multi(router_X, router_Y)
        train_dl = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)

        # Offline train router with BCE
        router = self.router
        router.train()
        bce = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(router.parameters(), lr=self.cfg.lr_router)

        for epoch in range(1, self.cfg.epochs + 1):
            running = 0.0
            for xb, yb in train_dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = router(xb)
                loss = bce(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item() * len(xb)
            avg_loss = running / len(train_dl.dataset)
            if epoch % 10 == 0 or epoch == self.cfg.epochs:
                print(f"Epoch {epoch}/{self.cfg.epochs} | BCE: {avg_loss:.4f}")

        # Evaluate on router validation (second‐half hold‐out)
        print("\n── EVALUATING ROUTER on validation slice ────────────")
        router.eval()
        pipe_acc = metrics.Accuracy()

        for i in range(rtr_train_end + 1, total + 1):
            inst = self.stream.next_instance()
            x_vec = inst.x
            y_true = inst.y_index

            x_t = MoEModel._to_tensor(x_vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scores = router(x_t).sigmoid().squeeze(0)  # [best_k]
                eid = int(torch.argmax(scores).item())
                y_pred = self.experts[eid].predict(inst)
                pipe_acc.update(y_true, y_pred)

        print(f"\n🏁  Pipeline accuracy on router-val slice: {pipe_acc.get():.4f}")

    def train_task(self):
        """
        Two-stage pipeline for 'task' mode (binary-capable CapyMOA HTs for each class):
         - Stage 1 (first half): train each expert_i on binary label (orig_y == i).
         - Stage 2 (second half):
             a) Build router offline training set (multi-hot correct mask).
             b) Offline train router with BCE.
             c) Evaluate on router validation slice.
        """
        total = self.total_samples
        half = total // 2

        # Stage 1: Train task experts on first half
        self.stream.restart()
        for i in range(1, half + 1):
            inst = self.stream.next_instance()
            x_vec = inst.x
            y_true = inst.y_index
            orig_label = y_true

            # For each expert_i: set inst.y_index=binary, train, restore
            for cid, expert in enumerate(self.experts):
                inst.y_index = 1 if (orig_label == cid) else 0
                expert.train(inst)
                inst.y_index = orig_label
                # No metric update yet for offline stage

        # Stage 2: Build offline router training set
        rem2_count = total - half
        rtr_train_end = half + int(rem2_count * 0.8)
        # rtr_val_end = total
        self.stream.restart()
        for _ in range(half):
            _ = self.stream.next_instance()

        router_X = []
        router_Y = []

        for i in range(half + 1, rtr_train_end + 1):
            inst = self.stream.next_instance()
            x_vec = inst.x
            y_true = inst.y_index
            orig_label = y_true

            # Build multi-hot: expert_i is correct if expert_i.predict(inst) == (orig_label==i)
            multi = np.zeros(self.n_experts, dtype=np.float32)
            for cid, expert in enumerate(self.experts):
                p_i = expert.predict(inst)  # 0 or 1
                is_true_i = int(orig_label == cid)
                if p_i == is_true_i:
                    multi[cid] = 1.0

            router_X.append(x_vec)
            router_Y.append(multi)

        if len(router_Y) == 0:
            print("No usable samples for router training (all skipped). Exiting.")
            return

        router_X = np.stack(router_X)  # [N_rtr_train, input_dim]
        router_Y = np.stack(router_Y)  # [N_rtr_train, n_experts]

        print(f"\nRouter-train samples: {len(router_Y):,}")
        print(f"Positive-label density: {router_Y.sum() / (router_Y.size):.4f}")

        # Create Torch Dataset for router (multi-label)
        from torch.utils.data import Dataset, DataLoader

        class TorchDS_Multi(Dataset):
            def __init__(self, X, Y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.Y = torch.tensor(Y, dtype=torch.float32)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.Y[idx]

        train_ds = TorchDS_Multi(router_X, router_Y)
        train_dl = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)

        # Offline train router with BCE
        router = self.router
        router.train()
        bce = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(router.parameters(), lr=self.cfg.lr_router)

        for epoch in range(1, self.cfg.epochs + 1):
            running = 0.0
            for xb, yb in train_dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = router(xb)
                loss = bce(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item() * len(xb)
            avg_loss = running / len(train_dl.dataset)
            if epoch % 10 == 0 or epoch == self.cfg.epochs:
                print(f"Epoch {epoch}/{self.cfg.epochs} | BCE: {avg_loss:.4f}")

        # Evaluate on router validation slice (second half)
        print("\n── EVALUATING ROUTER on validation slice ────────────")
        router.eval()
        pipe_acc = metrics.Accuracy()

        for i in range(rtr_train_end + 1, total + 1):
            inst = self.stream.next_instance()
            x_vec = inst.x
            y_true = inst.y_index

            x_t = MoEModel._to_tensor(x_vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scores = router(x_t).sigmoid().squeeze(0)  # [n_experts]
                eid = int(torch.argmax(scores).item())
                p_i = self.experts[eid].predict(inst)  # 0 or 1
                is_true = int(y_true == eid)
                final = eid if (p_i == is_true) else -1
                pipe_acc.update(y_true, final)

        print(f"\n🏁  Pipeline accuracy on router-val slice: {pipe_acc.get():.4f}")


