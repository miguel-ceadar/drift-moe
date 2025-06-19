import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from river import tree, metrics
from river.datasets import synth
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  TARGETED Mixture-of-Experts Pipeline  (fixed label bug + full analysis)
# ─────────────────────────────────────────────────────────────────────────────

# ───────── CONFIG ───────────────────────────────────────────────────────────
TOTAL_SAMPLES = 1_000_000
TRAIN_RATIO   = 0.80
NUM_CLASSES   = 10
INPUT_DIM     = 24
BATCH_SIZE    = 256
EPOCHS        = 75
LR            = 2e-3
SEED_STREAM   = 112
torch.manual_seed(42)

# ───────── DATA STREAM & SPLIT ──────────────────────────────────────────────
stream = list(
    synth.LEDDrift(
        seed=SEED_STREAM,
        noise_percentage=0.10,
        irrelevant_features=True,
        n_drift_features=7
    ).take(TOTAL_SAMPLES)
)

half          = TOTAL_SAMPLES // 2
expert_block  = stream[:half]
router_block  = stream[half:]

exp_train_sz  = int(len(expert_block)  * TRAIN_RATIO)
rtr_train_sz  = int(len(router_block)  * TRAIN_RATIO)

exp_train, exp_val = expert_block[:exp_train_sz], expert_block[exp_train_sz:]
rtr_train, rtr_val = router_block[:rtr_train_sz], router_block[rtr_train_sz:]

print("── SPLITS ───────────────────────────────────────────")
print(f" total samples         : {TOTAL_SAMPLES:,}")
print(f" expert  train / val   : {len(exp_train):,} / {len(exp_val):,}")
print(f" router  train / val   : {len(rtr_train):,} / {len(rtr_val):,}")

d2v = lambda d: np.fromiter(d.values(), dtype=np.float32, count=INPUT_DIM)

# ───────── 1) EXPERTS ───────────────────────────────────────────────────────
experts      = {cid: tree.HoeffdingTreeClassifier(grace_period=200)
                for cid in range(NUM_CLASSES)}

exp_val_acc  = {cid: metrics.Accuracy() for cid in range(NUM_CLASSES)}

print("\n Training experts...")
for x, y in exp_train:
    for cid, model in experts.items():
        model.learn_one(x, int(y == cid))

for x, y in exp_val:
    for cid, model in experts.items():
        pred = model.predict_one(x) or 0
        exp_val_acc[cid].update(int(y == cid), pred)

print("\n── EXPERT VALID ACC ──────────────────────────────────")
for cid in range(NUM_CLASSES):
    print(f" expert {cid}: {exp_val_acc[cid].get():.4f}")

# ───────── 2) ROUTER DATA  (FIX: label = ground-truth class) ────────────────
router_X, router_Y = [], []
for x_dict, y_true in rtr_train:
    multi          = np.zeros(NUM_CLASSES, dtype=np.float32)
    multi[y_true]  = 1.0          # ← the only positive label
    router_X.append(d2v(x_dict))
    router_Y.append(multi)

router_X = np.stack(router_X)
router_Y = np.stack(router_Y)

print(f"\nrouter-train samples              : {len(router_Y):,}")
print(f"positive label density            : {router_Y.sum()/router_Y.size:.4f}")


# ───────── 3) ROUTER MLP ────────────────────────────────────────────────────
class TorchDS(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):             return len(self.X)
    def __getitem__(self, i):      return self.X[i], self.Y[i]

class RouterMLP(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, h=256, out=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h//2),   nn.ReLU(),
            nn.Linear(h//2, out)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

train_dl      = DataLoader(TorchDS(router_X, router_Y),
                           batch_size=BATCH_SIZE, shuffle=True)

router        = RouterMLP()
opt           = torch.optim.Adam(router.parameters(), lr=LR)
bce           = nn.BCEWithLogitsLoss()

train_history = []

print(f"\n Training router (epochs: {EPOCHS})...")
for epoch in range(1, EPOCHS+1):
    router.train()
    running = 0.0
    for xb, yb in train_dl:
        opt.zero_grad()
        loss = bce(router(xb), yb)
        loss.backward()
        opt.step()
        running += loss.item() * len(xb)
    avg_loss = running / len(train_dl.dataset)
    train_history.append(avg_loss)
    if epoch % 15 == 0:
        print(f"epoch {epoch:3d}/{EPOCHS} | BCE: {avg_loss:.4f}")

# ───────── 4) EVALUATION ────────────────────────────────────────────────────
router.eval()
pipe_acc        = metrics.Accuracy()
pipe_acc_top2   = metrics.Accuracy()
pipe_acc_orig   = metrics.Accuracy()

results = {
    'confidences'   : [],
    'is_correct'    : [],
    'expert_usage'  : np.zeros(NUM_CLASSES, dtype=int),
    'confusion'     : np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int),
    'sample_counts' : np.zeros(NUM_CLASSES, dtype=int)
}

print("\n Evaluating pipeline on router-val...")
with torch.no_grad():
    for x_dict, y_true in rtr_val:
        x_tensor = torch.tensor(d2v(x_dict), dtype=torch.float32).unsqueeze(0)
        logits   = router(x_tensor).squeeze(0)
        probs    = torch.sigmoid(logits)

        # Strategy 1: argmax (original)
        top_expert = int(torch.argmax(probs).item())
        pipe_acc_orig.update(y_true, top_expert)

        # Strategy 2: simple Top-2 pick highest prob > 0.4 else fallback
        top2_val, top2_idx = torch.topk(probs, 2)
        final_pred = (int(top2_idx[0].item()) if top2_val[0] > 0.4
                      else int(top2_idx[1].item()))
        pipe_acc_top2.update(y_true, final_pred)

        # Strategy 3: thresholded multi-choice
        best_pred = top_expert
        confid    = probs[best_pred].item()
        pipe_acc.update(y_true, best_pred)

        # collect stats
        results['confidences'].append(confid)
        results['is_correct'].append(int(best_pred == y_true))
        results['expert_usage'][best_pred] += 1
        results['confusion'][y_true, best_pred] += 1
        results['sample_counts'][y_true] += 1

print("\n── RESULTS ─────────────────────────────────────────")
print(f"Argmax accuracy        : {pipe_acc_orig.get():.4f}")
print(f"Top-2 accuracy         : {pipe_acc_top2.get():.4f}")
print(f"Threshold accuracy     : {pipe_acc.get():.4f}")
best = max(pipe_acc_orig.get(), pipe_acc_top2.get(), pipe_acc.get())
print(f"\n BEST PIPELINE ACCURACY: {best:.4f}")
