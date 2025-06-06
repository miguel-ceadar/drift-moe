import torch, torch.nn as nn
import numpy as np
from river import tree, metrics
from river.datasets import synth
import warnings, time

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
TOTAL_SAMPLES = 1_000_000
NUM_CLASSES   = 10
INPUT_DIM     = 24
LR            = 1e-3
SEED_STREAM   = 112
LOG_EVERY     = 100
torch.manual_seed(42)

# ────────────────────────────────────────────────────────────────────────────
# DATA STREAM
# ────────────────────────────────────────────────────────────────────────────
stream = synth.LEDDrift(
    seed=SEED_STREAM,
    noise_percentage=0.10,
    irrelevant_features=True,
    n_drift_features=7
).take(TOTAL_SAMPLES)

print("── CONFIG ───────────────────────────────────────────")
print(f" total samples : {TOTAL_SAMPLES:,}")
print(f" num classes   : {NUM_CLASSES}")
print(f" input dim     : {INPUT_DIM}")
print(f" log frequency : {LOG_EVERY:,} samples")

# helper to turn feature-dict into fixed-length np.array
d2v = lambda d: np.fromiter(d.values(), dtype=np.float32, count=INPUT_DIM)

# ────────────────────────────────────────────────────────────────────────────
# 1) MODELS INITIALIZATION
# ────────────────────────────────────────────────────────────────────────────
# Experts are still trained in parallel to learn their specializations
experts = {cid: tree.HoeffdingTreeClassifier(grace_period=100) for cid in range(NUM_CLASSES)}

class RouterMLP(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, h=128, out=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, out)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

router = RouterMLP()
opt = torch.optim.Adam(router.parameters(), lr=LR)
# ⭐️ CHANGE 1: Switched to CrossEntropyLoss for single-label multi-class classification
criterion = nn.CrossEntropyLoss()

# ────────────────────────────────────────────────────────────────────────────
# 2) PARALLEL ONLINE TRAINING & INCREMENTAL VALIDATION
# ────────────────────────────────────────────────────────────────────────────
print("\n Starting parallel online training ...\n")
start_time = time.time()
pipe_acc = metrics.Accuracy()

for i, (x_dict, y_true) in enumerate(stream):

    x_vec = torch.tensor(d2v(x_dict), dtype=torch.float32).unsqueeze(0)

    # --- a) Incremental Pipeline Validation (Prequential) ---
    # The router predicts the final class directly.
    router.eval()
    with torch.no_grad():
        prediction_logits = router(x_vec)
        final_prediction = int(torch.argmax(prediction_logits).item())

    pipe_acc.update(y_true, final_prediction)

    # --- b) Simultaneous Training Step ---
    # 1. Train Router
    router.train()
    opt.zero_grad()
    training_logits = router(x_vec)

    # ⭐️ CHANGE 2: The target is now the ground truth label, giving a clear signal.
    target_tensor = torch.tensor([y_true], dtype=torch.long)
    loss = criterion(training_logits, target_tensor)

    if not torch.isnan(loss):
        loss.backward()
        opt.step()

    # 2. Train Experts (This happens in parallel, as before)
    # The experts continue to learn their binary classification tasks.
    for cid, expert in experts.items():
        y_expert_binary_true = 1 if y_true == cid else 0
        expert.learn_one(x_dict, y_expert_binary_true)

    # --- c) Logging ---
    if (i + 1) % LOG_EVERY == 0:
        elapsed = time.time() - start_time
        print(f"  Samples: {i+1:>7,}/{TOTAL_SAMPLES:,} | Pipeline Acc: {pipe_acc.get():.4f} | Time: {elapsed:.2f}s")


# ────────────────────────────────────────────────────────────────────────────
# 3) FINAL RESULTS
# ────────────────────────────────────────────────────────────────────────────
print("\n── FINAL RESULTS ───────────────────────────────────")
print(f"✅ Final Pipeline Accuracy: {pipe_acc.get():.4f}")
print("──────────────────────────────────────────────────")