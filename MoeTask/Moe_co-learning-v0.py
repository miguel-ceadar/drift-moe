# moe_streaming_pipeline.py
import torch, torch.nn as nn
import numpy as np
# from river import tree, metrics, synth

from river import tree, metrics
from river.datasets import synth

import warnings, itertools, time
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
TOTAL_SAMPLES = 1_000_000
TRAIN_RATIO   = 0.80                # unused now – we stream everything
NUM_CLASSES   = 10
INPUT_DIM     = 24
LR_ROUTER     = 2e-3
SEED_STREAM   = 112
PRINT_EVERY   = 10_000              # status interval
torch.manual_seed(42)

d2v = lambda d: np.fromiter(d.values(), dtype=np.float32, count=INPUT_DIM)

# ────────────────────────────────────────────────────────────────────────────
# STREAM (LED-Drift)   – one long generator
# ────────────────────────────────────────────────────────────────────────────
stream = synth.LEDDrift(
    seed=SEED_STREAM,
    noise_percentage=0.10,
    irrelevant_features=True,
    n_drift_features=7
).take(TOTAL_SAMPLES)   # generator, not list – keeps memory tiny

# ────────────────────────────────────────────────────────────────────────────
# EXPERTS  (incremental Decision Trees)
# ────────────────────────────────────────────────────────────────────────────
experts   = {cid: tree.HoeffdingTreeClassifier(grace_period=200)
             for cid in range(NUM_CLASSES)}
exp_metrics = {cid: metrics.Accuracy() for cid in range(NUM_CLASSES)}

# ────────────────────────────────────────────────────────────────────────────
# ROUTER  (tiny MLP, online BCE optimisation)
# ────────────────────────────────────────────────────────────────────────────
class RouterMLP(nn.Module):
    def __init__(self, d_in=INPUT_DIM, hidden=256, d_out=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, d_out)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

router = RouterMLP()
opt    = torch.optim.Adam(router.parameters(), lr=LR_ROUTER)
bce    = nn.BCEWithLogitsLoss()

# ────────────────────────────────────────────────────────────────────────────
# METRICS (pipeline accuracy)
# ────────────────────────────────────────────────────────────────────────────
pipe_acc = metrics.Accuracy()        # argmax strategy

# ────────────────────────────────────────────────────────────────────────────
# MAIN STREAM LOOP  (training + "online validation" in one pass)
# ────────────────────────────────────────────────────────────────────────────
t0 = time.time()
for i, (x_dict, y_true) in enumerate(stream, 1):
    # ---------- 1) EXPERT PREDICTIONS & multi-hot label -------------------
    correct_mask = np.zeros(NUM_CLASSES, dtype=np.float32)
    for cid, model in experts.items():
        p = model.predict_one(x_dict)  # binary {0,1}
        if p == 1 and y_true == cid:
            correct_mask[cid] = 1.0
        # update per-expert metric before learning (prequential)
        exp_metrics[cid].update(int(y_true == cid), p)

    # fallback: if no expert correct, still tag ground-truth
    if correct_mask.sum() == 0:
        correct_mask[y_true] = 1.0

    # ---------- 2) ROUTER FORWARD, METRIC, GRADIENT STEP ------------------
    x_vec   = torch.tensor(d2v(x_dict)).unsqueeze(0)
    target  = torch.tensor(correct_mask).unsqueeze(0)
    logits  = router(x_vec)
    # prediction used for pipeline decision (argmax of probs)
    pred_expert = int(torch.argmax(logits).item())
    pipe_acc.update(y_true, pred_expert)

    # router online update
    loss = bce(logits, target)
    opt.zero_grad(); loss.backward(); opt.step()

    # ---------- 3) EXPERTS ONLINE UPDATE ----------------------------------
    for cid, model in experts.items():
        model.learn_one(x_dict, int(y_true == cid))

    # ---------- 4) Console log -------------------------------------------
    if i % PRINT_EVERY == 0:
        print(f"[{i:>7}] PipeAcc={pipe_acc.get():.4f}  "
              f"AvgExpertAcc={np.mean([m.get() for m in exp_metrics.values()]):.4f}")

print("\n── DONE ─────────────────────────────────────────────")
print(f"Total time: {time.time()-t0:.1f}s")
print(f"Final pipeline accuracy (argmax): {pipe_acc.get():.4f}")
print("\nPer-expert accuracies (prequential):")
for cid in range(NUM_CLASSES):
    print(f"  Expert {cid}: {exp_metrics[cid].get():.4f}")