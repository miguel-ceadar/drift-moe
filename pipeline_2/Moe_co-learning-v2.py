import torch, torch.nn as nn
import numpy as np
import warnings, time
from river import stats, utils
from river import tree, metrics
from river.datasets import synth
warnings.filterwarnings("ignore")
# dt_stream_experiment.py – Online Mixture-of-Experts with Hoeffding Trees
import torch, torch.nn as nn
import numpy as np
# from river import tree, metrics, synth, utils
import warnings, time
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
TOTAL_SAMPLES = 1_000_000
TRAIN_RATIO = 0.80
NUM_CLASSES = 10
INPUT_DIM = 24
BATCH_SIZE = 1  # Online learning
EPOCHS = 1      # Single pass through data
LR = 1e-4        # Smaller LR for online stability
SEED_STREAM = 112
torch.manual_seed(42)

# ────────────────────────────────────────────────────────────────────────────
# DATA STREAM & SPLITS
# ────────────────────────────────────────────────────────────────────────────
stream = list(synth.LEDDrift(
    seed=SEED_STREAM,
    noise_percentage=0.10,
    irrelevant_features=True,
    n_drift_features=7
).take(TOTAL_SAMPLES))

half = TOTAL_SAMPLES // 2
expert_block = stream[:half]  # Initial expert training
router_block = stream[half:]  # Joint training + validation

rtr_train_sz = int(len(router_block) * TRAIN_RATIO)
rtr_online_train = router_block[:rtr_train_sz]  # Joint training
rtr_online_val = router_block[rtr_train_sz:]    # Incremental validation

print("── ONLINE SPLITS ────────────────────────────────────")
print(f" Expert pre-train samples : {len(expert_block):,}")
print(f" Joint train samples      : {len(rtr_online_train):,}")
print(f" Incremental val samples  : {len(rtr_online_val):,}")

# Feature dict to vector converter
d2v = lambda d: np.fromiter(d.values(), dtype=np.float32, count=INPUT_DIM)

# ────────────────────────────────────────────────────────────────────────────
# 1) INITIALIZE EXPERTS - Online Hoeffding Trees
# ────────────────────────────────────────────────────────────────────────────
experts = {cid: tree.HoeffdingTreeClassifier() for cid in range(NUM_CLASSES)}
exp_metrics = {cid: metrics.Accuracy() for cid in range(NUM_CLASSES)}

print("\n Pre-training experts on initial block...")
start = time.time()
for i, (x, y_true) in enumerate(expert_block):
    for cid in range(NUM_CLASSES):
        label = 1 if y_true == cid else 0
        experts[cid].learn_one(x, label)
    
    # Periodically report progress
    if i > 0 and i % 100_000 == 0:
        elapsed = time.time() - start
        print(f" Processed {i:,} samples ({elapsed:.1f}s)")

print(f" Expert pre-training completed in {time.time()-start:.1f} seconds")

# ────────────────────────────────────────────────────────────────────────────
# 2) ROUTER MODEL - Neural Network
# ────────────────────────────────────────────────────────────────────────────
class RouterMLP(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, h=128, out=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, out))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x): return self.net(x)

router = RouterMLP()
opt = torch.optim.Adam(router.parameters(), lr=LR, weight_decay=1e-5)
bce = nn.BCEWithLogitsLoss()
router.train()

# ────────────────────────────────────────────────────────────────────────────
# 3) JOINT ONLINE TRAINING PHASE
# ────────────────────────────────────────────────────────────────────────────
print("\n Starting joint training of experts and router...")
joint_metrics = {
    'exp_loss': utils.Rolling(mean=0.1),
    'router_loss': utils.Rolling(mean=0.1),
    'exp_acc': utils.Rolling(mean=0.1),
    'time': utils.Rolling(mean=0.1)
}

start = time.time()
for idx, (x_dict, y_true) in enumerate(rtr_online_train):
    # 3A) Update all experts
    exp_loss = 0
    exp_correct = 0
    for cid in range(NUM_CLASSES):
        # Get true binary label for this expert
        bin_label = 1 if y_true == cid else 0
        
        # Make prediction before learning (for metrics)
        pred_before = experts[cid].predict_proba_one(x_dict)
        before_acc = 1 if pred_before.get(bin_label, 0) > 0.5 else 0
        
        # Update expert with new sample
        experts[cid].learn_one(x_dict, bin_label)
        
        # Make prediction after learning
        pred_after = experts[cid].predict_proba_one(x_dict)
        after_acc = 1 if pred_after.get(bin_label, 0) > 0.5 else 0
        
        # Track expert performance
        exp_correct += after_acc
        exp_loss += 0 if after_acc else 1  # 0/1 loss

    # 3B) Update router
    x_vec = torch.tensor(d2v(x_dict), dtype=torch.float32).unsqueeze(0)
    target = torch.zeros(1, NUM_CLASSES, dtype=torch.float32)
    target[0, y_true] = 1.0  # One-hot true label

    # Forward + backward
    logits = router(x_vec)
    loss = bce(logits, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    # 3C) Track metrics
    joint_metrics['exp_loss'].update(exp_loss / NUM_CLASSES)
    joint_metrics['router_loss'].update(loss.item())
    joint_metrics['exp_acc'].update(exp_correct / NUM_CLASSES)
    joint_metrics['time'].update(time.time() - start)
    
    # Periodic reporting
    if idx > 0 and idx % 10_000 == 0:
        avg_time = joint_metrics['time'].get() * 1000
        print(f" Sample {idx:,} | "
              f"Exp Loss: {joint_metrics['exp_loss'].get():.4f} | "
              f"Router Loss: {joint_metrics['router_loss'].get():.4f} | "
              f"Exp Acc: {joint_metrics['exp_acc'].get():.4f} | "
              f"Time/sample: {avg_time:.2f}ms")

# ────────────────────────────────────────────────────────────────────────────
# 4) INCREMENTAL VALIDATION
# ────────────────────────────────────────────────────────────────────────────
print("\n Starting incremental validation...")
val_metrics = {
    'argmax': metrics.Accuracy(),
    'top2': metrics.Accuracy(),
    'threshold': metrics.Accuracy(),
    'exp_acc': metrics.Accuracy()
}

router.eval()
val_start = time.time()

for i, (x_dict, y_true) in enumerate(rtr_online_val):
    # Get expert predictions
    expert_preds = np.zeros(NUM_CLASSES)
    for cid in range(NUM_CLASSES):
        proba = experts[cid].predict_proba_one(x_dict)
        expert_preds[cid] = proba.get(1, 0.0)  # Probability for positive class
    
    # Get router predictions
    x_vec = torch.tensor(d2v(x_dict), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        router_probs = torch.sigmoid(router(x_vec)).squeeze().numpy()
    
    # Combined prediction strategies
    argmax_pred = np.argmax(router_probs)
    top2_pred = np.argmax(router_probs) if max(router_probs) > 0.4 else np.argsort(router_probs)[-2]
    threshold_pred = np.argmax(router_probs)
    
    # Update metrics
    val_metrics['argmax'].update(y_true, argmax_pred)
    val_metrics['top2'].update(y_true, top2_pred)
    val_metrics['threshold'].update(y_true, threshold_pred)
    
    # Expert accuracy (using true expert)
    exp_acc = 1 if expert_preds[y_true] > 0.5 else 0
    val_metrics['exp_acc'].update(exp_acc)
    
    # Periodic reporting
    if i > 0 and i % 20_000 == 0:
        elapsed = time.time() - val_start
        print(f" Validated {i:,} samples ({elapsed:.1f}s)")

# ────────────────────────────────────────────────────────────────────────────
# RESULTS
# ────────────────────────────────────────────────────────────────────────────
print("\n── FINAL RESULTS ───────────────────────────────────")
print(f"Argmax accuracy        : {val_metrics['argmax'].get():.4f}")
print(f"Top-2 accuracy         : {val_metrics['top2'].get():.4f}")
print(f"Threshold accuracy     : {val_metrics['threshold'].get():.4f}")
print(f"Expert accuracy        : {val_metrics['exp_acc'].get():.4f}")

best_acc = max(val_metrics['argmax'].get(), 
              val_metrics['top2'].get(), 
              val_metrics['threshold'].get())
print(f"\n BEST PIPELINE ACCURACY: {best_acc:.4f}")