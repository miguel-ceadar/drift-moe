# dt_eval_experiment.py  – Decision-Tree baseline (batch train, streaming eval)
from sklearn.tree import DecisionTreeClassifier
from river import metrics, datasets
import numpy as np, csv, os, time

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
TOTAL_SAMPLES = 1_000_000
TRAIN_RATIO   = 0.80
INPUT_DIM     = 24
SEED_STREAM   = 112
PRINT_EVERY   = 50

RESULTS_DIR   = "./dt_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# DATA STREAM  &  SPLITS
# ────────────────────────────────────────────────────────────────────────────
stream = list(
    datasets.synth.LEDDrift(
        seed=SEED_STREAM,
        noise_percentage=0.10,
        irrelevant_features=True,
        n_drift_features=7
    ).take(TOTAL_SAMPLES)
)

half            = TOTAL_SAMPLES // 2
expert_block    = stream[:half]
router_block    = stream[half:]

exp_train_sz    = int(len(expert_block) * TRAIN_RATIO)
rtr_train_sz    = int(len(router_block) * TRAIN_RATIO)

exp_train, exp_val = expert_block[:exp_train_sz], expert_block[exp_train_sz:]
rtr_train, rtr_val = router_block[:rtr_train_sz], router_block[rtr_train_sz:]

train_data = exp_train + exp_val + rtr_train  # everything except final hold-out

print("── SPLITS ───────────────────────────────────────────")
print(f" total samples       : {TOTAL_SAMPLES:,}")
print(f" train set size      : {len(train_data):,}")
print(f" validation (rtr_val): {len(rtr_val):,}")

# helper to convert x-dict → fixed-order vector
d2v = lambda d: np.fromiter((d[i] for i in range(INPUT_DIM)),
                            dtype=np.float32, count=INPUT_DIM)

# ────────────────────────────────────────────────────────────────────────────
# TRAIN  Decision-Tree  (batch)
# ────────────────────────────────────────────────────────────────────────────
X_train = np.stack([d2v(xd) for xd, _ in train_data])
y_train = np.array([y for _, y in train_data], dtype=int)

dt_model = DecisionTreeClassifier(random_state=42)
t0 = time.time()
dt_model.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Training time: {train_time:.2f}s")

# ────────────────────────────────────────────────────────────────────────────
# VALIDATION  (streaming / sequential predictions)
# ────────────────────────────────────────────────────────────────────────────
val_acc  = metrics.Accuracy()
val_prec = metrics.Precision()
val_rec  = metrics.Recall()
val_steps, val_accs, val_precs, val_recs = [], [], [], []

print("\n── VALIDATION ──────────────────────────────────────")
print("Step,Accuracy,Precision,Recall")

t0 = time.time()
for i, (x_dict, y) in enumerate(rtr_val):
    x_vec  = d2v(x_dict).reshape(1, -1)
    y_pred = int(dt_model.predict(x_vec)[0])

    val_acc.update(y, y_pred)
    val_prec.update(y, y_pred)
    val_rec.update(y, y_pred)

    if i % PRINT_EVERY == 0:
        val_steps.append(i)
        val_accs.append(val_acc.get())
        val_precs.append(val_prec.get())
        val_recs.append(val_rec.get())
        print(f"{i},{val_acc.get()},{val_prec.get()},{val_rec.get()}")

val_time = time.time() - t0
print(f"\nValidation time: {val_time:.2f}s")

# ────────────────────────────────────────────────────────────────────────────
# SAVE CSVs
# ────────────────────────────────────────────────────────────────────────────
def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header, *rows])

# step-by-step metrics
write_csv(
    os.path.join(RESULTS_DIR, "dt_validation_metrics.csv"),
    ["Step", "Accuracy", "Precision", "Recall"],
    zip(val_steps, val_accs, val_precs, val_recs)
)

# final summary
write_csv(
    os.path.join(RESULTS_DIR, "dt_final_metrics.csv"),
    ["Metric", "Value"],
    [
        ("Final Accuracy",    f"{val_acc.get():.4f}"),
        ("Final Precision",   f"{val_prec.get():.4f}"),
        ("Final Recall",      f"{val_rec.get():.4f}"),
        ("Training Time (s)", f"{train_time:.2f}"),
        ("Validation Time (s)", f"{val_time:.2f}")
    ]
)

print(f"\n✔ Metrics saved to {os.path.abspath(RESULTS_DIR)}")