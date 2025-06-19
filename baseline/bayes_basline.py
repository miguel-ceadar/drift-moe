# nb_stream_experiment.py  —  Gaussian Naive Bayes baseline
from river import metrics, naive_bayes, datasets, stream
import numpy as np
import time, csv, os

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
TOTAL_SAMPLES = 1_000_000
TRAIN_RATIO   = 0.80
SEED_STREAM   = 112
PRINT_EVERY   = 100

RESULTS_DIR   = "./nb_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
#  DATA STREAM & SPLITS
# ────────────────────────────────────────────────────────────────────────────
raw_stream = list(
    datasets.synth.LEDDrift(
        seed=SEED_STREAM,
        noise_percentage=0.10,
        irrelevant_features=True,
        n_drift_features=7
    ).take(TOTAL_SAMPLES)
)

half            = TOTAL_SAMPLES // 2
expert_block    = raw_stream[:half]
router_block    = raw_stream[half:]

exp_train_sz    = int(len(expert_block)  * TRAIN_RATIO)
rtr_train_sz    = int(len(router_block)  * TRAIN_RATIO)

exp_train, exp_val = expert_block[:exp_train_sz],  expert_block[exp_train_sz:]
rtr_train, rtr_val = router_block[:rtr_train_sz],  router_block[rtr_train_sz:]

train_data = exp_train + exp_val + rtr_train  # all except final hold-out

print("── SPLITS ───────────────────────────────────────────")
print(f" total samples         : {TOTAL_SAMPLES:,}")
print(f" expert  train / val   : {len(exp_train):,} / {len(exp_val):,}")
print(f" router  train / val   : {len(rtr_train):,} / {len(rtr_val):,}")

# ────────────────────────────────────────────────────────────────────────────
#  MODEL & METRICS
# ────────────────────────────────────────────────────────────────────────────
nb_model = naive_bayes.GaussianNB()

train_acc = metrics.Accuracy(); train_prec = metrics.Precision(); train_rec = metrics.Recall()
val_acc   = metrics.Accuracy();  val_prec  = metrics.Precision();  val_rec = metrics.Recall()

print("\n── TRAINING (Prequential) ───────────────────────────")
print("Step,Accuracy,Precision,Recall")

t0 = time.time()
for i, (x_dict, y) in enumerate(train_data):
    # GaussianNB expects dict features; x_dict is already a dict from LEDDrift
    y_pred = nb_model.predict_one(x_dict)
    train_acc.update(y, y_pred)
    train_prec.update(y, y_pred)
    train_rec.update(y, y_pred)
    nb_model.learn_one(x_dict, y)

    if i % PRINT_EVERY == 0:
        print(f"{i},{train_acc.get()},{train_prec.get()},{train_rec.get()}")

train_time = time.time() - t0

print("\n── VALIDATION (Hold-out) ───────────────────────────")
print("Step,Accuracy,Precision,Recall")

t0 = time.time()
for i, (x_dict, y) in enumerate(rtr_val):
    y_pred = nb_model.predict_one(x_dict)
    val_acc.update(y, y_pred)
    val_prec.update(y, y_pred)
    val_rec.update(y, y_pred)
    nb_model.learn_one(x_dict, y) # ← online update: so that the model can adapt to new data

    if i % PRINT_EVERY == 0:
        print(f"{i},{val_acc.get()}")

val_time = time.time() - t0

# ────────────────────────────────────────────────────────────────────────────
#  WRITE FINAL CSV
# ────────────────────────────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "nb_final_metrics.csv"), "w", newline="") as f:
    csv.writer(f).writerows([
        ["Metric", "Value"],
        ["Final Accuracy",    f"{val_acc.get():.4f}"],
        ["Final Precision",   f"{val_prec.get():.4f}"],
        ["Final Recall",      f"{val_rec.get():.4f}"],
        ["Training Time (s)", f"{train_time:.2f}"],
        ["Validation Time (s)", f"{val_time:.2f}"]
    ])

print(f"\n✔ Final metrics saved to {os.path.abspath(RESULTS_DIR)}")