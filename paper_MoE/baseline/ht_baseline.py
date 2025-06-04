# ht_stream_experiment.py
from river import metrics, tree, datasets
import matplotlib.pyplot as plt
import time, csv, os

# ────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────
TOTAL_SAMPLES = 1_000_000
TRAIN_RATIO    = 0.80
SEED_STREAM    = 112

RESULTS_DIR = "./ht_results"           # <── change to wherever you like
os.makedirs(RESULTS_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
#  DATA STREAM & SPLITS
# ────────────────────────────────────────────────────────────────────────────
stream = list(
    datasets.synth.LEDDrift(
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

exp_train, exp_val = expert_block[:exp_train_sz],  expert_block[exp_train_sz:]
rtr_train, rtr_val = router_block[:rtr_train_sz], router_block[rtr_train_sz:]

print("── SPLITS ───────────────────────────────────────────")
print(f" total samples         : {TOTAL_SAMPLES:,}")
print(f" expert  train / val   : {len(exp_train):,} / {len(exp_val):,}")
print(f" router  train / val   : {len(rtr_train):,} / {len(rtr_val):,}")

# Combine everything except rtr_val for training
train_data = exp_train + exp_val + rtr_train

# ────────────────────────────────────────────────────────────────────────────
#  INITIALISE MODEL & METRICS
# ────────────────────────────────────────────────────────────────────────────
ht = tree.HoeffdingTreeClassifier(grace_period=200)



preq_acc  = metrics.Accuracy()
preq_prec = metrics.Precision()
preq_rec  = metrics.Recall()
preq_steps, preq_accs, preq_precs, preq_recs = [], [], [], []

start_time = time.time()

# ────────────────────────────────────────────────────────────────────────────
#  TRAINING  (Prequential)
# ────────────────────────────────────────────────────────────────────────────
print("── TRAINING HT (Prequential) ───────────────────────")
for i, (x, y) in enumerate(train_data):
    y_pred = ht.predict_one(x)
    preq_acc.update(y, y_pred)
    preq_prec.update(y, y_pred)
    preq_rec.update(y, y_pred)
    ht.learn_one(x, y)

    if i % 100 == 0:
        preq_steps.append(i)
        preq_accs.append(preq_acc.get())
        preq_precs.append(preq_prec.get())
        preq_recs.append(preq_rec.get())
        print(f"[{i}] Accuracy: {preq_acc.get():.4f}")

train_time = time.time() - start_time
print(f"Training time: {train_time:.2f}s")

# ────────────────────────────────────────────────────────────────────────────
#  VALIDATION  (Hold-out on rtr_val)
# ────────────────────────────────────────────────────────────────────────────
val_acc  = metrics.Accuracy()
val_prec = metrics.Precision()
val_rec  = metrics.Recall()
val_steps, val_accs, val_precs, val_recs = [], [], [], []

val_start = time.time()
for i, (x, y) in enumerate(rtr_val):
    y_pred = ht.predict_one(x)
    val_acc.update(y, y_pred)
    val_prec.update(y, y_pred)
    val_rec.update(y, y_pred)

    if i % 50 == 0:
        val_steps.append(i)
        val_accs.append(val_acc.get())
        val_precs.append(val_prec.get())
        val_recs.append(val_rec.get())
        print(f"[VAL {i}] Accuracy: {val_acc.get():.4f}")

val_time = time.time() - val_start
print(f"Validation time: {val_time:.2f}s")

# ────────────────────────────────────────────────────────────────────────────
#  SAVE CSV METRICS
# ────────────────────────────────────────────────────────────────────────────
def to_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

to_csv(
    os.path.join(RESULTS_DIR, "ht_training_metrics.csv"),
    ["Step", "Accuracy", "Precision", "Recall"],
    zip(preq_steps, preq_accs, preq_precs, preq_recs)
)

to_csv(
    os.path.join(RESULTS_DIR, "ht_validation_metrics.csv"),
    ["Step", "Accuracy", "Precision", "Recall"],
    zip(val_steps, val_accs, val_precs, val_recs)
)

to_csv(
    os.path.join(RESULTS_DIR, "ht_final_metrics.csv"),
    ["Metric", "Value"],
    [
        ("Final Accuracy",   f"{val_acc.get():.4f}"),
        ("Final Precision",  f"{val_prec.get():.4f}"),
        ("Final Recall",     f"{val_rec.get():.4f}"),
        ("Training Time (s)",f"{train_time:.2f}"),
        ("Validation Time (s)",f"{val_time:.2f}")
    ]
)

# ────────────────────────────────────────────────────────────────────────────
#  PLOTS
# ────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10,6))
plt.plot(preq_steps, preq_accs, label="Accuracy")
plt.plot(preq_steps, preq_precs, label="Precision")
plt.plot(preq_steps, preq_recs, label="Recall")
plt.xlabel("Samples"); plt.ylabel("Metric")
plt.title("Hoeffding Tree – Training (Prequential)")
plt.grid(True); plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "ht_training_plot.png"))
plt.close()

plt.figure(figsize=(10,6))
plt.plot(val_steps, val_accs, label="Accuracy")
plt.plot(val_steps, val_precs, label="Precision")
plt.plot(val_steps, val_recs, label="Recall")
plt.xlabel("Samples"); plt.ylabel("Metric")
plt.title("Hoeffding Tree – Validation on rtr_val")
plt.grid(True); plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "ht_validation_plot.png"))
plt.close()

print(f"\n✔ All results saved to: {os.path.abspath(RESULTS_DIR)}")