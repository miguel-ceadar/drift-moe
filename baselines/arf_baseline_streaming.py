# arf_stream_experiment.py  (streaming validation)
"""
Description: Train ARF online and validate in online (learn_one + predict_one) mode.
"""
from river import metrics, forest, datasets
import time, csv, os

# ────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────
TOTAL_SAMPLES = 1_000_000
TRAIN_RATIO   = 0.80
SEED_STREAM   = 112

RESULTS_DIR   = "./arf_results_streaming_10_experts"       # <── change as needed
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

half           = TOTAL_SAMPLES // 2
expert_block   = stream[:half]
router_block   = stream[half:]

exp_train_sz   = int(len(expert_block)  * TRAIN_RATIO)
rtr_train_sz   = int(len(router_block)  * TRAIN_RATIO)

exp_train, exp_val = expert_block[:exp_train_sz],  expert_block[exp_train_sz:]
rtr_train, rtr_val = router_block[:rtr_train_sz], router_block[rtr_train_sz:]

print("── SPLITS ───────────────────────────────────────────")
print(f" total samples         : {TOTAL_SAMPLES:,}")
print(f" expert  train / val   : {len(exp_train):,} / {len(exp_val):,}")
print(f" router  train / val   : {len(rtr_train):,} / {len(rtr_val):,}")

# Everything except rtr_val is used for initial training
train_data = exp_train + exp_val + rtr_train

# ────────────────────────────────────────────────────────────────────────────
#  INITIALISE MODEL & METRICS
# ────────────────────────────────────────────────────────────────────────────
arf = forest.ARFClassifier(n_models=10, seed=42)

preq_acc  = metrics.Accuracy();   preq_prec = metrics.Precision(); preq_rec = metrics.Recall()
val_acc   = metrics.Accuracy();   val_prec  = metrics.Precision();  val_rec = metrics.Recall()

preq_steps, preq_accs, preq_precs, preq_recs = [], [], [], []
val_steps,  val_accs,  val_precs,  val_recs  = [], [], [], []

# ────────────────────────────────────────────────────────────────────────────
#  TRAINING  (Prequential)
# ────────────────────────────────────────────────────────────────────────────
print("── TRAINING ARF (Prequential) ───────────────────────")
t0 = time.time()
for i, (x, y) in enumerate(train_data):
    yp = arf.predict_one(x)
    preq_acc.update(y, yp)
    preq_prec.update(y, yp)
    preq_rec.update(y, yp)
    arf.learn_one(x, y)

    if i % 100 == 0:
        print(f"[{i}] Train Acc: {preq_acc.get():.4f}")
train_time = time.time() - t0

# ────────────────────────────────────────────────────────────────────────────
#  VALIDATION  (Streaming / Prequential)
# ────────────────────────────────────────────────────────────────────────────
print("── VALIDATING ARF (Streaming) ───────────────────────")
t0 = time.time()
for i, (x, y) in enumerate(rtr_val):
    yp = arf.predict_one(x)
    val_acc.update(y, yp)
    val_prec.update(y, yp)
    val_rec.update(y, yp)
    arf.learn_one(x, y) # ← online update: so that the model can adapt to new data

    if i % 10 == 0:
        print(f"[VAL {i}] Acc: {val_acc.get():.4f}")
val_time = time.time() - t0

# ────────────────────────────────────────────────────────────────────────────
#  SAVE CSVs
# ────────────────────────────────────────────────────────────────────────────
def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header, *rows])

write_csv(
    os.path.join(RESULTS_DIR, "arf_final_metrics.csv"),
    ["Metric", "Value"],
    [
        ("Final Accuracy",    f"{val_acc.get():.4f}"),
        ("Final Precision",   f"{val_prec.get():.4f}"),
        ("Final Recall",      f"{val_rec.get():.4f}"),
        ("Training Time (s)", f"{train_time:.2f}"),
        ("Validation Time (s)",f"{val_time:.2f}")
    ]
)

print(f"\n✔ CSVs saved to: {os.path.abspath(RESULTS_DIR)}")