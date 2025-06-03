from river import metrics
from river import forest
from river.datasets import synth
import matplotlib.pyplot as plt
import time
import csv
import os

# ─────────────────────────────────────────────────────────────────────────────
#  TARGETED Mixture-of-Experts Pipeline (Your Core Logic + Key Improvements)
# ─────────────────────────────────────────────────────────────────────────────

# ───────── OPTIMIZED CONFIG ─────────────────────────────────────────────────
TOTAL_SAMPLES = 1000000
TRAIN_RATIO = 0.80
NUM_CLASSES = 10
INPUT_DIM = 24
BATCH_SIZE = 256           # Keep your original batch size
EPOCHS = 75                # More epochs for better convergence
LR = 2e-3                 # Slightly lower LR for stability
SEED_STREAM = 112

# ───────── STREAM & SPLIT (Your Original Logic) ─────────────────────────────
stream = list(
    synth.LEDDrift(
        seed=SEED_STREAM,
        noise_percentage=0.10,
        irrelevant_features=True,
        n_drift_features=7
    ).take(TOTAL_SAMPLES)
)

half = TOTAL_SAMPLES // 2
expert_block = stream[:half]
router_block = stream[half:]

exp_train_sz = int(len(expert_block) * TRAIN_RATIO)
rtr_train_sz = int(len(router_block) * TRAIN_RATIO)

exp_train, exp_val = expert_block[:exp_train_sz], expert_block[exp_train_sz:]
rtr_train, rtr_val = router_block[:rtr_train_sz], router_block[rtr_train_sz:]

print("── SPLITS ───────────────────────────────────────────")
print(f" total samples         : {TOTAL_SAMPLES:,}")
print(f" expert  train / val   : {len(exp_train):,} / {len(exp_val):,}")
print(f" router  train / val   : {len(rtr_train):,} / {len(rtr_val):,}")

# ───────── PREPARE DATA ─────────────────────────────────────────────────
# Combine exp_train, exp_val, and rtr_train for training the ARF model
train_data = exp_train + exp_val + rtr_train
print(f"Total training samples for ARF: {len(train_data):,}")

# ───────── INITIALIZE MODEL & METRICS ────────────────────────────────────
arf = forest.ARFClassifier(n_models=100, seed=42)
preq_metric = metrics.Accuracy()
preq_precision = metrics.Precision()
preq_recall = metrics.Recall()
preq_steps, preq_accuracies, preq_precisions, preq_recalls = [], [], [], []

start_time = time.time()

# ───────── TRAIN ARF MODEL (Prequential Evaluation) ───────────────────────
print("── TRAINING ARF (Prequential) ─────────────────────────")
for i, (x, y) in enumerate(train_data):
    # Prequential: Predict first, then train
    y_pred = arf.predict_one(x)
    preq_metric.update(y, y_pred)
    preq_precision.update(y, y_pred)
    preq_recall.update(y, y_pred)
    arf.learn_one(x, y)  # Train ARF
    if i % 100 == 0:
        preq_steps.append(i)
        preq_accuracies.append(preq_metric.get())
        preq_precisions.append(preq_precision.get())
        preq_recalls.append(preq_recall.get())
        print(f"[{i}] Prequential ARF Accuracy: {preq_metric.get():.2f}%")

train_time = time.time() - start_time
print(f"Training time: {train_time:.2f} seconds")

# ───────── SAVE TRAINING METRICS TO CSV ──────────────────────────────────
with open('arf_training_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Step', 'Accuracy', 'Precision', 'Recall'])
    for step, acc, prec, rec in zip(preq_steps, preq_accuracies, preq_precisions, preq_recalls):
        writer.writerow([step, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}"])

# ───────── VALIDATE ARF MODEL (Holdout on rtr_val) ───────────────────────
print("── VALIDATING ARF ─────────────────────────────────────")
val_metric = metrics.Accuracy()
val_precision = metrics.Precision()
val_recall = metrics.Recall()
val_steps, val_accuracies, val_precisions, val_recalls = [], [], [], []

val_start_time = time.time()

for i, (x, y) in enumerate(rtr_val):
    y_pred = arf.predict_one(x)  # Predict on validation set
    val_metric.update(y, y_pred)
    val_precision.update(y, y_pred)
    val_recall.update(y, y_pred)
    if i % 50 == 0:  # More frequent updates for smoother curve
        val_steps.append(i)
        val_accuracies.append(val_metric.get())
        val_precisions.append(val_precision.get())
        val_recalls.append(val_recall.get())
        print(f"[{i}] Validation ARF Accuracy: {val_metric.get():.2f}%")

val_time = time.time() - val_start_time
print(f"Validation time: {val_time:.2f} seconds")

# ───────── SAVE VALIDATION METRICS TO CSV ────────────────────────────────
with open('arf_validation_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Step', 'Accuracy', 'Precision', 'Recall'])
    for step, acc, prec, rec in zip(val_steps, val_accuracies, val_precisions, val_recalls):
        writer.writerow([step, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}"])

# ───────── PRINT AND SAVE OVERALL VALIDATION METRICS ─────────────────────
final_val_accuracy = val_metric.get()
final_val_precision = val_precision.get()
final_val_recall = val_recall.get()
print(f"\n✅ Final Accuracy: {final_val_accuracy:.2f}%")
print(f"✅ Final Precision: {final_val_precision:.2f}%")
print(f"✅ Final Recall: {final_val_recall:.2f}%")

with open('arf_final_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Final Accuracy', f"{final_val_accuracy:.4f}"])
    writer.writerow(['Final Precision', f"{final_val_precision:.4f}"])
    writer.writerow(['Final Recall', f"{final_val_recall:.4f}"])
    writer.writerow(['Training Time (seconds)', f"{train_time:.4f}"])
    writer.writerow(['Validation Time (seconds)', f"{val_time:.4f}"])

# ───────── PLOT AND SAVE TRAINING RESULTS ────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(preq_steps, preq_accuracies, label="ARF Accuracy (Prequential)")
plt.plot(preq_steps, preq_precisions, label="ARF Precision (Prequential)")
plt.plot(preq_steps, preq_recalls, label="ARF Recall (Prequential)")
plt.xlabel("Samples")
plt.ylabel("Metric Value")
plt.title("ARF Training Metrics Over Time")
plt.grid(True)
plt.legend()
plt.savefig('arf_training_plot.png')
plt.close()

# ───────── PLOT AND SAVE VALIDATION RESULTS ──────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(val_steps, val_accuracies, label="ARF Accuracy (rtr_val)")
plt.plot(val_steps, val_precisions, label="ARF Precision (rtr_val)")
plt.plot(val_steps, val_recalls, label="ARF Recall (rtr_val)")
plt.xlabel("Samples")
plt.ylabel("Metric Value")
plt.title("ARF Accuracy on rtr_val Over Time")
plt.grid(True)
plt.legend()
plt.savefig('arf_validation_plot.png')
plt.close()