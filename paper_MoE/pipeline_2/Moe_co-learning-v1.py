# online_moe_system.py – Online Mixture-of-Experts with Incremental Training

"""
Online learning system where router and experts train simultaneously
on streaming data with concept drift adaptation.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque, defaultdict
from sklearn.tree import DecisionTreeClassifier
from river import tree, metrics
from river.datasets import synth
import warnings
import time
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
TOTAL_SAMPLES = 1_000_000
NUM_CLASSES = 10
INPUT_DIM = 24
BATCH_SIZE = 32  # Smaller batches for online learning
LR = 1e-3
SEED_STREAM = 112
EXPERT_UPDATE_FREQ = 100  # Update experts every N samples
ROUTER_UPDATE_FREQ = 50   # Update router every N samples
VALIDATION_FREQ = 500     # Validate every N samples
DRIFT_DETECTION_WINDOW = 1000  # Window for drift detection
PERFORMANCE_WINDOW = 200  # Window for performance tracking

torch.manual_seed(42)

class OnlineExpert:
    """Online Decision Tree Expert using River"""
    def __init__(self, class_id: int, max_depth: int = 10):
        self.class_id = class_id
        self.model = tree.HoeffdingTreeClassifier(
            max_depth=max_depth,
            split_criterion='gini',
            delta=1e-7,
            tau=0.05,
            leaf_prediction='mc'
        )
        self.samples_seen = 0
        self.recent_accuracy = metrics.Accuracy()
        self.performance_history = deque(maxlen=PERFORMANCE_WINDOW)
        
    def update(self, x_dict: dict, y_true: int):
        """Update expert with new sample"""
        binary_label = 1 if y_true == self.class_id else 0
        self.model.learn_one(x_dict, binary_label)
        self.samples_seen += 1
        
    def predict_proba(self, x_dict: dict) -> float:
        """Get prediction probability for this expert's class"""
        try:
            pred_dict = self.model.predict_proba_one(x_dict)
            return pred_dict.get(1, 0.0)  # Probability of positive class
        except:
            return 0.5  # Default probability if model not ready
    
    def predict(self, x_dict: dict) -> int:
        """Binary prediction for this expert"""
        return int(self.predict_proba(x_dict) > 0.5)
    
    def evaluate_and_update_performance(self, x_dict: dict, y_true: int):
        """Evaluate prediction and update performance metrics"""
        pred = self.predict(x_dict)
        binary_label = 1 if y_true == self.class_id else 0
        self.recent_accuracy.update(binary_label, pred)
        self.performance_history.append(binary_label == pred)

class OnlineRouter(nn.Module):
    """Online Neural Network Router"""
    def __init__(self, in_dim: int = INPUT_DIM, hidden: int = 128, out_dim: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        self.criterion = nn.BCEWithLogitsLoss()
        self.samples_seen = 0
        
    def forward(self, x):
        return self.net(x)
    
    def update(self, x_dict: dict, y_true: int):
        """Update router with new sample"""
        # Convert dict to tensor
        x_vec = torch.tensor(
            np.fromiter(x_dict.values(), dtype=np.float32, count=INPUT_DIM)
        ).unsqueeze(0)
        
        # Create multi-hot target
        target = torch.zeros(1, NUM_CLASSES)
        target[0, y_true] = 1.0
        
        # Forward pass and update
        self.train()
        self.optimizer.zero_grad()
        logits = self.forward(x_vec)
        loss = self.criterion(logits, target)
        loss.backward()
        self.optimizer.step()
        
        self.samples_seen += 1
        return loss.item()

class DriftDetector:
    """Simple drift detection based on performance degradation"""
    def __init__(self, window_size: int = DRIFT_DETECTION_WINDOW, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.performance_history = deque(maxlen=window_size)
        self.baseline_performance = None
        
    def add_sample(self, is_correct: bool):
        """Add new performance sample"""
        self.performance_history.append(is_correct)
        
        # Set baseline after collecting enough samples
        if len(self.performance_history) == self.window_size and self.baseline_performance is None:
            self.baseline_performance = np.mean(self.performance_history)
    
    def detect_drift(self) -> bool:
        """Detect if concept drift occurred"""
        if len(self.performance_history) < self.window_size or self.baseline_performance is None:
            return False
        
        current_performance = np.mean(list(self.performance_history)[-self.window_size//2:])
        return (self.baseline_performance - current_performance) > self.threshold

class OnlineMixtureOfExperts:
    """Complete Online Mixture of Experts System"""
    def __init__(self):
        self.experts = {i: OnlineExpert(i) for i in range(NUM_CLASSES)}
        self.router = OnlineRouter()
        self.drift_detector = DriftDetector()
        
        # Performance tracking
        self.overall_accuracy = metrics.Accuracy()
        self.router_accuracy = metrics.Accuracy()
        self.expert_predictions = defaultdict(list)
        
        # Counters
        self.samples_processed = 0
        self.drift_detections = 0
        
        # Validation metrics
        self.validation_history = []
        
    def d2v(self, d: dict) -> np.ndarray:
        """Convert feature dict to vector"""
        return np.fromiter(d.values(), dtype=np.float32, count=INPUT_DIM)
    
    def predict(self, x_dict: dict) -> Tuple[int, Dict]:
        """Make prediction using the full pipeline"""
        # Get router probabilities
        self.router.eval()
        with torch.no_grad():
            x_vec = torch.tensor(self.d2v(x_dict)).unsqueeze(0)
            logits = self.router(x_vec).squeeze(0)
            router_probs = torch.sigmoid(logits).numpy()
        
        # Get expert predictions
        expert_probs = np.array([
            self.experts[i].predict_proba(x_dict) for i in range(NUM_CLASSES)
        ])
        
        # Combine router and expert predictions (weighted average)
        combined_probs = 0.7 * router_probs + 0.3 * expert_probs
        
        # Select best expert
        best_expert = int(np.argmax(combined_probs))
        
        return best_expert, {
            'router_probs': router_probs,
            'expert_probs': expert_probs,
            'combined_probs': combined_probs
        }
    
    def update(self, x_dict: dict, y_true: int):
        """Update the entire system with new sample"""
        # Make prediction first (for evaluation)
        prediction, info = self.predict(x_dict)
        
        # Update performance metrics
        is_correct = (prediction == y_true)
        self.overall_accuracy.update(y_true, prediction)
        self.drift_detector.add_sample(is_correct)
        
        # Update all experts in parallel
        for expert_id, expert in self.experts.items():
            expert.update(x_dict, y_true)
            expert.evaluate_and_update_performance(x_dict, y_true)
        
        # Update router
        if self.samples_processed % ROUTER_UPDATE_FREQ == 0:
            router_loss = self.router.update(x_dict, y_true)
        
        # Check for drift
        if self.drift_detector.detect_drift():
            self.handle_drift()
        
        self.samples_processed += 1
        
        return {
            'prediction': prediction,
            'is_correct': is_correct,
            'drift_detected': self.drift_detector.detect_drift()
        }
    
    def handle_drift(self):
        """Handle detected concept drift"""
        print(f"  🚨 Drift detected at sample {self.samples_processed}")
        self.drift_detections += 1
        
        # Reset drift detector baseline
        self.drift_detector.baseline_performance = None
        
        # Optionally: Reset or adapt models more aggressively
        # For now, just continue with current adaptive learning
    
    def validate_incremental(self, validation_stream: List[Tuple]) -> Dict:
        """Perform incremental validation"""
        val_accuracy = metrics.Accuracy()
        val_samples = min(len(validation_stream), 1000)  # Limit validation size
        
        for x_dict, y_true in validation_stream[:val_samples]:
            pred, _ = self.predict(x_dict)
            val_accuracy.update(y_true, pred)
        
        val_result = {
            'accuracy': val_accuracy.get(),
            'samples_used': val_samples,
            'timestamp': self.samples_processed
        }
        
        self.validation_history.append(val_result)
        return val_result
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'samples_processed': self.samples_processed,
            'overall_accuracy': self.overall_accuracy.get(),
            'drift_detections': self.drift_detections,
            'expert_accuracies': {
                i: expert.recent_accuracy.get() 
                for i, expert in self.experts.items()
            },
            'router_samples': self.router.samples_seen,
            'validation_history_length': len(self.validation_history)
        }

# ────────────────────────────────────────────────────────────────────────────
# MAIN ONLINE LEARNING EXPERIMENT
# ────────────────────────────────────────────────────────────────────────────

def run_online_experiment():
    """Run the complete online learning experiment"""
    print("🚀 Starting Online Mixture-of-Experts Experiment")
    print(f"📊 Total samples: {TOTAL_SAMPLES:,}")
    print(f"🔄 Expert updates every: {EXPERT_UPDATE_FREQ} samples")
    print(f"🎯 Router updates every: {ROUTER_UPDATE_FREQ} samples")
    print(f"✅ Validation every: {VALIDATION_FREQ} samples")
    
    # Initialize system
    moe_system = OnlineMixtureOfExperts()
    
    # Create data stream with concept drift
    stream = synth.LEDDrift(
        seed=SEED_STREAM,
        noise_percentage=0.10,
        irrelevant_features=True,
        n_drift_features=7
    )
    
    # Create validation stream (separate from training)
    val_stream = list(synth.LEDDrift(
        seed=SEED_STREAM + 1,
        noise_percentage=0.05,
        irrelevant_features=True,
        n_drift_features=5
    ).take(5000))
    
    start_time = time.time()
    
    # Process stream incrementally
    for i, (x_dict, y_true) in enumerate(stream.take(TOTAL_SAMPLES)):
        # Update system
        result = moe_system.update(x_dict, y_true)
        
        # Periodic reporting
        if (i + 1) % VALIDATION_FREQ == 0:
            # Incremental validation
            val_result = moe_system.validate_incremental(val_stream)
            status = moe_system.get_system_status()
            
            elapsed = time.time() - start_time
            
            print(f"\n📈 Sample {i+1:,}/{TOTAL_SAMPLES:,} ({elapsed:.1f}s)")
            print(f"   Training Accuracy: {status['overall_accuracy']:.4f}")
            print(f"   Validation Accuracy: {val_result['accuracy']:.4f}")
            print(f"   Drift Detections: {status['drift_detections']}")
            
            # Show expert performance
            expert_accs = [f"E{j}:{acc:.3f}" for j, acc in status['expert_accuracies'].items()]
            print(f"   Expert Accs: {' '.join(expert_accs[:5])}...")
    
    # Final results
    final_status = moe_system.get_system_status()
    final_validation = moe_system.validate_incremental(val_stream)
    
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS")
    print("="*60)
    print(f"📊 Samples Processed: {final_status['samples_processed']:,}")
    print(f"🎯 Final Training Accuracy: {final_status['overall_accuracy']:.4f}")
    print(f"✅ Final Validation Accuracy: {final_validation['accuracy']:.4f}")
    print(f"🚨 Total Drift Detections: {final_status['drift_detections']}")
    print(f"⏱️ Total Time: {time.time() - start_time:.1f}s")
    
    print("\n🔍 Expert Performance:")
    for i, acc in final_status['expert_accuracies'].items():
        print(f"   Expert {i}: {acc:.4f}")
    
    return moe_system, final_status

if __name__ == "__main__":
    system, results = run_online_experiment()