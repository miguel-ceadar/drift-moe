#!/usr/bin/env python
# coding: utf-8

# In[127]:


# ────────────────────────────────────────────────────────────────────────────
#  Dependencies
# ────────────────────────────────────────────────────────────────────────────

from river import tree, naive_bayes
from river.datasets import synth
import matplotlib.pyplot as plt
from river import metrics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict
import numpy as np
import random

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from capymoa.stream.generator import LEDGeneratorDrift, LEDGenerator
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
# ────────────────────────────────────────────────────────────────────────────
#  CONFIG - same as pipeline 2
# ────────────────────────────────────────────────────────────────────────────
TOTAL_SAMPLES   = 1_000_000          # change for quick tests
TRAIN_RATIO     = 0.80
NUM_CLASSES     = 10
INPUT_DIM       = 24                 # 7 relevant + 17 irrelevant
BATCH           = 256
EPOCHS          = 75
LR              = 2e-3
SEED_STREAM     = 112
SEED_TORCH      = 42
torch.manual_seed(SEED_TORCH)
random.seed(SEED_TORCH)


# In[128]:


from capymoa.stream import MOAStream
from moa.streams import ConceptDriftStream
cli = "-s (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 1)   -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 3) -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 5)  -d (generators.LEDGeneratorDrift -d 7) -w 50000 -p 250000 ) -w 50000 -p 250000 ) -w 50000 -p 250000)"
led_g_stream = MOAStream(moa_stream=ConceptDriftStream(), CLI=cli)


# In[129]:


led_g_stream.restart()


# In[131]:


schema = led_g_stream.get_schema()
schema


# In[132]:


# ────────────────────────────────────────────────────────────────────────────
#  LOAD STREAM (River)
# ────────────────────────────────────────────────────────────────────────────
"""
stream = list(
    synth.LEDDrift(
        seed                = SEED_STREAM,
        noise_percentage    = 0.10,
        irrelevant_features = True,
        n_drift_features    = 7
    ).take(TOTAL_SAMPLES)
)

TOTAL_SAMPLES = 1_000_000          # match the ARF paper
stream = list(led_g(seed=42))[:TOTAL_SAMPLES]

print(f"Loaded LED-g ARF stream ➜ {len(stream):,} samples")
# helper: dict→24-float vector
d2v = lambda d: np.fromiter(d.values(), dtype=np.float32, count=INPUT_DIM)
"""


# In[36]:


BURN_IN     = int(0.1 * TOTAL_SAMPLES)         # number of initial samples used only for clustering
MAX_K       = 15             # search k = 1 … MAX_K
SAMPLE_FRAC = 0.05           # sample 25 % of burn-in for PCA + GMM/Kmeans


# 3-A) Carve out burn-in block
stream        = list(stream_list)      # make it slice-able
TOTAL         = len(stream_list)
assert BURN_IN < TOTAL, "BURN_IN must be less than total stream length"
burn_in_block = stream[:BURN_IN]
rest_stream   = stream[BURN_IN:]  # will be used in Stage 1+2

print(f"Total samples: {TOTAL:,}  |  Burn-in: {len(burn_in_block):,}  |  Rest: {len(rest_stream):,}")

# 3-B) Draw an unsupervised sample (10% of burn-in) for clustering
sample_len    = int(SAMPLE_FRAC * len(burn_in_block))
indices       = random.sample(range(len(burn_in_block)), sample_len)
cluster_block = [burn_in_block[i] for i in indices]

# 3-C) Embed → Standardize → PCA → Whiten
#   (Assumes you have a function `d2v(x_dict)` that returns a 1×INPUT_DIM numpy array)
INPUT_DIM = 24  # adjust if your d2v output dimension is different

X = np.stack([d2v(x_dict) for x_dict, _ in cluster_block])   # (sample_len × INPUT_DIM)
scaler = StandardScaler().fit(X)
X_std  = scaler.transform(X)
pca    = PCA(whiten=True).fit(X_std)
X_wht  = pca.transform(X_std)                                 # (sample_len × INPUT_DIM)


# In[39]:


# 3-D) Fit GMMs for k in 1 … MAX_K, pick best by BIC
bic_scores = []
gmms       = []

for k in range(1, MAX_K + 1):
    gmm_candidate = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=SEED_TORCH
    ).fit(X_wht)
    bic_scores.append(gmm_candidate.bic(X_wht))
    gmms.append(gmm_candidate)

best_k_index = int(np.argmin(bic_scores))  # index in [0 … MAX_K-1]
n_experts    = best_k_index + 1            # because k started at 1
gmm          = gmms[best_k_index]          # optional, if you want cluster IDs later

print(f"Stage 0 complete → selected n_experts = {n_experts} (via BIC)")


# In[38]:


# 0-D) K-means sweep  (k = 2 … MAX_K)
best_k          = 1            # fallback if all silhouettes invalid
best_score      = -1.0
kmeans_models   = {}

for k in range(2, MAX_K + 1):
    km = KMeans(n_clusters=k, random_state=SEED_TORCH, n_init="auto").fit(X_wht)
    kmeans_models[k] = km
    try:
        score = silhouette_score(X_wht, km.labels_)
    except ValueError:          # rare case: silhouette undefined
        score = -1
    print(f"k={k:2d}  |  silhouette={score:.4f}")
    if score > best_score:
        best_score, best_k = score, k

n_experts = best_k
kmeans    = kmeans_models[best_k]

print(f"Stage 0 complete → selected n_experts = {n_experts} (highest silhouette = {best_score:.4f})")


# In[154]:


# ──────────────────────────────────────────────────────────────
# 1.  Hyper-params & boiler-plate
# ──────────────────────────────────────────────────────────────

from capymoa.classifier import HoeffdingTree


n_experts = 15

TOP_K         = 3            # update the K heaviest-weighted experts
PRINT_EVERY   = 10_000
CLASSES       = list(range(NUM_CLASSES))

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

# ──────────────────────────────────────────────────────────────
# 2.  Initialise experts and router
# ──────────────────────────────────────────────────────────────
experts = {i: HoeffdingTree(schema=schema, grace_period=50, confidence=1e-07, binary_split=False, stop_mem_management=False) for i in range(n_experts)}

class RouterMLP(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, h=256, out_dim=n_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Linear(h // 2, out_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

router = RouterMLP()
opt    = torch.optim.Adam(router.parameters(), lr=LR)
nll    = nn.NLLLoss(reduction="mean")

pipeline_acc = metrics.Accuracy()
running_loss = 0.0

# ──────────────────────────────────────────────────────────────
# 3.  Online joint-training loop 
# ──────────────────────────────────────────────────────────────
router.train()
micro_X, micro_y = [], []

for t in range(TOTAL_SAMPLES):
    # 3-A  Embed sample

    instance = stream.next_instance()
    x_vec = instance.x
    y_true = instance.y_index
    x_t   = to_tensor(x_vec).unsqueeze(0)         # 1×24

    # 3-B  Router forward
    logits  = router(x_t) # 1×n_experts
    #tau0, tau_min, decay_steps = 2.0, 0.7, 80_000   
    #tau = max(tau_min, tau0 * (1 - t / decay_steps)) # linear-cosine also works
    #weights = torch.softmax(logits / tau, dim=1)     # replaces previous softmax                       
    weights = torch.softmax(logits, dim=1)        # 1×n_experts

    # 3-C  Gather experts’ probability vectors
    exp_probs = []
    for e in experts.values():
        p_list = e.predict_proba(instance) or [1/NUM_CLASSES for c in CLASSES]
        if p_list is None:                            # brand-new leaf
            padded_p_list = [1 / NUM_CLASSES] * NUM_CLASSES  # uniform prior
        elif len(p_list) < NUM_CLASSES:               # seen some classes
            
            padded_p_list = list(p_list) + [0.0] * (NUM_CLASSES - len(list(p_list)))
        else:                                      # already full length
            padded_p_list = list(p_list)
        exp_probs.append(padded_p_list)
    exp_probs = torch.tensor(exp_probs)           # n_experts × C

    mix_prob = torch.mm(weights, exp_probs) + 1e-9
    log_mix  = (mix_prob / mix_prob.sum()).log()  # 1×C log-probs

    # 3-D  Accumulate mini-batch for router update
    micro_X.append(log_mix)
    micro_y.append(y_true)
    if len(micro_X) == BATCH:
        batch_X = torch.cat(micro_X, dim=0)       # B×C
        batch_y = torch.tensor(micro_y)
        loss = nll(batch_X, batch_y)
        opt.zero_grad(); loss.backward(); opt.step()
        running_loss += loss.item() * BATCH
        micro_X.clear(); micro_y.clear()

    # 3-E  Top-K expert updates
    with torch.no_grad():
        topk_ids = torch.topk(weights, k=TOP_K, dim=1).indices.squeeze(0)
    for eid in topk_ids.tolist():
        experts[eid].train(instance)

    # 3-F  Running metrics
    y_hat = CLASSES[int(torch.argmax(mix_prob))]
    pipeline_acc.update(y_true, y_hat)

    if t % PRINT_EVERY == 0:
        avg_ce = running_loss / max(1, (t // BATCH))
        print(f"[{t:,} samples]  router CE: {avg_ce:.4f}   "
              f"pipeline acc: {pipeline_acc.get():.4f}")
        running_loss = 0.0

print("🏁 train-window accuracy:", pipeline_acc.get())

"""
# ──────────────────────────────────────────────────────────────
# 4.  Hold-out evaluation  (last 10 %)
# ──────────────────────────────────────────────────────────────
router.eval()
hold_acc = metrics.Accuracy()

with torch.no_grad():
    for x_dict, y_true in hold_stream:
        x_vec = d2v(x_dict)
        logits  = router(to_tensor(x_vec).unsqueeze(0))
        weights = torch.softmax(logits, dim=1)
        exp_probs = []
        for e in experts.values():
            pdict = e.predict_proba_one(x_dict) or {c: 1/NUM_CLASSES for c in CLASSES}
            exp_probs.append([pdict.get(c, 0.0) for c in CLASSES])
        exp_probs = torch.tensor(exp_probs)
        mix_prob  = torch.mm(weights, exp_probs)
        y_hat     = CLASSES[int(torch.argmax(mix_prob))]
        hold_acc.update(y_true, y_hat)

print("🏁 hold-out (10 %) accuracy:", hold_acc.get())
"""


# In[ ]:





# In[ ]:




