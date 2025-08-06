# DriftMoE: A Mixture of Experts Approach to Handle Concept Drifts

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.00024-b31b1b.svg)](https://arxiv.org/abs/2507.18464)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/miguel-ceadar/drift-moe)

This repository contains the code and experiments for the paper:

**DriftMoE: A Mixture of Experts Approach to Handle Concept Drifts**
*(Aspis\*, Cajas Ordóñez\*, Suárez-Cetrulo, Simón Carbajo, 2025)*

---

## 🧪 Overview

**DriftMoE** introduces a fully streaming Mixture-of-Experts (MoE) architecture to tackle concept drift in non-stationary data streams. It features a lightweight neural router and a pool of incremental decision tree experts trained in an online, co-adaptive fashion. The framework is evaluated on multiple synthetic and real-world benchmarks and shown to outperform traditional adaptive ensembles with fewer learners and reduced computation.

---

## 📁 Repository Structure

```text
paper_MoE/
│
├── baselines/              # Scripts for baseline models and exploratory ablations
│
├── MoEData/            # Ablation notebooks for MoEData configuration
│
├── MoETask/            # Ablation notebooks for MoETask configuration
│
├── drift-moe/                 # Code used in the paper's experiments
│   ├── baselines.py           # Test-then-train evaluation of baseline methods
│   ├── config.py              # Experiment configuration module
│   ├── data_loader.py         # Stream loading utilities
│   ├── experiment_tracker.py  # TensorBoard + CSV logger for all evaluation metrics
│   ├── experts.py             # Expert wrapper using CapyMOA's Hoeffding Trees
│   ├── moe_model.py           # DriftMoE model definition and training logic
│   └── run_experiments.py     # Master script to execute all configured experiments
```

> ⚠️ **Note:** The code under `drift-moe/` is the exact codebase used to generate the results in the paper. Scripts in `baselines/`, `MoEData/`, and `MoETask` were used for ablation and exploratory studies.

---

## 🛠️ Requirements

* Python 3.10, 3.11, or 3.12
* PyTorch
* CapyMOA
* scikit-learn
* TensorBoard

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running Experiments

To run all experiments as defined in the configuration:

```bash
python train/run_experiments.py
```

Modify `train/config.py` to set up or extend experiment definitions.

---

## 📊 Logging

Experiments are logged using both TensorBoard and CSVs. Visualize results with:

```bash
tensorboard --logdir runs/
```

---

## 📄 Citation

If you use this software or refer to our framework, please cite the following:

```
@inproceedings{driftmoe2025,
  title={DriftMoE: A Mixture of Experts Approach to Handle Concept Drifts},
  author={Aspis, Miguel and Cajas Ordóñez, Sebastián and Suárez-Cetrulo, Andrés L. and Simón Carbajo, Ricardo},
  year={2025}
}
```

Or in APA style:

**Aspis, M., Cajas Ordóñez, S. A., Suárez-Cetrulo, A. L., & Simón Carbajo, R. (2025).** *DriftMoE: A Mixture of Experts Approach to Handle Concept Drifts (Version 1.0.0) \[Computer software].* [https://github.com/miguel-ceadar/drift-moe](https://github.com/miguel-ceadar/drift-moe)

---

## 🔗 License

[CC BY-NC-SA 4.0](https://github.com/miguel-ceadar/drift-moe/blob/main/LICENSE)
---

## 📩 Contact

For questions or contributions, please open an issue or contact the authors via GitHub.
