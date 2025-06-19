Repository for the paper DriftMoE: A Mixture of Experts Approach to Handle Concept Drifts

Repository structure:
- paper_MoE
    - baseline: contains initial scripts studying baseline performances
    - pipeline_1: notebooks containing initial explorations on MoEData
    - pipeline_2: notebooks containing initial explorations on MoETask
    - train: scripts for the training and all experiments run:
        - baselines.py: prequential test-than-train of baselines used in the paper
        - config.py: configuration of experiments to run, allows one-off arguments to be passed or experiments to be set programatically
        - data_loader.py: Utilities for loading and going through data streams
        - experiment_tracker.py: tensorboard + csv based experiment tracker that tracks Loss, Accuracy, Kappa M and Kappa Temporal metrics
        - experts.py: Wrapper around CapyMOA's HoeffdingTree class for easier use when training
        - moe_model.py: main MoE class and training functions containing all variations
        - run_experiments.py: Helper script that loops through all experiments defined in config.py