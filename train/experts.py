# experts.py

from capymoa.classifier import HoeffdingTree
import numpy as np
from river import metrics

class Expert:
    """
    Wraps a single CapyMOA HoeffdingTree classifier. On `.predict_proba`, returns
    a length-num_classes probability list (padded/uniform if necessary). On `.train`, uses
    HoeffdingTree.train() exactly as in joint_moe.py. On `.predict`, we choose argmax of
    predict_proba to get a single class prediction. We also maintain a River Accuracy metric.
    """

    def __init__(self, schema, num_classes: int, grace_period=50, confidence=1e-07, 
                 binary_split=False, stop_mem_management=False):
        self.num_classes = num_classes
        # Initialize CapyMOA HoeffdingTree with identical params across all experts
        self._clf = HoeffdingTree(
            schema=schema,
            grace_period=grace_period,
            confidence=confidence,
            binary_split=binary_split,
            stop_mem_management=stop_mem_management
        )
        # Maintain per-expert streaming accuracy (prequential)
        self.metric = metrics.Accuracy()

    def predict_proba(self, instance: object) -> list:
        """
        Return a length-num_classes list of P(y=c | x). If the tree is brand-new and
        returns None, we fallback to uniform. If returned list is shorter than num_classes,
        we pad with zeros on the right.
        """
        p_list = self._clf.predict_proba(instance)
        if p_list is None:
            # brand-new leaf: uniform prior
            return [1.0 / self.num_classes] * self.num_classes

        # CapyMOA returns a Python list of length ≤ num_classes
        if len(p_list) < self.num_classes:
            return list(p_list) + [0.0] * (self.num_classes - len(p_list))
        return list(p_list)

    def predict(self, inst) -> int:
        """
        Return a single predicted class by calling predict_one under the hood.
        CapyMOA's HoeffdingTree provides predict_one(x_dict) which returns a scalar class index.
        """
        return self._clf.predict(inst)

    def train(self, instance: object):
        """
        Update the HoeffdingTree with the new MOA instance. Exactly as in joint_moe.py,
        they call experts[eid].train(instance).
        """
        self._clf.train(instance)

    

    def update_metric(self, y_true: int, y_pred: int):
        """Update the internal Accuracy metric in a prequential fashion."""
        self.metric.update(y_true, y_pred)

    def get_metric(self) -> float:
        return self.metric.get()
