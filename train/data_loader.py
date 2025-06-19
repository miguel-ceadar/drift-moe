# data_loader.py

from capymoa.stream.generator import LEDGeneratorDrift, SEA, RandomRBFGeneratorDrift
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
from capymoa.datasets import Electricity, Covtype
from capymoa.stream import stream_from_file
from capymoa.instance import LabeledInstance

from collections import deque

class DelayedLabelStream:
    def __init__(self, base_stream, delay: int):
        self._stream = base_stream
        self._delay = delay
        self._buffer = deque(maxlen=delay)
        for _ in range(delay):
            if self._stream.has_more_instances():
                inst = self._stream.next_instance()
                self._buffer.append(inst.y_index)
            else:
                self._buffer.append(0.)

    def has_more_instances(self):
        return self._stream.has_more_instances()

    def restart(self):
        """Restart the MOAStream from the beginning."""
        self._stream.restart()

    def get_schema(self):
        """Retrieve the schema object (needed to initialize CapyMOA HoeffdingTree)."""
        return self._stream.get_schema()

    def next_instance(self):
        inst = self._stream.next_instance()
        y_old = self._buffer.popleft()
        self._buffer.append(inst.y_index)
        buffered_inst = LabeledInstance.from_array(schema=inst.schema, x=inst.x, y_index=y_old)
        return buffered_inst




class DataLoader:
    """
    Wraps CapyMOA's MOAStream. Always uses a single MOAStream under the hood.
    """

    def __init__(self, dataset: str, seed: int, label_delay: int = 0):
        if dataset == "led_g":
            stream = DriftStream(
                stream=[LEDGeneratorDrift(number_of_attributes_with_drift=1, instance_random_seed=seed),
                
                GradualDrift(width=50_000, position=225_000),
            
                LEDGeneratorDrift(number_of_attributes_with_drift=3, instance_random_seed=seed),
                GradualDrift(width=50_000, position=475_000),
                
                LEDGeneratorDrift(number_of_attributes_with_drift=5, instance_random_seed=seed),
                GradualDrift(width=50_000, position=725_000),
                
                LEDGeneratorDrift(number_of_attributes_with_drift=7, instance_random_seed=seed)
                    ]
            )
            
        if dataset == "led_a":
            stream = DriftStream(
                stream=[LEDGeneratorDrift(number_of_attributes_with_drift=1, instance_random_seed=seed),
                
                GradualDrift(width=50, position=249_975),
            
                LEDGeneratorDrift(number_of_attributes_with_drift=3, instance_random_seed=seed),
                GradualDrift(width=50, position=499_975),
                
                LEDGeneratorDrift(number_of_attributes_with_drift=5, instance_random_seed=seed),
                GradualDrift(width=50, position=749_975),
                
                LEDGeneratorDrift(number_of_attributes_with_drift=7, instance_random_seed=seed)
                    ]
            )
            

        if dataset == "sea_g":
            stream = DriftStream(
                stream=[SEA(function=1,instance_random_seed=seed),
                
                GradualDrift(width=50_000, position=225_000),
            
                SEA(function=2, instance_random_seed=seed),
                GradualDrift(width=50_000, position=475_000),
                
                SEA(function=4, instance_random_seed=seed),
                GradualDrift(width=50_000, position=725_000),
                SEA(function=1, instance_random_seed=seed)
                
                    ]
            )
        if dataset =="sea_a":
            stream = DriftStream(
                    stream=[SEA(function=1, instance_random_seed=seed),
                    GradualDrift(width=50, position=249_975),
                    SEA(function=2, instance_random_seed=seed),
                    GradualDrift(width=50, position=499_975),
                    SEA(function=4, instance_random_seed=seed),
                    GradualDrift(width=50, position=749_975),
                    SEA(function=1, instance_random_seed=seed)
                    ]
                )
        if dataset == "rbf_m":
            stream = RandomRBFGeneratorDrift(number_of_drifting_centroids=50, magnitude_of_change=0.0001, instance_random_seed=seed, model_random_seed=seed)
        
        if dataset == "rbf_f":
            stream = RandomRBFGeneratorDrift(number_of_drifting_centroids=50, magnitude_of_change=0.001, instance_random_seed=seed, model_random_seed=seed)
        
        if dataset == "elec":
            stream = Electricity(directory="/home/miguel/drift_moe/datasets")
        
        if dataset == "covt":
            stream = Covtype(directory="/home/miguel/drift_moe/datasets")
        
        if dataset == "airl":
            stream = stream_from_file("/home/miguel/drift_moe/datasets/airlines.arff", dataset_name="Airlines")

        if label_delay > 0:
            self._stream = DelayedLabelStream(stream, delay=label_delay)
        else:
            self._stream = stream
        

    def restart(self):
        """Restart the MOAStream from the beginning."""
        self._stream.restart()

    def get_schema(self):
        """Retrieve the schema object (needed to initialize CapyMOA HoeffdingTree)."""
        return self._stream.get_schema()

    def next_instance(self):
        """
        Return the next (x_vec, y_index) tuple.
        Internally we rely on MOAStream.next_instance(), which yields an instance with
        `.x` and `.y_index`.
        """
        inst = self._stream.next_instance()
        
        return inst
    def has_more_instances(self):
        return self._stream.has_more_instances()

