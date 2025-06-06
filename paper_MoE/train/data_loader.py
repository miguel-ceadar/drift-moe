# data_loader.py

from capymoa.stream.generator import LEDGeneratorDrift, SEA, RandomRBFGeneratorDrift
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
from capymoa.datasets import Electricity, Covtype
from capymoa.stream import stream_from_file

class DataLoader:
    """
    Wraps CapyMOA's MOAStream. Always uses a single MOAStream under the hood.
    You must supply a valid CLI string (e.g. '-s (ConceptDriftStream -s (...) …) -w 50000 -p 250000').
    """

    def __init__(self, dataset: str):
        if dataset == "led_g":
            self._stream = DriftStream(
                stream=[LEDGeneratorDrift(number_of_attributes_with_drift=1),
                
                GradualDrift(width=50_000, position=225_000),
            
                LEDGeneratorDrift(number_of_attributes_with_drift=3),
                GradualDrift(width=50_000, position=475_000),
                
                LEDGeneratorDrift(number_of_attributes_with_drift=5),
                GradualDrift(width=50_000, position=725_000),
                
                LEDGeneratorDrift(number_of_attributes_with_drift=7)
                    ]
            )
            
        if dataset == "led_a":
            self._stream = DriftStream(
                stream=[LEDGeneratorDrift(number_of_attributes_with_drift=1),
                
                GradualDrift(width=50, position=249_975),
            
                LEDGeneratorDrift(number_of_attributes_with_drift=3),
                GradualDrift(width=50, position=499_975),
                
                LEDGeneratorDrift(number_of_attributes_with_drift=5),
                GradualDrift(width=50, position=749_975),
                
                LEDGeneratorDrift(number_of_attributes_with_drift=7)
                    ]
            )
            

        if dataset == "sea_g":
            self._stream = DriftStream(
                stream=[SEA(function=1),
                
                GradualDrift(width=50_000, position=225_000),
            
                SEA(function=2),
                GradualDrift(width=50_000, position=475_000),
                
                SEA(function=4),
                GradualDrift(width=50_000, position=725_000),
                SEA(function=1)
                
                    ]
            )
        if dataset == "rbf_m":
            self._stream = RandomRBFGeneratorDrift(number_of_drifting_centroids=50, magnitude_of_change=0.0001)
        
        if dataset == "rbf_f":
            self._stream = RandomRBFGeneratorDrift(number_of_drifting_centroids=50, magnitude_of_change=0.001)
        
        if dataset == "elec":
            self._stream = Electricity(directory="~/drift_moe/datasets")
        
        if dataset == "covt":
            self._stream = Covtype(directory="~/drift_moe/datasets")
        
        if dataset == "airl":
            self._stream = stream_from_file("~/drift_moe/datasets/airlines.arff", dataset_name="Airlines")

        
        
        

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

