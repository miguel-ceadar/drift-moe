# data_loader.py

from capymoa.stream import MOAStream
from moa.streams import ConceptDriftStream


class DataLoader:
    """
    Wraps CapyMOA's MOAStream. Always uses a single MOAStream under the hood.
    You must supply a valid CLI string (e.g. '-s (ConceptDriftStream -s (...) …) -w 50000 -p 250000').
    """

    def __init__(self, cli_command: str):
        from capymoa.stream import MOAStream
        # Create the MOAStream; we pass `moa_stream=None` so that CLI is fully self-contained.
        self._stream = MOAStream(moa_stream=ConceptDriftStream(), CLI=cli_command)

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
        
        x_vec = inst.x 
        y_true = inst.y_index
        return x_vec, y_true
