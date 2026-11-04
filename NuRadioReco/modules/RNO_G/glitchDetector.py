import os
import json
import gzip

class GlitchInfo:

    def __init__(self, run_number, station, directory):
        """
        Load and cache glitch information for a single run.

        Parameters
        ----------
        run_number : int
            Run number.
        station : int
            Station number.
        directory : str
            Parent directory containing station subdirectories.
        """

        self.run_number = run_number
        self.station = station
        self.directory = directory
        
        filename = f"run{self.run_number:07d}_glitch.json.gz"
        
        self.filepath = os.path.join(
            directory,
            f"station{station}",
            filename
        )

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Glitch information file not found: {self.filepath}"
            )

        # Load and cache the compressed JSON file.
        with gzip.open(self.filepath, "rt", encoding="utf-8") as f:
            self.data = json.load(f)

    def _get_event_data(self, event_id):
        """Return cached information for an event."""
        return self.data.get(str(event_id), {})

    def get_bad_channels(self, event_id):
        """Return bad channels for an event."""
        return self._get_event_data(event_id).get("bad", [])

    def get_glitch_channels(self, event_id):
        """Return glitching channels for an event."""
        return self._get_event_data(event_id).get("glitch", [])

    def get_transition_channels(self, event_id):
        """Return transition channels for an event."""
        return self._get_event_data(event_id).get("transition", [])

