import os
import json


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

        self.filepath = os.path.join(
            directory,
            f"station{station}",
            f"run{run_number}_glitch.json"
        )
        
        # Load JSON once and cache it
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                self.data = json.load(f)
        else:
            raise FileNotFoundError(
            f"Glitch information file not found: {self.filepath}")

    def _get_event_data(self, event_id):
        """Return cached information for an event."""
        return self.data.get(str(event_id), {})

    def get_bad_channels(self, event_id):
        """Return bad channels for an event."""
        return self._get_event_data(event_id).get("bad_channels", [])

    def get_glitch_channels(self, event_id):
        """Return glitching channels for an event."""
        return self._get_event_data(event_id).get("glitch_channels", [])

    def get_transition_channels(self, event_id):
        """Return transition channels for an event."""
        return self._get_event_data(event_id).get("transition_channels", [])



