import copy
import h5py
import numpy as np
from typing import Generator

import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.sim_station

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.io.coreas.coreas import (
    create_sim_shower_from_hdf5,
    add_electric_field_to_sim_station,
)

from smiet.numpy import Shower, SlicedShower, TemplateSynthesis, geo_ce_to_e


class smietSynthesis:
    """
    Interface to the SMIET synthesis software, to synthesise electric field traces.

    Parameters
    ----------
    freq: {[30, 500], [30, 80]}
        The frequency interval to use for synthesis.

    Notes
    -----
    The synthesis algorithm also needs a central frequency. This is assigned automatically based on the
    frequency interval. If [30, 500] is selected, the central frequency is set to 100. In the [30, 80]
    case this is set to 50.
    """

    def __init__(self, freq=[30, 500]) -> None:
        if freq[0] == 30 and freq[1] == 500:
            freq_ar = [*freq, 100]
        elif freq[0] == 30 and freq[1] == 80:
            freq_ar = [*freq, 50]
        else:
            raise ValueError(
                "Currently only the values [30, 500] and [30, 80] are supported for the `freq` argument"
            )

        self.evt_nr = None
        self.gps_secs = None
        self.origin_shower = None
        self.origin_sim_shower = None
        self.template_synthesis = TemplateSynthesis(freq_ar=freq_ar)

    def begin(self, shower_path: str) -> None:
        """
        Process an origin shower into a template and obtain necessary parameters for the `run()` function.

        Parameters
        ----------
        shower_path: str
            The path to the HDF5 file containing the sliced simulation.
        """
        self.origin_shower = SlicedShower(shower_path)
        self.template_synthesis.make_template(self.origin_shower)

        # Some parameters are not read out by the SMIET software, so we get them out manually
        corsika = h5py.File(shower_path, "r")

        self.evt_nr = corsika["inputs"].attrs["EVTNR"]
        self.gps_secs = corsika["CoREAS"].attrs["GPSSecs"]
        self.origin_sim_shower = create_sim_shower_from_hdf5(corsika)

        corsika.close()

    @register_run()
    def run(
        self, target_showers: list[Shower]
    ) -> Generator[NuRadioReco.framework.event.Event]:
        """
        Synthesise showers and put them in an Event object.

        Parameters
        ----------
        target_showers: list[Shower]
            A list of target showers, each as Shower object

        Yields
        ------
        evt: NuRadioReco.framework.event.Event

        """
        for shower_ind, shower in enumerate(target_showers):
            synth_geo, synth_ce = self.template_synthesis.map_template(shower)
            x, y = self.template_synthesis.antenna_information["position_showerplane"].T
            synth_shower_plane = geo_ce_to_e(
                np.stack((synth_geo, synth_ce), axis=2), x, y
            )  # shape = (ANT, SAMPLES, 3)

            # Transform synth_shower into Event
            evt = NuRadioReco.framework.event.Event(shower_ind, self.evt_nr)
            evt.set_event_time(self.gps_secs, format="gps")

            # The traces are stored in a SimStation
            sim_station = NuRadioReco.framework.sim_station.SimStation(0)
            sim_station.set_is_cosmic_ray()

            for efield_ind, (efield, efield_position, efield_time_axis) in enumerate(
                zip(
                    synth_shower_plane,
                    self.template_synthesis.antenna_information["position"],
                    self.template_synthesis.get_time_axis(),
                )
            ):
                add_electric_field_to_sim_station(
                    sim_station,
                    efield_ind,
                    efield.T,
                    efield_time_axis[0],
                    self.origin_shower.zenith,
                    self.origin_shower.azimuth,
                    1 / self.origin_shower.coreas_settings["time_resolution"],
                    efield_position=efield_position,
                )

            # SimStation is an attribute of a Station
            stn = NuRadioReco.framework.station.Station(0)
            stn.set_sim_station(sim_station)
            evt.set_station(stn)

            # Add RadioShower to Event such that it works nicely with the interpolator
            evt.add_sim_shower(copy.deepcopy(self.origin_sim_shower))

            yield evt

    def end(self):
        pass
