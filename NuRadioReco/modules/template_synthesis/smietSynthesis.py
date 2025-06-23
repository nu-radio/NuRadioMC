import copy
import h5py
import bisect
import numpy as np
from typing import Generator

import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.sim_station

from NuRadioReco.utilities import units
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.parameters import showerParameters as shp
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
        self._origin_shower = None
        self.origin_sim_shower = None
        self.template_synthesis = TemplateSynthesis(freq_ar=freq_ar)

    def begin(self, shower_path: str, template_path: str | None = None) -> None:
        """
        Process an origin shower into a template and obtain necessary parameters for the `run()` function.

        Parameters
        ----------
        shower_path: str
            The path to the HDF5 file containing the sliced simulation.
        """
        self._origin_shower: SlicedShower = SlicedShower(shower_path)

        if template_path is not None:
            self.template_synthesis.load_template(template_path)
        else:
            self.template_synthesis.make_template(self._origin_shower)

        # Some parameters are not read out by the SMIET software, so we get them out manually
        corsika = h5py.File(shower_path, "r")

        self.evt_nr = corsika["inputs"].attrs["EVTNR"]
        self.gps_secs = corsika["CoREAS"].attrs["GPSSecs"]
        self.time_resolution = corsika["CoREAS"].attrs["TimeResolution"] * units.s
        self.origin_sim_shower = create_sim_shower_from_hdf5(corsika)

        corsika.close()

    def origin_shower(self):
        """
        Return the origin shower as an Event object

        Returns
        -------
        evt: NuRadioReco.framework.event.Event
            The origin shower wrapped in an Event object
        """
        evt = NuRadioReco.framework.event.Event(0, self.evt_nr)
        evt.set_event_time(self.gps_secs, format="gps")

        # The traces are stored in a SimStation
        sim_station = NuRadioReco.framework.sim_station.SimStation(0)
        sim_station.set_is_cosmic_ray()

        for efield_ind, ant_name in enumerate(self._origin_shower.antenna_names):
            efield = np.zeros((self._origin_shower.trace_length, 3))
            for slice_ind in range(self._origin_shower.nr_of_slices):
                efield_slice, efield_start_time = (
                    self._origin_shower.get_trace_slice_on_sky(
                        ant_name,
                        int((slice_ind + 1) * self._origin_shower.slice_grammage),
                        return_start_time=True,
                    )
                )
                efield += efield_slice

            add_electric_field_to_sim_station(
                sim_station,
                efield_ind,
                efield.T,
                efield_start_time,
                self._origin_shower.zenith,
                self._origin_shower.azimuth,
                1 / self.time_resolution,
                efield_position=np.squeeze(
                    self._origin_shower.get_antenna_position(ant_name)
                ),
            )

        # SimStation is an attribute of a Station
        stn = NuRadioReco.framework.station.Station(0)
        stn.set_sim_station(sim_station)
        evt.set_station(stn)

        # Add RadioShower to Event such that it works nicely with the interpolator
        evt.add_sim_shower(copy.deepcopy(self.origin_sim_shower))

        yield evt

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
        if type(target_showers) is not list:
            target_showers = [target_showers]

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
                    self._origin_shower.zenith,
                    self._origin_shower.azimuth,
                    1 / self._origin_shower.coreas_settings["time_resolution"],
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


# Interpolated synthesis class
class smietInterpolated:
    def __init__(self) -> None:
        self.synthesis: list[TemplateSynthesis] = []
        self.origin_xmax = []

        self.evt_nr = 999999
        self.gps_secs = 0

        self._time_resolution = None

    @property
    def zenith(self):
        return self.synthesis[0].template_information["geometry"][0]

    @property
    def azimuth(self):
        return self.synthesis[0].template_information["geometry"][1]

    @property
    def time_resolution(self):
        if self._time_resolution is None:
            self._time_resolution = (
                1
                / self.synthesis[0].antenna_information["time_axis"].shape[1]
                / self.synthesis[0].frequencies[1]
            )
        return self._time_resolution

    def begin(self, showers: list[str], templates: list[str] | None = None):
        my_synthesis = []
        for shower_ind, shower_path in enumerate(showers):
            synthesis = TemplateSynthesis()

            if templates is not None:
                synthesis.load_template(templates[shower_ind])
            else:
                origin_shower: SlicedShower = SlicedShower(shower_path)
                synthesis.make_template(origin_shower)

            my_synthesis.append(synthesis)

        self.synthesis = sorted(
            my_synthesis,
            key=lambda x: x.template_information["xmax"],
        )

        self.origin_xmax = (
            np.array([synth.template_information["xmax"] for synth in self.synthesis])
            * units.g
            / units.cm2
        )

    def synthesise_single_shower(self, synth_ind, target):
        synth_geo, synth_ce = self.synthesis[synth_ind].map_template(target)

        x, y = self.synthesis[synth_ind].antenna_information["position_showerplane"].T

        synth_shower_plane = geo_ce_to_e(np.stack((synth_geo, synth_ce), axis=2), x, y)

        return synth_shower_plane

    def calculate_interpolation_t(self, target_xmax, index_upper):
        """
        This is the `t` value that needs to be multiplied by the upper synthesis
        """
        lower_xmax = self.origin_xmax[index_upper - 1]
        upper_xmax = self.origin_xmax[index_upper]

        return (target_xmax - lower_xmax) / (upper_xmax - lower_xmax)

    def run(self, target_showers: list[Shower]):
        if type(target_showers) is not list:
            target_showers = [target_showers]

        for shower_ind, shower in enumerate(target_showers):
            upper_shower = bisect.bisect(
                self.origin_xmax, shower.xmax * units.g / units.cm2
            )
            if upper_shower == 0 or upper_shower == len(self.origin_xmax):
                raise ValueError(
                    f"The Xmax {shower.xmax} is outside of the interpolation range"
                )

            lower_shower = upper_shower - 1

            synth_upper = self.synthesise_single_shower(upper_shower, shower)
            synth_lower = self.synthesise_single_shower(lower_shower, shower)

            t = self.calculate_interpolation_t(
                shower.xmax * units.g / units.cm2, upper_shower
            )
            synth_shower_plane = t * synth_upper + (1 - t) * synth_lower

            # Transform synth_shower into Event
            evt = NuRadioReco.framework.event.Event(shower_ind, self.evt_nr)
            evt.set_event_time(self.gps_secs, format="gps")

            # The traces are stored in a SimStation
            sim_station = NuRadioReco.framework.sim_station.SimStation(0)
            sim_station.set_is_cosmic_ray()

            for efield_ind, (efield, efield_position, efield_time_axis) in enumerate(
                zip(
                    synth_shower_plane,
                    self.synthesis[lower_shower].antenna_information["position"],
                    self.synthesis[lower_shower].get_time_axis(),
                )
            ):
                add_electric_field_to_sim_station(
                    sim_station,
                    efield_ind,
                    efield.T,
                    efield_time_axis[0],
                    self.zenith,
                    self.azimuth,
                    1 / self.time_resolution,
                    efield_position=efield_position,
                )

            # SimStation is an attribute of a Station
            stn = NuRadioReco.framework.station.Station(0)
            stn.set_sim_station(sim_station)
            evt.set_station(stn)

            # Add RadioShower to Event such that it works nicely with the interpolator
            # evt.add_sim_shower(copy.deepcopy(self.origin_sim_shower))

            yield evt
