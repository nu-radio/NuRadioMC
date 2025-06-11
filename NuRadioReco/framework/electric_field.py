from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.parameters as parameters
import NuRadioReco.framework.parameter_storage
import radiotools.coordinatesystems
from NuRadioReco.utilities.trace_utilities import get_stokes

import pickle
from NuRadioReco.utilities.io_utilities import _dumps
import logging
logger = logging.getLogger('NuRadioReco.ElectricField')


class ElectricField(NuRadioReco.framework.base_trace.BaseTrace,
                    NuRadioReco.framework.parameter_storage.ParameterStorage):
    def __init__(self, channel_ids, position=None,
                 shower_id=None, ray_tracing_id=None):
        """
        Initialize a new electric field object

        This object stores a 3 dimensional trace plus additional meta parameters

        Parameters
        ----------
        channel_ids: list of ints
            the channels ids this electric field is valid for.
            (For cosmic rays one electric field is typically valid
            for several channels. For neutrino simulations, we typically
            simulate the electric field for each
            channel separately)
        position: 3-dim array/list of floats
            the position of the electric field
        shower_id: int or None
            the id of the corresponding shower object
        ray_tracing_id: int or None
            the id of the corresponding ray tracing solution
        """
        NuRadioReco.framework.base_trace.BaseTrace.__init__(self)
        NuRadioReco.framework.parameter_storage.ParameterStorage.__init__(
            self, parameters.electricFieldParameters)

        self._channel_ids = channel_ids
        self._position = position
        if position is None:
            self._position = [0, 0, 0]

        self._shower_id = shower_id
        self._ray_tracing_id = ray_tracing_id

    def get_unique_identifier(self):
        """
        returns a unique identifier consisting of the tuple channel_ids, shower_id and ray_tracing_id
        """
        return (self._channel_ids, self._shower_id, self._ray_tracing_id)

    def set_channel_ids(self, channel_ids):
        self._channel_ids = channel_ids

    def get_channel_ids(self):
        return self._channel_ids

    def has_channel_ids(self, channel_ids):
        for channel_id in channel_ids:
            if channel_id not in self._channel_ids:
                return False
        return True

    def get_shower_id(self):
        return self._shower_id

    def get_ray_tracing_solution_id(self):
        return self._ray_tracing_id

    def get_position(self):
        """
        get position of the electric field relative to station position
        """
        return self._position

    def set_position(self, position):
        """
        set position of the electric field relative to station position
        """
        self._position = position

    def get_stokes_parameters(
            self, window_samples=None, vxB_vxvxB=False, magnetic_field_vector=None,
            site=None, filter_kwargs=None
        ):
        """
        Return the stokes parameters for the electric field.

        By default, the stokes parameters are returned in (eTheta, ePhi);
        this assumes the 3d efield trace is stored in (eR, eTheta, ePhi).
        To return the stokes parameters in (vxB, vxvxB) coordinates instead,
        one has to specify the magnetic field vector.

        Parameters
        ----------
        window_samples : int | None, default: None
            If None, return the stokes parameters over the full traces.
            If not None, returns a rolling average of the stokes parameters
            over ``window_samples``. This may be more optimal if the duration
            of the signal is much shorter than the length of the full trace.
        vxB_vxvxB : bool, default: False
            If False, returns the stokes parameters for the
            (assumed) (eTheta, ePhi) coordinates of the electric field.
            If True, convert to (vxB, vxvxB) first. In this case,
            one has to additionally specify either the magnetic field vector
            or the sit.
        magnetic_field_vector : 3-tuple of floats | None, default: None
            The direction of the magnetic field (in x,y,z)
        site : string | None, default: None
            The site of the detector. Can be used instead of the ``magnetic_field_vector``
            if the magnetic field vector for this site is included in ``radiotools``
        filter_kwargs : dict | None, default: None
            Optional arguments to bandpass filter the trace
            before computing the stokes parameters. They are passed on to
            `get_filtered_trace(**filter_kwargs)`

        Returns
        -------
        stokes : array of floats
            The stokes parameters. If ``window_samples=None`` (default), the shape of
            the returned array is ``(4,)`` and corresponds to the I, Q, U and V parameters.
            Otherwise, the array will have shape ``(4, len(efield) - window_samples + 1)``
            and correspond to the values of the stokes parameters over the specified
            window sizes.

        See Also
        --------
        NuRadioReco.utilities.trace_utilities.get_stokes : Function that computes the stokes parameters
        """
        if filter_kwargs:
            trace = self.get_filtered_trace(**filter_kwargs)
        else:
            trace = self.get_trace()

        if not vxB_vxvxB:
            return get_stokes(trace[1], trace[2], window_samples=window_samples)
        else:
            try:
                zenith = self.get_parameter(parameters.electricFieldParameters.zenith)
                azimuth = self.get_parameter(parameters.electricFieldParameters.azimuth)
                cs = radiotools.coordinatesystems.cstrafo(
                    zenith, azimuth, magnetic_field_vector=magnetic_field_vector, site=site)
                efield_trace_vxB_vxvxB = cs.transform_to_vxB_vxvxB(
                    cs.transform_from_onsky_to_ground(trace)
                )
                return get_stokes(*efield_trace_vxB_vxvxB[:2], window_samples=window_samples)
            except KeyError as e:
                logger.error("Failed to compute stokes parameters in (vxB, vxvxB), electric field does not have a signal direction")
                raise(e)


    def serialize(self, save_trace):
        base_trace_pkl = None
        if save_trace:
            base_trace_pkl = NuRadioReco.framework.base_trace.BaseTrace.serialize(self)

        data = NuRadioReco.framework.parameter_storage.ParameterStorage.serialize(self)

        data.update({
            'channel_ids': self._channel_ids,
            '_shower_id': self._shower_id,
            '_ray_tracing_id': self._ray_tracing_id,
            'position': self._position,
            'base_trace': base_trace_pkl
        })

        return _dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        if data['base_trace'] is not None:
            NuRadioReco.framework.base_trace.BaseTrace.deserialize(self, data['base_trace'])

        NuRadioReco.framework.parameter_storage.ParameterStorage.deserialize(self, data)

        if 'position' in data:  # for backward compatibility
            self._position = data['position']

        self._channel_ids = data['channel_ids']
        self._shower_id = data.get('_shower_id', None)
        self._ray_tracing_id = data.get('_ray_tracing_id', None)
