import numpy as np
import sys
from scipy import signal, constants
from radiotools import helper as hp
from NuRadioReco.utilities import units

# to convert V**2/m**2 * ns -> V**2/m**2 * s -> J/m**2 -> eV/m**2
conversion_factor_integrated_signal = 1 / units.s * \
    constants.c * constants.epsilon_0 / units.eSI


def get_signal(sum_trace, tstep, window_width=100 * units.ns, kind="power"):
    """ 
    Calculates signal quantity from beam-formed waveform 
    
    Parameters
    ----------

    sum_trace : np.array(m,)
        beam-formed waveform with m samples

    tstep : double
        Sampling bin size
    
    window_width : double
        Time window size to calculate power 

    kind : str
        Key-word what to do: "amplitude", "power", or "hilbert_sum"
    
    Returns
    -------

    signal : double
        Signal calculated according to the specified metric
    """

    # find signal peak
    hilbenv = np.abs(signal.hilbert(sum_trace))
    peak_idx = np.argmax(hilbenv)

    if kind == "amplitude":
        return hilbenv[peak_idx]

    elif kind == "power" or kind == "hilbert_sum":
        trace_length = len(sum_trace)
        # shift peak in middle of trace
        sum_trace = np.roll(sum_trace, trace_length // 2 - peak_idx)
        peak_idx += trace_length // 2 - peak_idx

        # define signal window. If trace is to small -> pad
        idx_width = int(window_width / 2 // tstep)
        if trace_length < 2 * idx_width:
            sum_trace = np.hstack(
                [np.zeros(idx_width), sum_trace, np.zeros(idx_width)])
            peak_idx += idx_width

        sum_trace *= conversion_factor_integrated_signal * tstep

        if kind == "power":
            # return sum of squares within signal window
            return np.sum(sum_trace[peak_idx-idx_width:peak_idx+idx_width] ** 2)

        elif kind == "hilbert_sum":
            hilbenv = np.abs(signal.hilbert(sum_trace))
            return np.sum(hilbenv[peak_idx-idx_width:peak_idx+idx_width])

    else:
        sys.exit("get_signal(), kind = '{}' not supported".format(kind))


def interfere_traces_interpolation(target_pos, positions, traces, times, tab):
    """ 
    Calculate sum of time shifted waveforms.

    Performs a linear interpolation between samples.
    
    Parameters
    ----------

    target_pos : np.array(3,)
        source/traget location

    positions : np.array(n, 3)
        observer positions

    traces : np.array(n, m)
        waveforms of n observers with m samples

    times : np.array(n, m)
        time stampes of the waveforms of each observer

    tab : radiotools.atmosphere.refractivity.RefractivityTable
        Tabulated table of the avg. refractive index between two points
    
    Returns
    -------

    sum_trace : np.array(n, m)
        Summed trace

    """

    # positions all have to be in sea level plane coordinates!
    times = times
    tstep = times[0, 1] - times[0, 0]

    tshifts = get_time_shifts(target_pos, positions, tab)

    times_new = times - tshifts[:, None]
    first_time = np.amin(times_new)
    last_time = np.amax(times_new)

    time_sum = np.arange(first_time, last_time + tstep, tstep)
    sum_trace = np.zeros(len(time_sum))
    for trace, time in zip(traces, times_new):

        fidx = np.around((time[1:] - time_sum[0]) / tstep, 4)  # TODO: check if that makes sense
        idx = np.array(fidx, dtype=int)

        if not np.unique(idx).size == len(idx):
            sys.exit(
                "Index array has not unique entries. That is most probably a rounding issue!")

        f = (fidx - idx)[0]  # are all the same

        """ Linear interplation to match the binning of time_sum. """
        trace_new = (1 - f) * trace[1:] + f * trace[:-1]
        sum_trace[idx] += trace_new

    return sum_trace


def get_time_shifts(target_pos, positions, tab):
    """
    Calculates the time delay of an electromagnetic wave along a straight trajectories between 
    a source/traget location and several observers.
    
    Parameters
    ----------

    target_pos : np.array(3,)
        source/traget location

    positions : np.array(n, 3)
        observer positions (n observers)

    tab : radiotools.atmosphere.refractivity.RefractivityTable
        Tabulated table of the avg. refractive index between two points

    Returns
    -------

    tshifts : np.array(n,)
        Time delay in sec
    
    """

    tshifts = np.zeros(len(positions))
    for idx, pos in enumerate(positions):
        effective_refractivity = tab.get_refractivity_between_two_points_tabulated(
            target_pos, pos)

        dt = np.linalg.norm(target_pos - pos) * \
            (effective_refractivity + 1) / constants.c
        tshifts[idx] = dt

    return tshifts * units.s


def fit_axis(z, theta, phi, coreX, coreY):
    """ 
    Predicts the intersetction of an axis/line with horizontal layers at different heights.

    Line is described by an anchor on a horizontal plane (coreX, coreY) and a direction
    in spherical coordinates (theta, phi).

    Returns the position/intersection of the line with flat horizontal layers at 
    given height(s) z. Resulting array (positions) is flatten.

    Parameters
    ----------

    z : array
        The height(s) for which the position on the defined axis should be evaluated.

    theta : double
        Zenith angle of the axis

    phi : double
        Azimuth angle of the axis

    coreX : Double
        x-coordinate of the intersection of the axis with a horizontal plane with z = 0.

    coreY : Double
        y-coordinate of the intersection of the axis with a horizontal plane with z = 0.

    Returns
    -------

    points : array
        The flatten array of the positions on along the defined axis at heights given by "z"
    """
    axis = hp.spherical_to_cartesian(theta, phi)
    norm = z / axis[-1]
    norm = np.asarray(norm)  # when z is a float
    points = axis.reshape(1, 3) * norm[:, None] + \
        np.array([coreX, coreY, 0])[None, :]
    return points.flatten()


def get_intersection_between_line_and_plane(plane_normal, plane_anchor, line_direction, line_anchor, epsilon=1e-6):
    """
    Find intersection betweem a line and a plane.

    From https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python

    Parameters
    ----------
    plane_normal : np.array(3,)
        Normal vector of a plane

    plane_anchor : np.array(3,)
        Anchor of this plane

    line_direction : np.array(3,)
        Direction of a line

    line_anchor : np.array(3,)
        Anchor of this line

    epsilon : double
        Numerical precision 

    Returns
    -------

    psi : array(3,)
        Position of the intersection between plane and line

    """
    ndotu = plane_normal.dot(line_direction)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = line_anchor - plane_anchor
    si = -plane_normal.dot(w) / ndotu
    psi = w + si * line_direction + plane_anchor

    return psi
