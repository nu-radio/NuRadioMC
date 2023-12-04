import numpy as np
from radiotools import helper as hp
from NuRadioReco.utilities import units
from NuRadioMC.SignalGen import askaryan
from radiotools import coordinatesystems as cstrans
from NuRadioReco.utilities import fft
from NuRadioReco.detector import antennapattern
from matplotlib import pyplot as plt
import logging
logging.basicConfig()
antenna_pattern_provider = antennapattern.AntennaPatternProvider()

mylog = logging.getLogger("SignalModel")


def get_signal(N, dt, shower_energy, zenith, azimuth, vertex, observer, antenna_type):
    mylog.setLevel(10)
    shower_type = "had"

    mylog.info(f"calculating signal for {shower_energy/units.eV:.2g}eV shower energy, zenith = {zenith/units.deg:.0f}deg, "\
                f"azimuth = {azimuth/units.deg:.0f}deg, vertex = {vertex}, observer = {observer}, and {antenna_type}")
    # assume straight line propagation for now
    launch_vector = observer - vertex
    receive_vector = -1 * launch_vector
    distance = np.linalg.norm(launch_vector)
    launch_vector /= distance
    # be careful, zenith/azimuth angle always refer to where the neutrino came from,
    # i.e., opposite to the direction of propagation. We need the propagation direction here,
    # so we multiply the shower axis with '-1'
    shower_axis = -1 * hp.spherical_to_cartesian(zenith, azimuth)
    viewving_angle = hp.get_angle(launch_vector, shower_axis)

    polarization_direction = np.cross(launch_vector, np.cross(shower_axis, launch_vector))
    polarization_direction /= np.linalg.norm(polarization_direction)
    cs = cstrans.cstrafo(*hp.cartesian_to_spherical(*launch_vector))
    polarization_direction_onsky = cs.transform_from_ground_to_onsky(polarization_direction)

    mylog.info(f"d = {distance/units.m:.0f}m, viewing angle = {viewving_angle/units.deg:.0f}")

    spectrum = askaryan.get_frequency_spectrum(shower_energy, viewving_angle, N, dt, shower_type, 1.78, distance,
                                                "Alvarez2000")
    eR, eTheta, ePhi = np.outer(polarization_direction_onsky, spectrum)

    
    ori = [0, 0, 90 * units.deg, 0]
    frequencies = np.fft.rfftfreq(N, dt)
    antenna_pattern = antenna_pattern_provider.load_antenna_pattern(antenna_type)
    zenith_ant, azimuth_ant = hp.cartesian_to_spherical(*receive_vector)
    VEL = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith_ant, azimuth_ant, *ori)
    voltage_fft = VEL['theta'] *eTheta + VEL['phi'] * ePhi
    voltage_fft[np.where(frequencies < 5 * units.MHz)] = 0.
    return fft.freq2time(voltage_fft, sampling_rate=1/dt, n=N)
    

if __name__ == "__main__":
    N = 1048
    dt = 1 * units.ns
    signal = get_signal(N, dt, 1e18*units.eV, 90*units.deg, 0, np.array([500., 0., -500.]),
                        np.array([0.,0.,-10.]), "bicone_v8_inf_n1.78")
    
    tt = np.arange(0, N*dt, dt)
    fig, ax = plt.subplots(1, 1)
    ax.plot(tt, signal, label="90deg")
    
    
    signal = get_signal(N, dt, 1e18*units.eV, 91*units.deg, 0, np.array([500., 0., -500.]),
                        np.array([0.,0.,-10.]), "bicone_v8_inf_n1.78")
    ax.plot(tt, signal, label="91deg")
    ax.legend()
    fig.tight_layout()
    plt.show()
    