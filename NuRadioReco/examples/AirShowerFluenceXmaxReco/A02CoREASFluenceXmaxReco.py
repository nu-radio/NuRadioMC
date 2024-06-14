"""
Reconstruction of the position of the position of the shower maximum Xmax
using the "LOFAR"-style analysis where the best CoREAS simulation is found
which matches the data best by comparing the energy fluence footprint.

The procedure is the following (following the description in S. Buitink Phys. Rev. D 90, 082003 https://doi.org/10.1103/PhysRevD.90.082003)
 * One star-shape CoREAS simulation is read in and interpolate to obtain the electric field
   at every antenna position. This is handled by the `readCoREASDetector` module
 * A full detector simulation is performed, i.e., simulation of antenna and signal chain
   response, adding of noise, electric field reconstruction (using standard unfolding), and
   calculation of the energy fluence at every antenna position
 * we loop over a set of CoREAS simulations (excluding the simulation we used to generate the data)
   of the same air-shower direction and energy and
   fit the fluence footprint to the simulated data to determine the core position and amplitude
   normalization factor. The chi^2 of the best fit is saved. 
 * the best fit chi^2 values are plotted against the true Xmax value from the simulations
 * a parabola is fitted to determine the most likely Xmax value
"""

import os
import glob
import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import constants
from scipy import optimize as opt
from astropy import time
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.utilities import units
import NuRadioReco.modules.io.coreas.readCoREASDetector
import NuRadioReco.modules.io.coreas.simulationSelector
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.voltageToAnalyticEfieldConverter
import NuRadioReco.modules.voltageToEfieldConverterPerChannelGroup
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.io.eventWriter
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp

if not os.path.exists("plots"):
    os.makedirs("plots")

# TODO: Add boilerplate code to download data

# Initialize detector
det = detector.Detector(json_filename="grid_array_SKALA_InfFirn.json",
                        assume_inf=False, antenna_by_depth=False)
det.update(time.Time("2023-01-01T00:00:00", format='isot', scale='utc'))


# Initialize all modules that are used in the reconstruction pipeline
electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
electricFieldSignalReconstructor.begin()
simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter(log_level=logging.INFO)
efieldToVoltageConverter.begin(debug=False, pre_pulse_time=0, post_pulse_time=0)
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
triggerSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
triggerSimulator.begin()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin()
voltageToEfieldConverter = NuRadioReco.modules.voltageToEfieldConverter.voltageToEfieldConverter()
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
voltageToEfieldConverterPerChannelGroup = NuRadioReco.modules.voltageToEfieldConverterPerChannelGroup.voltageToEfieldConverterPerChannelGroup()
voltageToEfieldConverterPerChannelGroup.begin(use_MC_direction=True)

coreas_reader = NuRadioReco.modules.io.coreas.readCoREASDetector.readCoREASDetector()

antenna_pattern_provider = antennapattern.AntennaPatternProvider()

core = np.array([0, 0, 0]) * units.m


filt_settings = {'passband': [80 * units.MHz, 300 * units.MHz],
                            'filter_type': 'butter',
                            'order': 10}

input_file = "data/LOFAR/SIM000000.hdf5"
coreas_reader.begin(input_file)
# we only need a single realization of the shower, so we set the core position to zero for simplicity
for iE, evt in enumerate(coreas_reader.run(det, [core])):
    station = evt.get_station()
    simsh = evt.get_first_sim_shower()

    if 0:  # example plotting of the interpolated efields.
        ss = station.get_sim_station()
        for efield in ss.get_electric_fields():
            fig, ax = plt.subplots(1, 1)
            ax.plot(efield.get_times(), efield.get_trace()[1], label='eTheta')
            ax.plot(efield.get_times(), efield.get_trace()[2], label='ePhi')
            ax.set_xlabel('time [ns]')
            ax.set_ylabel('electric field [V/m]')
            ax.legend()
            pos  = det.get_relative_position(station.get_id(), efield.get_unique_identifier()[0][0])
            ax.set_title(f"E = {simsh[shp.energy]:.2g}eV, obs = {simsh[shp.observation_level]:.0f}m, pos = ({pos[0]:.0f}m , {pos[1]:.0f}m, {pos[2]:.0f}m) zenith = {efield[efp.zenith]/units.deg:.0f}deg, azimuth = {efield[efp.azimuth]/units.deg:.0f}deg", fontsize="small")
            fig.tight_layout()
            # fig.savefig(f"plots/efield_{efield.get_unique_identifier()[0][0]}.png")
            plt.show()

    # apply antenna response
    efieldToVoltageConverter.run(evt, station, det)

    # approximate the rest of the signal chain with a bandpass filter
    channelBandPassFilter.run(evt, station, det, **filt_settings)

    # add thermal noise of fixed noise temperature
    Tnoise = 300  # in Kelvin

    # calculate Vrms and normalize such that after filtering the correct Vrms is obtained
    min_freq = 0
    max_freq = 0.5 * det.get_sampling_frequency(station.get_id(), 1)
    ff = np.linspace(0, max_freq, 10000)
    filt = channelBandPassFilter.get_filter(ff, station.get_id(), None, det, **filt_settings)
    bandwidth = np.trapz(np.abs(filt) ** 2, ff)
    Vrms = (Tnoise * 50 * constants.k * bandwidth / units.Hz) ** 0.5
    amplitude = Vrms/ (bandwidth / max_freq) ** 0.5
    channelGenericNoiseAdder.run(evt, station, det, type="rayleigh", amplitude=1 * amplitude,
                                 min_freq=min_freq, max_freq=max_freq)

    # simulate if the air shower triggers the station. Requireing a 10 sigma simple threshold trigger
    triggerSimulator.run(evt, station, det, number_concidences=1, threshold=10*Vrms)
    if station.get_trigger('default_simple_threshold').has_triggered():

        eventTypeIdentifier.run(evt, station, "forced", 'cosmic_ray')
        channelSignalReconstructor.run(evt, station, det)

        # reconstruct the electric field for each dual-polarized antenna through standard unfolding
        voltageToEfieldConverterPerChannelGroup.run(evt, station, det)
        simsh = evt.get_sim_shower(0)

        # calcualte the energy fluence from every reconstructed efield and save in arrays
        electricFieldSignalReconstructor.run(evt, station, det)
        ff = []
        ff_error = []
        pos = []
        for efield in station.get_electric_fields():
            ff.append(np.sum(efield[efp.signal_energy_fluence]))
            ff_error.append(np.sum(efield.get_parameter_error(efp.signal_energy_fluence)))
            pos.append(efield.get_position())
        pos = np.array(pos)
        ff = np.array(ff)
        ff_error = np.array(ff_error)
        ff_error[ff_error == 0] = ff[ff_error != 0].min()  # ensure that the error is always larger than 0 to avoid division by zero in objective function

        if 1: # plot energy fluence footprint
            fig, ax = plt.subplots(1, 1)
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=ff, cmap=cm.gnuplot2_r, marker='o', edgecolors='k')
            cbar = fig.colorbar(sc, ax=ax, orientation='vertical')
            cbar.set_label('fluence [eV/m^2]')
            ax.set_title(f"E = {simsh[shp.energy]:.2g}eV, obs = {simsh[shp.observation_level]:.0f}m, core = {core[0]:.0f}m, {core[1]:.0f}m, {core[2]:.0f}m")
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            fig.savefig(f"plots/footprint_{iE}_core{core[0]:0f}_{core[1]:0f}.png")
            plt.close("all")
            # plt.show()

        if 0: # plot individual reconstructed efields
            for efield in station.get_electric_fields():
                if np.abs(efield.get_position()[1]) < 1:
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(efield.get_times(), efield.get_trace()[1], label='eTheta')
                    ax.plot(efield.get_times(), efield.get_trace()[2], label='ePhi')
                    ax.set_xlabel('time [ns]')
                    ax.set_ylabel('electric field [V/m]')
                    ax.legend()
                    tpos  = det.get_relative_position(station.get_id(), efield.get_unique_identifier()[0][0])
                    ax.set_title(f"E = {simsh[shp.energy]:.2g}eV, obs = {simsh[shp.observation_level]:.0f}m, pos = ({tpos[0]:.0f}m , {tpos[1]:.0f}m, {tpos[2]:.0f}m) zenith = {efield[efp.zenith]/units.deg:.0f}deg, azimuth = {efield[efp.azimuth]/units.deg:.0f}deg", fontsize="small")
                    fig.tight_layout()
                    fig.savefig(f"plots/efield/efield_core{iE}_{efield.get_unique_identifier()[0][0]}.png")
                    plt.close("all")
                    plt.show()

        # perform Xmax fit:

        input_files = glob.glob("data/LOFAR/*.hdf5")
        N_showers = len(input_files)

        # initialize arrays to store the fit results
        fmin = np.zeros(N_showers)
        success = np.zeros(N_showers, dtype=bool)
        Xmax = np.zeros(N_showers)
        # loop through all 
        for i, filename2 in enumerate(input_files):
            # skip the CoREAS simulation that was used to generate the event under test
            if(filename2 == input_file):
                continue

            # read in CoREAS simulation
            evt2 = coreas.read_CORSIKA7(filename2)
            Xmax[i] = evt2.get_first_sim_shower()[shp.shower_maximum]
            # preprocess simulation, apply the same bandpass filter 
            # that the fake data has. This is required because the energy 
            # fluence depends on the bandwidth/choosen filter
            ss = evt2.get_station(0).get_sim_station()
            electricFieldBandPassFilter.run(evt2, ss, det, **filt_settings)
            electricFieldSignalReconstructor.run(evt2, ss, None)
            # Initialize interpolator for core position fit
            coreasInterpolator = coreas.coreasInterpolator(evt2)
            coreasInterpolator.initialize_fluence_interpolator()

            # define objective function for core position and amplitude fit
            def obj(xyA):
                A = xyA[2]  # normalization
                core = np.array([xyA[0], xyA[1], 0]) * units.m  # core position
                ff_true = np.zeros_like(ff)
                for ip, p in enumerate(pos):
                    ff_true[ip] = coreasInterpolator.get_interp_fluence_value(p, core)
                ff_true *= A
                return np.sum((ff - ff_true) ** 2/ff_error**2)

            # perform minimization
            res = opt.minimize(obj, [core[0], core[1], 1], method='Nelder-Mead')
            # save fit results
            fmin[i] = res.fun
            success[i] = res.success
            print(f"fitting footprint {i} from file {filename2}")
            print(res)


        # End loop CoREAS simulations
        # now we can plot the best chi2 vs. Xmax
        ndf = len(ff) - 3
        m = fmin > 0 # let's exclude unsuccessful footprint fits
        xx = Xmax[m]/units.g * units.cm**2
        yy = fmin[m]/ndf

        # define parabola in standard-normal form
        def func(x, a, b, c):
            return a * (x-b)**2 + np.abs(c)

        # define objective function
        def obj(p, xx, yy):
            return np.sum((yy - func(xx, *p))**2)

        # fit parabola
        xmax_best_fit = xx[yy.argmin()]
        mask = (xx > (xmax_best_fit - 100)) & (xx < (xmax_best_fit + 100)) & success[m]
        res = opt.minimize(obj, [1, xmax_best_fit, 0], args=(xx[mask], yy[mask]), method='Nelder-Mead')
        popt = res.x

        xmax_best_fit = popt[1]
        true_xmax = simsh[shp.shower_maximum]/units.g * units.cm**2
        xmax_diff = np.abs(xmax_best_fit - true_xmax)

        print(f"true Xmax = {true_xmax:.2f} g/cm^2, best fit Xmax = {xmax_best_fit:.2f} g/cm^2, diff = {xmax_diff:.2f} g/cm^2")

        if 1:  # plot parabola
            fig, ax = plt.subplots(1, 1)
            ax.plot(xx, yy, 'o')
            ax.plot(xx[~success[m]], yy[~success[m]], 'ro')
            xxx = np.linspace(xx[mask].min(), xx[mask].max(), 1000)
            ax.plot(xxx, func(xxx, *popt))
            ax.vlines(simsh[shp.shower_maximum]/units.g * units.cm**2, 0, yy.max(), linestyles='dashed', label='true Xmax')
            ax.set_xlabel('Xmax [g/cm^2]')
            ax.set_ylabel('chi2 / ndf')
            ax.set_title(f"true Xmax = {true_xmax:.2f} g/cm^2, best fit Xmax = {xmax_best_fit:.2f} g/cm^2, diff = {xmax_diff:.2f} g/cm^2", fontsize="small")
            fig.tight_layout()
            fig.savefig(f"plots/Xmax_fit_{iE}.png")
            plt.close("all")
            # plt.show()