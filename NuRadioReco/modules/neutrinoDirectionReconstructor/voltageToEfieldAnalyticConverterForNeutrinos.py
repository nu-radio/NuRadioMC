from __future__ import absolute_import, division, print_function, unicode_literals
from NuRadioReco.modules.base.module import register_run
import os
import time
import random
import numpy as np
from scipy import signal
from scipy.signal import correlate
from scipy import optimize as opt
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from types import SimpleNamespace
import collections

from radiotools import helper as hp
from radiotools import plthelpers as php
from radiotools import coordinatesystems as cs

from NuRadioReco.detector import detector
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import units, fft, trace_utilities


from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.modules import channelResampler as CchannelResampler
import NuRadioReco.framework.electric_field
from NuRadioReco.utilities.geometryUtilities import get_time_delay_from_direction

from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalGen.parametrizations import get_time_trace

from NuRadioMC.utilities import medium
from radiotools import coordinatesystems as cstrans
from NuRadioMC.SignalProp import analyticraytracing as ray
from radiotools import plthelpers as php
from NuRadioMC.SignalGen import askaryan as ask
from NuRadioReco.utilities import geometryUtilities as geo_utl
import time

channelResampler = CchannelResampler.channelResampler()
channelResampler.begin(debug=False)

class voltageToAnalyticEfieldConverterNeutrinos:
    """
    This module reconstructs the electric-field by forward folding an analytic
    pulse function through the antenna.

    This module is specified for neutrinos.

    Before the reconstruction, the channels are upsampled.

    The module is optimized for an electric-field reconstruction using two
    channels. First, the energy and the viewing angle are fitted using a Vpol
    antenna. Then the viewing angle is fixed in the Hpol antenna, and the energy
    is fitted.
    """

    def __init__(self):
        self.begin()

    def begin(self):
        """
        begin method. This function is executed before the event loop.
        The antenna pattern provieder is initialized here.
        """
        self.antenna_provider = antennapattern.AntennaPatternProvider()
        pass

    def run(self, evt, station, det, icemodel, shower_type='HAD', use_channels=[0,1,2,3], attenuation_model='SP1',
            parametrization='Alvarez2000', hilbert=False, use_bandpass_filter=False, passband_low={}, passband_high={},
            include_focusing=False, use_MC=True, n_samples_multiplication_factor=1, plot_traces_with_true_input=False, debug=False):

        """
        run method. This function is executed for each event

        Parameters
        ----------
        evt
        station
        det
        icemodel
        shower_type: string
            determines the shower type during reconstructing the voltage traces.
        use_channels: array of ints
            the channel ids for the antennas used for reconstructing voltage traces.
        attenuation_model: string
            the frequency dependent ice attenuation model used
        parametrization: string
            neutrino parametrization model used for the reconstruction.
        hilbert: bool
            if True, use the hilbert envelope of the traces in the chi2 fit
        use_bandpass_filter: bool
            if True, the voltage traces are filtered according to the passband_low and passband_high inputs
        passband_low: dictionary
            map between channel id and complex filter amplitudes which themselves are array of complex floats.
        passband_high: dictionary
            map between channel id and complex filter amplitudes which themselves are array of complex floats
        include_focusing: bool
            if True, include amplification of signal due to focusing effects.
        use_MC: bool
            if True use simulated properties such as vertex position instead of reconstructed properties
        n_samples_multiplication_factor: int
            expands the time window length by n. It is used to consider all ray tracing solutions when fitting reconstructed traces against event. The value of
            n is site dependent and can speed up the algorithm and save space considerably.
        plot_traces_with_true_input: bool,
            for simulated events only, if True produces a figure overlaying the simulated voltage traces with the reconstructed voltage traces using the true nu direction and energy
        debug: bool
            if True debug plotting is enabled
        """

        def minimizer(params, minimizer=True):
            """
            params: array of floats
                parameters to be minimized, [neutrino zenith angle, neutrino azimuth angle, log10(shower energy)]
            hilbert: bool
                if True, the hilbert envelop is used for the fit
            """
            if(len(params) != 3):
                raise Exception("Length of input params does not equal 3! Input parameters should contain neutrino zenith, neutrino azimuth, and log10(shower energy)")
            else:
                nu_zenith, nu_azimuth, log10_shower_energy = params
                shower_energy = 10**log10_shower_energy

            chi2 = np.inf

            tmp = 0


            """
            calculates the voltage traces at the antenna for a given shower direction and shower energy
            """
            thetas = {}
            traces = np.zeros((n_antennas, n_samples))

            nu_direction = -1*hp.spherical_to_cartesian(nu_zenith, nu_azimuth)

            for iA, position in enumerate(antenna_positions):
                trace_spectrum = np.zeros(len(ff), dtype=complex)
                # loop through both ray tracing solutions and add up resulting voltage traces
                for iS in range(n_ray_tracing_solutions[iA]):
                    # calculate polarization of radio signal
                    polarization_direction = np.cross(launch_vectors[iA, iS], np.cross(nu_direction, launch_vectors[iA, iS]))
                    polarization_direction /= np.linalg.norm(polarization_direction)
                    cs = cstrans.cstrafo(*hp.cartesian_to_spherical(*launch_vectors[iA, iS]))
                    polarization_on_sky = cs.transform_from_ground_to_onsky(polarization_direction)  # this is the polarization in eR, eTheta, ePhi
                    cs_at_antenna = cstrans.cstrafo(*hp.cartesian_to_spherical(*receive_vectors[iA, iS]))


                    theta = hp.get_angle(nu_direction, launch_vectors[iA, iS])  # calculate viewing angle
                    if iA not in thetas:
                        thetas[iA] = {}
                    thetas[iA][iS] = theta
                    # get the efield at the antenna
                    spectrum = ask.get_frequency_spectrum(shower_energy, theta, n_samples, dt, shower_type=shower_type, n_index=n_index,
                                               R=travel_distance[iA, iS], model=parametrization)
                    spectrum *= attenuation[iA, iS]  # apply ice attenuation
                    eR, eTheta, ePhi = np.outer(polarization_on_sky, spectrum)
                    eTheta *= reflection_coefficients_theta[iA, iS]  # apply reflection coefficients
                    ePhi *= reflection_coefficients_phi[iA, iS]  # apply reflection coefficients

                    # convolve efield with antenna response
                    rec_zenith, rec_azimuth = hp.cartesian_to_spherical(*receive_vectors[iA, iS])
                    VEL = antenna_patterns[iA].get_antenna_response_vectorized(ff, rec_zenith, rec_azimuth, *antenna_orientations[iA])
                    voltage_spectrum = VEL["theta"] * eTheta + VEL["phi"] * ePhi

                    # add another time delay to account for propagation time differences
                    dT = travel_time[iA][iS] - travel_time_min
                    phase = -1j * ff * 2* np.pi * dT
                    voltage_spectrum *= np.exp(phase)

                    # add another time delay to account cable delay differences
                    dT = cable_delays[iA] - cable_delays.min()
                    phase = -1j * ff * 2* np.pi * dT
                    voltage_spectrum *= np.exp(phase)

                    if include_focusing:
                        voltage_spectrum = voltage_spectrum * focusing[iA,iS]

                    trace_spectrum += voltage_spectrum



                if use_bandpass_filter:
                    mask = ff > 0
                    order = 5
                    b, a = signal.butter(order, passband_high[use_channels[iA]], 'bandpass', analog=True)
                    w, h = signal.freqs(b, a, ff[mask])
                    f = np.zeros_like(ff, dtype=np.complex)
                    f[mask] = h
                    trace_spectrum *= f

                    order = 10
                    b, a = signal.butter(order, passband_low[use_channels[iA]], 'bandpass', analog=True)
                    w, h = signal.freqs(b, a, ff[mask])
                    f = np.zeros_like(ff, dtype=np.complex)
                    f[mask] = h
                    trace_spectrum *= f

                traces[iA] = fft.freq2time(trace_spectrum, sampling_rate=1 / dt, n=n_samples)


            # Gets an array of maximum amplitudes
            maxAmps = []
            for iT, channel in enumerate(station.iter_channels(use_channels=use_channels)):
                maxAmps.append(np.max(channel.get_trace()))

            # Correlate channel with max SNR between reconstructed and true traces to find global time time_offset
            # Commented out code is for upsampling before finding the correlation value. However it does make results better so needs to be further studied
            #channelResampler.run(evt, station, det, sampling_rate=50*units.GHz)
            #sampling_rate_ratio = 50.0*units.GHz / sampling_rate
            maxChSNR = np.argmax(maxAmps)
            analytic_trace = traces[maxChSNR]
            trace = station.get_channel(use_channels[maxChSNR]).get_trace()
            #analytic_trace = signal.resample(analytic_trace, len(trace)*n_expand)
            corr = signal.hilbert(signal.correlate(trace,analytic_trace))
            toffset = (np.argmax(corr) - (len(corr)/2))#/sampling_rate_ratio
            #channelResampler.run(evt, station, det, sampling_rate=sampling_rate)


            # This prevents wrap arounds of signals that are at the edge of the window and when n_expand is small. First pad with zeros, then roll, then chop.
            padNum = 500
            for iT in range(len(traces)):
                hstack = [np.zeros(padNum, dtype=traces[iT][0].dtype), traces[iT], np.zeros(padNum, dtype=traces[iT][0].dtype)]
                trace = np.hstack(hstack)
                trace = np.roll(trace,int(toffset))
                traces[iT] = trace[padNum:padNum + len(traces[iT])]


            for iCh, channel in enumerate(station.iter_channels(use_channels=use_channels)):
                trace = channel.get_trace()
                # Note that the additional cut on the analytic_trace is to account for the variable n_expand which can double triple or whatever the size of the analytic trace
                # This is necessary to capture additional pulses correctly with the global time offsets
                # The expansion is always equal in length before and after trace, so the resulting trace to minimize on is always the centered bit with length equal to the channel.get_trace() length
                analytic_trace = traces[iCh][int(n_samples/2) - int(len(trace)/2):int(n_samples/2)+int(len(trace)/2)]

                if hilbert:
                    tmp += np.sum(abs((abs(signal.hilbert(analytic_trace)) - abs(signal.hilbert(trace))))**2)
                else:
                    tmp += np.sum((abs(analytic_trace - trace))**2)

            if tmp < chi2:
                chi2 = tmp

            if minimizer:
                return chi2/(2*noise_RMS**2)
            else:
                 return traces, thetas


        def plotSimulatedVersusReconstructedTraces(nu_zenith, nu_azimuth, shower_energy, save):
            traces, viewing_angles_reco = minimizer([nu_zenith, nu_azimuth, np.log10(shower_energy)],minimizer = False)
            ### Plots simulated and reconstructed traces/FFTs using the true values for nu direction and shower energy
            fig, ax = plt.subplots(len(use_channels), 2, sharex='col', figsize=(15, 7.5))
            for iCh, channel in enumerate(station.iter_channels(use_channels=use_channels)):
                trace = channel.get_trace()
                ax[iCh][0].plot(trace, label = 'voltage trace', color='blue')
                if hilbert:
                    ax[iCh][0].plot(abs(signal.hilbert(trace)),'--', color='blue')
                ax[iCh][1].plot(ff/units.MHz, abs(fft.time2freq(trace, 1/dt)), color='blue')
                ax[iCh][0].plot(traces[iCh], label = 'fit', color='orange')
                if hilbert:
                    ax[iCh][0].plot(abs(signal.hilbert(traces[iCh])),'--', color='orange')
                ax[iCh][1].plot(ff/units.MHz, abs(fft.time2freq(traces[iCh], 1/dt)),color='orange')
                ax[iCh][1].legend(loc='upper right', fontsize = 'xx-small')
            ax[0][0].legend(loc='upper left', fontsize = 'xx-small')
            ax[iCh][0].set_xlabel("time [ns]")
            ax[iCh][1].set_xlabel("frequecy [MHz]")
            fig.tight_layout()
            fig.savefig(save)



        if(len(use_channels) == 0):
            raise Exception("Cannot reconstruct neutrino direction and shower energy without any channels to reference")
        if(use_bandpass_filter and (len(use_channels) != len(passband_low) or len(use_channels) != len(passband_high) )):
            raise Exception("Length of passband maps do not match number of use channels. Must instantiate a passband for each channel individually both for lowpass and highpass filters. This allows channels to use different band pass filters.")


        use_channels.sort()
        station_id = station.get_id()
        noise_RMS = det.get_noise_RMS(station_id, 0) #assume noise is the same in all channels

        # n_expand will multiply the time window of the reconstructed trace by n.
        # This allows for this module to fit on any of the ray tracing solutions between the reconstructed traces and the true traces.
        # This is especially important for Moore's Bay where the ray tracing solutions can be microseconds apart. Setting n_expand to 50 for Moore's Bay suffices.
        n_expand = n_samples_multiplication_factor

        # assume all channles have same number of samples and sampling rate
        first_channel = station.get_channel(use_channels[0])
        n_samples = first_channel.get_number_of_samples() * n_expand
        sampling_rate = first_channel.get_sampling_rate()

        dt = 1./sampling_rate
        T = n_samples * dt  # trace length
        tt = np.arange(0, T, dt)
        ff = np.fft.rfftfreq(n_samples, dt)

        if use_MC and (station.get_sim_station() is not None):
            # Get the ray tracing ids for simulated values
            # Note they are not ordered in any particular way
            sim_shower = evt.get_first_sim_shower()
            channels_with_existing_sol = set()
            for i_efield, efield in enumerate(station.get_sim_station().get_electric_fields()):
                if efield.get_channel_ids()[0] in use_channels:
                    channels_with_existing_sol.add(efield.get_channel_ids()[0])

            # Sometimes some events do not have ray tracing solutions to all channels that is requested with use_channels.
            # This block of code catches that so the following does not try to access data for channels which does not exist
            use_channels_tmp = []
            for channel in channels_with_existing_sol:
                use_channels_tmp.append(channel)
            use_channels = use_channels_tmp
            use_channels.sort()

            nu_zenith_sim = sim_shower[shp.zenith]
            nu_azimuth_sim = sim_shower[shp.azimuth]
            shower_energy_sim = sim_shower[shp.energy] #inelasticity * nu_energy

            vertex_position = sim_shower[shp.vertex]

        else:
            vertex_position = station.get_parameter(stnp.nu_vertex)


        n_index = icemodel.get_index_of_refraction(vertex_position)

        n_antennas = len(use_channels)
        antenna_orientations = np.zeros((n_antennas, 4))
        antenna_positions = np.zeros((n_antennas, 3))
        cable_delays = np.zeros((n_antennas, 1))
        antenna_patterns = []
        for iA, iCh in enumerate(use_channels):
            antenna_orientations[iA] = det.get_antenna_orientation(station_id,iCh)
            antenna_positions[iA] = det.get_relative_position(station_id,iCh)
            antenna_model = det.get_antenna_model(station_id, iCh, antenna_orientations[iA][0])
            antenna_patterns.append(antennapattern.AntennaPatternProvider().load_antenna_pattern(antenna_model))
            cable_delays[iA] = det.get_cable_delay(station.get_id(), iCh)

        # Used to initial numpy array data structures.
        maxNumRayTracingSolPerChan = 2
        n_reflections = 0
        if(attenuation_model == "MB1"):
            maxNumRayTracingSolPerChan = 6
            n_reflections = 1

        n_ray_tracing_solutions = np.zeros(n_antennas, dtype=np.int)
        launch_vectors = np.zeros((n_antennas, maxNumRayTracingSolPerChan, 3))
        receive_vectors = np.zeros((n_antennas, maxNumRayTracingSolPerChan, 3))
        travel_time = np.zeros((n_antennas, maxNumRayTracingSolPerChan))
        travel_distance = np.zeros((n_antennas, maxNumRayTracingSolPerChan))
        attenuation = np.zeros((n_antennas, maxNumRayTracingSolPerChan, len(ff)))
        focusing = np.zeros((n_antennas, maxNumRayTracingSolPerChan, 1))
        reflection_coefficients_theta = np.ones((n_antennas, maxNumRayTracingSolPerChan), dtype=np.complex)
        reflection_coefficients_phi = np.ones((n_antennas, maxNumRayTracingSolPerChan), dtype=np.complex)
        travel_time_min = float('inf')
        for iA, position in enumerate(antenna_positions):
            r = ray.ray_tracing(icemodel, attenuation_model=attenuation_model, n_frequencies_integration=25, n_reflections=n_reflections)
            r.set_start_and_end_point(vertex_position, position)
            r.find_solutions()
            n_ray_tracing_solutions[iA] = min(r.get_number_of_solutions(),maxNumRayTracingSolPerChan)
            for iS in range(r.get_number_of_solutions()):
                launch_vectors[iA, iS] = r.get_launch_vector(iS)
                receive_vectors[iA, iS] = r.get_receive_vector(iS)
                travel_time[iA, iS] = r.get_travel_time(iS)
                travel_time_min = min(travel_time_min,r.get_travel_time(iS))
                travel_distance[iA, iS] = r.get_path_length(iS)
                attenuation[iA, iS] = r.get_attenuation(iS, ff)
                focusing[iA, iS] = r.get_focusing(iS, 1*units.cm)

                # calculate the Fresnel reflection coefficients
                r_theta = None
                r_phi = None
                i_reflections = r.get_results()[iS]['reflection']
                zenith_reflections = np.atleast_1d(r.get_reflection_angle(iS))  # lets handle the general case of multiple reflections off the surface (possible if also a reflective bottom layer exists)
                n_surface_reflections = np.sum(zenith_reflections != None)
                for zenith_reflection in zenith_reflections:  # loop through all possible reflections
                    if(zenith_reflection is None):  # skip all ray segments where not reflection at surface happens
                        continue
                    r_theta = geo_utl.get_fresnel_r_p(
                        zenith_reflection, n_2=1., n_1=icemodel.get_index_of_refraction([position[0], position[1], -1 * units.cm]))
                    r_phi = geo_utl.get_fresnel_r_s(
                        zenith_reflection, n_2=1., n_1=icemodel.get_index_of_refraction([position[0], position[1], -1 * units.cm]))

                    reflection_coefficients_theta[iA, iS] *= r_theta
                    reflection_coefficients_phi[iA, iS] *= r_phi
                if(i_reflections > 0):  # take into account possible bottom reflections (only relevant for Moore's Bay)
                    # each reflection lowers the amplitude by the reflection coefficient and introduces a phase shift
                    reflection_coefficient = icemodel.reflection_coefficient ** i_reflections
                    phase_shift = (i_reflections * icemodel.reflection_phase_shift) % (2 * np.pi)
                    # we assume that both efield components are equally affected
                    reflection_coefficients_theta[iA, iS] *= reflection_coefficient * np.exp(1j * phase_shift)
                    reflection_coefficients_phi[iA, iS] *= reflection_coefficient * np.exp(1j * phase_shift)


        if plot_traces_with_true_input:
            save = "tracesWithTrueInput_"+str(evt.get_run_number())+".png"
            plotSimulatedVersusReconstructedTraces(nu_zenith_sim, nu_azimuth_sim, shower_energy_sim, save)
            plt.show()
            plt.close()



        if use_MC:
            # Takes roughly 20 minutes to perform
            results = opt.brute(minimizer, ranges=(slice(nu_zenith_sim - np.deg2rad(10.0),nu_zenith_sim + np.deg2rad(10.0), np.deg2rad(1.0)),slice(nu_azimuth_sim - np.deg2rad(10.0), nu_azimuth_sim + np.deg2rad(10.0), np.deg2rad(1.0)),slice(14, 19, 0.1)),
                            full_output=True, finish=opt.fmin)  # slow but does the trick
        else:
            # The number of inputs here will be really really slow to process but how many true neutrino events will we actually get? Takes about 3 days. Note that it is naively scanning the entire range of zenith and azimuth and energies
            results = opt.brute(minimizer, ranges=(slice(np.deg2rad(0.0),np.deg2rad(180.0), np.deg2rad(1.0)),slice(np.deg2rad(0.0),np.deg2rad(360.0), np.deg2rad(1.0)),slice(14, 21, 0.1)),
                            full_output=True, finish=opt.fmin)  # slow but does the trick


        nu_zenith = results[0][0]
        nu_azimuth = results[0][1]
        shower_energy = results[0][2]

        station.set_parameter(stnp.nu_zenith,nu_zenith)
        station.set_parameter(stnp.nu_azimuth,nu_azimuth)
        station.set_parameter(stnp.shower_energy,10**shower_energy)

        # store the viewing angles for the reconstructed properties of the neutrino
        traces, viewing_angles_reco = minimizer([nu_zenith, nu_azimuth, shower_energy], minimizer=False)
        station.set_parameter(stnp.viewing_angles,viewing_angles_reco)


        if debug:
            #### Plots simulated and reconstructed traces/FFTs along with labeling each with the direct ray solutions reconstructed viewing angle
            save = "traces_"+str(evt.get_run_number())+".png"
            plotSimulatedVersusReconstructedTraces(nu_zenith, nu_azimuth, 10**shower_energy, save)
            print('Finished making traces figure')


            def hist2DHelper(ax, xvals, yvals, zmesh, begX0, endX, begY0, endY, xscale="linear", yscale="linear"):
                xrange = xvals
                xrange = np.append(xrange, endX)
                xrange = np.append(begX0, xrange)
                yrange = yvals
                yrange = np.append(yrange, endY)
                yrange = np.append(begY0, yrange)
                php.get_histogram2d(x=xvals, y=yvals, z=zmesh,
                    bins=[xrange, yrange], range=None,
                    xscale=xscale, yscale=yscale, cscale="linear",
                    normed=False, cmap='viridis', clim=(np.min(zmesh), np.max(zmesh)),
                    ax1=ax, grid=True, shading='flat', colorbar={},
                    cbi_kwargs={'orientation': 'vertical'},
                    xlabel="", ylabel="", clabel="", title="",
                    fname="hist2d.png")

            #### Create spacings for the 3 variables in the minimizer, nu zenith, nu azimuth, and shower energy. Used in 2d hist plots
            spaceingZenith = np.deg2rad(1.0)
            begZenith = nu_zenith - np.deg2rad(20.0)
            endZenith = nu_zenith + np.deg2rad(20.0)
            begZenith0 = nu_zenith - np.deg2rad(20.0) - spaceingZenith
            zenith = np.arange(begZenith, endZenith, spaceingZenith)

            spaceingAzimuth = np.deg2rad(1.0)
            begAzimuth = nu_azimuth - np.deg2rad(20.0)
            endAzimuth = nu_azimuth + np.deg2rad(20.0)
            begAzimuth0 = nu_azimuth - np.deg2rad(20.0) - spaceingAzimuth
            azimuth = np.arange(begAzimuth, endAzimuth, spaceingAzimuth)

            spaceingEnergy = 0.1
            numEnergyPoints = 30
            begEnergy = 14
            endEnergy = 17.1
            begEnergy0 = begEnergy - spaceingEnergy
            energy = np.arange(begEnergy, endEnergy, spaceingEnergy)

            ################### Plots the minimizer for varying neutrino zenith and azimuth
            zmesh = np.zeros((len(azimuth),len(zenith)))
            # X -> collumn, Y -> row
            for iC, c in enumerate(zenith):
                for iR, r in enumerate(azimuth):
                    zmesh[iR][iC] = np.log(minimizer([c, r, shower_energy]))

            fig2, ax2 = plt.subplots(1, 1, sharex=False, figsize=(15, 7.5))
            hist2DHelper(ax2, np.rad2deg(zenith), np.rad2deg(azimuth), zmesh, begZenith0, endZenith, begAzimuth0, endAzimuth)
            ax2.axhline(np.rad2deg(nu_azimuth_sim), color = 'red',label = 'simulated values', linewidth = 3)
            ax2.axhline(np.rad2deg(nu_azimuth), color = 'orange',label = 'reconstructed values', linewidth = 3)
            ax2.axvline(np.rad2deg(nu_zenith_sim), color = 'red', linewidth = 3)
            ax2.axvline(np.rad2deg(nu_zenith), color = 'orange', linewidth = 3)
            ax2.legend(fontsize = 'xx-small')
            ax2.set_xlabel("Neutrino Zenith direction [degrees]")
            ax2.set_ylabel("Neutrino Azimuth direction [degrees]")
            save = "AZ_"+str(evt.get_run_number())+".png"
            fig2.savefig(save)

            ################### Plots the minimizer for varying neutrino zenith and shower energy
            zmesh = np.zeros((len(zenith),len(energy)))
            # X -> collumn, Y -> row
            for iC, c in enumerate(energy):
                for iR, r in enumerate(zenith):
                    zmesh[iR][iC] = np.log(minimizer([r, nu_azimuth, c]))

            fig3, ax3 = plt.subplots(1, 1, sharex=False, figsize=(15, 7.5))
            hist2DHelper(ax3, energy, np.rad2deg(zenith), zmesh, begEnergy0, endEnergy, begZenith0, endZenith)
            ax3.axhline(np.rad2deg(nu_zenith_sim), color = 'red',label = 'simulated values', linewidth = 3)
            ax3.axhline(np.rad2deg(nu_zenith), color = 'orange',label = 'reconstructed values', linewidth = 3)
            ax3.axvline(np.log10(shower_energy_sim), color = 'red', linewidth = 3)
            ax3.axvline(shower_energy, color = 'orange', linewidth = 3)
            ax3.legend(fontsize = 'xx-small')
            ax3.set_xlabel("log10 shower energy [eV]")
            ax3.set_ylabel("Neutrino Zenith direction [degrees]")
            save = "ZE_"+str(evt.get_run_number())+".png"
            fig3.savefig(save)

            ################### Plots the minimizer for varying neutrino azimuth and shower energy
            zmesh = np.zeros((len(azimuth),len(energy)))
            # X -> collumn, Y -> row
            for iC, c in enumerate(energy):
                for iR, r in enumerate(azimuth):
                    zmesh[iR][iC] = np.log(minimizer([nu_zenith, r, c]))

            fig4, ax4 = plt.subplots(1, 1, sharex=False, figsize=(15, 7.5))
            hist2DHelper(ax4, energy, np.rad2deg(azimuth), zmesh, begEnergy0, endEnergy, begAzimuth0, endAzimuth)
            ax4.axhline(np.rad2deg(nu_azimuth_sim), color = 'red',label = 'simulated values', linewidth = 3)
            ax4.axhline(np.rad2deg(nu_azimuth), color = 'orange',label = 'reconstructed values', linewidth = 3)
            ax4.axvline(np.log10(shower_energy_sim), color = 'red', linewidth = 3)
            ax4.axvline(shower_energy, color = 'orange', linewidth = 3)
            ax4.legend(fontsize = 'xx-small')
            ax4.set_xlabel("log10 shower energy [eV]")
            ax4.set_ylabel("Neutrino Azimuth direction [degrees]")
            save = "AE_"+str(evt.get_run_number())+".png"
            fig4.savefig(save)

            plt.show()
            plt.close('all')


    def end(self):
        pass
