
"""
A module for reconstruction of the Electric Field
for LOFAR. Based on a method by Welling et al. (2021)
https://arxiv.org/abs/2102.00258

Author: Karen Terveer

"""

import NuRadioReco.detector.antennapattern
from NuRadioReco.utilities import units
from NuRadioReco.detector.LOFAR import analog_components
from . import iftModels as models
import nifty8.re as jft # type: ignore
import numpy as np
import scipy

from jax import random, vmap # type: ignore
import matplotlib.pyplot as plt

from NuRadioReco.modules.base.module import register_run
import numpy as np
import NuRadioReco.detector.antennapattern
from NuRadioReco.framework.parameters import stationParameters, showerParameters

import logging

def phi_pol_calc(zenith_vel, azimuth_vel, zenith_unit=90, azimuth_unit=0, zenith_mag=23.5, azimuth_mag=1.5):

    """
    Calculate the angle between the cross product of the geomagnetic field vector and the direction of the cosmic ray,
    and the arbitrary unit vector.
    
    Parameters:
    -----------
    zenith_mag : float, optional
        Zenith angle of the geomagnetic field in degrees. Default for LOFAR is 23.5.
    azimuth_mag : float, optional
        Azimuth angle of the geomagnetic field in degrees. Default for LOFAR is 1.5.
    zenith_vel : float
        Zenith angle of the velocity vector in degrees.
    azimuth_vel : float
        Azimuth angle of the velocity vector in degrees.
    zenith_unit : float
        Zenith angle of the arbitrary unit vector in degrees.
    azimuth_unit : float
        Azimuth angle of the arbitrary unit vector in degrees.
    
    Returns:
    --------
    float
        The angle between the cross product vector and the unit vector in radians.
    """

    # Convert spherical coordinates (zenith, azimuth) to Cartesian coordinates (x, y, z)
    def spherical_to_cartesian(zenith, azimuth):
        zenith_rad = np.deg2rad(zenith)
        azimuth_rad = np.deg2rad(azimuth)
        x = np.sin(zenith_rad) * np.cos(azimuth_rad)
        y = np.sin(zenith_rad) * np.sin(azimuth_rad)
        z = np.cos(zenith_rad)
        return np.array([x, y, z])
    
    # Convert geomagnetic field vector, velocity vector, and unit vector to Cartesian coordinates
    B = spherical_to_cartesian(zenith_mag, azimuth_mag)
    V = spherical_to_cartesian(zenith_vel, azimuth_vel)
    U = spherical_to_cartesian(zenith_unit, azimuth_unit)
    
    # Calculate cross product
    cross_product = np.cross(B, V)
    
    # Normalize the cross product vector to get the direction only
    cross_product_norm = cross_product / np.linalg.norm(cross_product)
    
    # Calculate the dot product between the normalized cross product and the unit vector
    dot_product = np.dot(cross_product_norm, U)
    
    # Calculate the angle between the vectors using the arccosine of the dot product
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))

    return angle_rad



class iftEfieldReco:
    """
    This class performs computation for voltage trace to electric field per channel.

    Attributes
    ----------
    logger : logging.Logger
        The logger object for logging messages.
    sim : bool
        Flag indicating if simulation or real data is being processed.

    Methods
    -------
    __init__()
        Initializes the iftEfieldReco object.
    begin()
        Initializes the antenna_provider attribute.
    run(evt, station, det, debug=False)
        Performs computation for voltage trace to electric field per channel.

    """

    def __init__(self):
        """
        Initializes the iftEfieldReco object.

        """
        self.logger = logging.getLogger('NuRadioReco.LOFAR.iftEfieldReco')
        self.begin()
        self.sim=False

    def begin(self):
        """
        Initializes the antenna_provider attribute.

        """
        self.antenna_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

    @register_run()
    def run(self, evt, station, det, debug=True):
        """
        Performs computation for voltage trace to electric field per channel.

        This method provides a deconvoluted (electric field) trace for each channel from the stations input voltage traces.

        Parameters
        ----------
        evt : event data structure
            The event data structure.
        station : station data structure
            The station data structure.
        det : detector object
            The detector object.
        debug : bool, optional
            Flag to enable debug mode. Default is False.

        """


        station_id = station.get_id()
        self.logger.debug("event {}, station {}".format(evt.get_id(), station_id))

        # Check if simulation data is available TODO write full simulation code
        if station.get_sim_station() is not None and station.get_sim_station().has_parameter(stationParameters.zenith):
            zenith = station.get_sim_station()[stationParameters.zenith]
            azimuth = station.get_sim_station()[stationParameters.azimuth]
            self.logger.info("This seems to be a simulation. Welcome to MonteCarlo dreamland!")
            self.sim = True
            sim_station = station.get_sim_station()
            efield = [field for field in sim_station.get_electric_fields() if
                  np.max(np.abs(field.get_trace())) >= 1.0e-10]

            sim_channels = [field.get_channel_ids() for field in sim_station.get_electric_fields()]


        else:
            self.logger.debug("Using reconstructed angles as no simulation present")
            zenith = evt.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.zenith) / units.deg
            azimuth = evt.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.azimuth) / units.deg


        # Calculate phi polarization # TODO improve direction fitting somehow
        phi_pol = {"mean": phi_pol_calc(zenith, azimuth), "std": 0.5}

        if self.sim==True:
            channels = sim_channels
            e_index = 0
        else:
            channels = station.iter_channels()

        for iCh, channel in enumerate(channels): # TODO loop only over non-empty channels

            # get channel ids and channel group ID. Check whether current channel ID and group ID match, and if not, skip iteration.
            # for E-Field reconstruction, we need two channels within one channel group which correspond to the same E-Field

            if self.sim==True:
                channel = channel[0]

            channel = station.get_channel(channel)
            channel_id = channel.get_id()
            channel_id_odd = channel_id + 1
            channel_group_id = channel.get_group_id()
            

            if str(channel_group_id) != str(channel_id):
                print("Channel group ID does not match channel ID")
                continue

            print("Channel group ID matches channel ID")
            channel_odd = evt.get_station(station_id).get_channel(channel_id_odd)
            channel_even = evt.get_station(station_id).get_channel(channel_id)

            sampling_rate = station.get_channel(channel_id).get_sampling_rate()

            # prepare traces for processing 

            trace0 = channel.get_trace()
            trace1 = channel_odd.get_trace()
            time0 = channel.get_times()
            time1 = channel_odd.get_times()

            if self.sim==True and debug==True:
                
                import matplotlib.pyplot as plt
                plt.plot(time0, trace0, 'b-', label='trace X')
                efield_channels = efield[iCh]
                efield_trace = efield_channels.get_trace()
                efield_times = efield_channels.get_times()
                efield_times = efield_times + time0[len(time0)//2] - efield_times[len(efield_times)//2]

                plt.plot(time1, trace1, 'g-', label='trace Y')
                plt.plot(time0, trace0, 'b-', label='trace X')
                plt.plot(efield_times, efield_trace[1], 'r-', label='efield Theta')
                plt.plot(efield_times, efield_trace[2], 'y-', label='efield Phi')
                plt.legend()
                plt.show()

                efield_spec = efield_channels.get_frequency_spectrum()
                efield_freqs = efield_channels.get_frequencies()
                plt.plot(efield_freqs, np.abs(efield_spec[1]), 'r-', label='efield Theta')
                plt.plot(efield_freqs, np.abs(efield_spec[2]), 'y-', label='efield Phi')
                plt.xlim(0.03,0.08)
                plt.legend()
                plt.show()


                

            assert np.allclose(time0, time1)
            dt = time0[1:] - time0[:-1]
            assert np.allclose(dt, dt[0]*np.ones_like(dt))

            # find pulse as maximum of trace TODO implement more sophisticated pulse finding

            #apply hilbert envelope to pulse to find maximum

            trace_hilbert = np.abs(scipy.signal.hilbert(trace0))
            pulse_max = np.max(trace_hilbert)

            pulse_max_ind = np.argmax(trace_hilbert)#-int(50/dt[0])
            pulse_len = 200
            start_time = time0[pulse_max_ind - int(pulse_len/4)]
            end_time = time0[pulse_max_ind + int(3*pulse_len/4)]

            # Process traces

            data0, noise0 = models.process_trace(
                trace1, time1, start_time=start_time, end_time=end_time, cut_fraq=0.1
            )
            data1, noise1 = models.process_trace(
                trace0, time0, start_time=start_time, end_time=end_time, cut_fraq=0.1
            )

            data = np.stack([data0, data1], axis=-1)
            noise = np.stack([noise0, noise1], axis=-1)
            time = np.arange(data.shape[0]) * dt[0] + start_time

            # Calculate noise covariance matrix using noise traces
            nCov = models.stationary_noise(noise, debug=debug)

            # initialise prior parameters
            mt0 = -0.0062834 * (pulse_max - 85)# - 2.6
            m_phi = {"mean": mt0, "std": 2}

            # generate signal model
            signal = models.EFieldModel(times=time, params_m_phi=m_phi, params_phi_pol=phi_pol)  

            # import system response for signal receiving angle # TODO is LORA angle best here?
            system_res = np.zeros((len(signal.freqs), 2, 2), dtype=complex)

            cab_len = det.get_cable_type_and_length(station.get_id(), channel_id)[1]

            antenna_orientation = det.get_antenna_orientation(station.get_id(), channel_even.get_id())
            antenna_pattern = self.antenna_provider.load_antenna_pattern(
                                det.get_antenna_model(station.get_id(), channel_even.get_id()))



            cable_response = analog_components.get_cable_response(signal.freqs * units.MHz, cable_length=int(cab_len))
            RCU_response = analog_components.get_RCU_response(signal.freqs * units.MHz)
            antenna_response = antenna_pattern.get_antenna_response_vectorized(signal.freqs * units.MHz, zenith, azimuth, *antenna_orientation)

            system_response = np.power(10.0, (cable_response['attenuation'] / 10.0)) * np.power(10.0, (RCU_response['gain'] / 10.0))

            system_res[:, 0, 0] = antenna_response["theta"] * system_response
            system_res[:, 0, 1] = antenna_response["phi"] * system_response


            antenna_orientation = det.get_antenna_orientation(station.get_id(), channel_odd.get_id())
            antenna_pattern = self.antenna_provider.load_antenna_pattern(
                                det.get_antenna_model(station.get_id(), channel_odd.get_id()))
            

            cable_response = analog_components.get_cable_response(signal.freqs * units.MHz, cable_length=int(cab_len))
            RCU_response = analog_components.get_RCU_response(signal.freqs * units.MHz)
            antenna_response = antenna_pattern.get_antenna_response_vectorized(signal.freqs * units.MHz, zenith, azimuth, *antenna_orientation)


            system_res[:, 1, 0] = antenna_response["theta"] * system_response
            system_res[:, 1, 1] = antenna_response["phi"] * system_response
 
            signal_response = models.AntennaResponse(signal, system_res)

            seed = 42
            key = random.PRNGKey(seed)


            # construct likelihood

            likelihood = jft.Gaussian(
                data=data, noise_cov_inv=nCov.noise_cov_inv, noise_std_inv=nCov.noise_std_inv
            )

            nll = likelihood.amend(signal_response)


            problem_size = min(jft.size(nll.likelihood.data), jft.size(nll.domain))

            # initialise the optimization
            n_vi_iterations = 15
            delta = 1e-5
            absdelta = delta * problem_size
            n_samples = 6

            key, k_i, k_o = random.split(key, 3)
            # NOTE, changing the number of samples always triggers a resampling even if
            # `resamples=False`, as more samples have to be drawn that did not exist before.
            samples, state = jft.optimize_kl(
                nll,
                jft.Vector(nll.init(k_i)),
                n_total_iterations=n_vi_iterations,
                n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
                # Source for the stochasticity for sampling
                key=k_o,
                # Names of parameters that should not be sampled but still optimized
                # can be specified as point_estimates (effectively we are doing MAP for
                # these degrees of freedom).
                # point_estimates=("cfax1flexibility", "cfax1asperity"),
                # Arguments for the conjugate gradient method used to drawing samples from
                # an implicit covariance matrix
                draw_linear_kwargs=dict(
                    cg_name="SL",
                    cg_kwargs=dict(absdelta=1E-10, maxiter=problem_size),
                ),
                # Arguements for the minimizer in the nonlinear updating of the samples
                nonlinearly_update_kwargs=dict(
                    minimize_kwargs=dict(
                        name="SN",
                        xtol=delta,
                        cg_kwargs=dict(name=None, maxiter=problem_size),
                        maxiter=3,
                    )
                ),
                # Arguments for the minimizer of the KL-divergence cost potential
                kl_kwargs=dict(
                    minimize_kwargs=dict(
                        name="M", absdelta=absdelta, cg_kwargs=dict(name=None, maxiter=problem_size), maxiter=25
                    )
                ),
                sample_mode="linear_sample", # "nonliner_sample"
                odir="results_intro",
                kl_map=vmap,
                residual_map='lmap', # if "nonliner_sample" replace with 'lmap'
                resume=False,
            )

            # get results: voltage traces, electric field traces, and frequency traces
            post_sr_mean_response, post_sr_std_response = jft.mean_and_std(
                tuple(signal_response(s) for s in samples), correct_bias=True
            ) 
            post_sr_mean, post_sr_std = jft.mean_and_std(
                tuple(signal(s) for s in samples), correct_bias=True
            ) 
            post_sr_mean_freq, post_sr_std_freq = jft.mean_and_std(
                tuple(signal.trace_freq(s) for s in samples), correct_bias=True
            ) 

            # convert to correct units TODO check whether this is really correct
            post_sr_mean_response = post_sr_mean_response * units.V / units.m
            post_sr_std_response = post_sr_std_response * units.V / units.m
            post_sr_mean = post_sr_mean * units.V / units.m
            post_sr_std = post_sr_std * units.V / units.m
            post_sr_mean_freq = post_sr_mean_freq * units.V / units.m
            post_sr_std_freq = post_sr_std_freq * units.V / units.m   

            # reconstruct efield with original trace size, apply hanning filter to prevent windowing effects
            rec_efield = np.zeros((3,len(trace0)), dtype=np.complex128)
            window = np.hanning(len(post_sr_mean[:,0]))

            post_sr_mean_filtered = np.zeros_like(post_sr_mean)
            post_sr_mean_filtered[:,0] = post_sr_mean[:,0] * window 
            post_sr_mean_filtered[:,1] = post_sr_mean[:,1] * window 

            post_sr_std_filtered = np.zeros_like(post_sr_std)
            post_sr_std_filtered[:,0] = post_sr_std[:,0] * window
            post_sr_std_filtered[:,1] = post_sr_std[:,1] * window

            rec_efield[1:,pulse_max_ind - len(post_sr_mean[:,0])//2:pulse_max_ind + len(post_sr_mean[:,0])//2] = np.swapaxes(post_sr_mean_filtered,0,1)

            # set efield in even channel
            efield_even = NuRadioReco.framework.electric_field.ElectricField([channel_id])
            efield_even.set_trace(rec_efield, sampling_rate)


            rec_efield = np.zeros((3,len(trace0)), dtype=np.complex128)
            window = np.hanning(len(post_sr_mean[:,0]))
            rec_efield[1:,pulse_max_ind - len(post_sr_mean[:,0])//2:pulse_max_ind + len(post_sr_mean[:,0])//2] = np.swapaxes(post_sr_mean_filtered,0,1)

            # set efield in odd channel
            efield_odd = NuRadioReco.framework.electric_field.ElectricField([channel_id_odd])
            efield_odd.set_trace(rec_efield, sampling_rate)

            if debug==True:

                # some pretty plots

                import matplotlib.pyplot as plt
                times = np.arange(len(post_sr_mean[:,0])) * dt[0] + start_time

                # Plotting post-signal response mean and standard deviation
                fig, axs = plt.subplots(1, 3, figsize=(18, 5))


                axs[0].plot(time0[int(pulse_max_ind - pulse_len/4):int(pulse_max_ind + 3*pulse_len/4)], trace0[int(pulse_max_ind - pulse_len/4):int(pulse_max_ind + 3*pulse_len/4)], 'bisque', label='trace Y', alpha=1)
                axs[0].plot(time1[int(pulse_max_ind - pulse_len/4):int(pulse_max_ind + 3*pulse_len/4)], trace1[int(pulse_max_ind - pulse_len/4):int(pulse_max_ind + 3*pulse_len/4)], 'lightsteelblue', label='trace X',alpha=0.9)
                axs[0].plot(times,post_sr_mean_response[:, 0], 'slateblue', alpha=1, label="reco X")
                axs[0].fill_between(times, post_sr_mean_response[:, 0] + post_sr_std_response[:, 0], post_sr_mean_response[:, 0] - post_sr_std_response[:, 0], color='slateblue', alpha=0.3)
                axs[0].plot(times,post_sr_mean_response[:, 1], 'darkorange', alpha=1, label="reco Y")
                axs[0].fill_between(times, post_sr_mean_response[:, 1] + post_sr_std_response[:, 1], post_sr_mean_response[:, 1] - post_sr_std_response[:, 1], color='darkorange', alpha=0.3)

                axs[0].set_xlabel('Time (ns)')
                axs[0].set_ylabel('Amplitude')
                axs[0].set_title(f'Voltage Trace Ch. Group {channel_group_id}')
                axs[0].legend()


                axs[1].plot(times,post_sr_mean_filtered[:, 0], 'coral', alpha=0.8, label="phi pol")
                axs[1].fill_between(times, post_sr_mean_filtered[:, 0] + post_sr_std_filtered[:, 0], post_sr_mean_filtered[:, 0] - post_sr_std_filtered[:, 0], color='coral', alpha=0.3)
                axs[1].plot(times,post_sr_mean_filtered[:, 1], 'lightseagreen', alpha=0.8, label="theta pol")
                axs[1].fill_between(times, post_sr_mean_filtered[:, 1] + post_sr_std_filtered[:, 1], post_sr_mean_filtered[:, 1] - post_sr_std_filtered[:, 1], color='lightseagreen', alpha=0.3)
                axs[1].legend()
                axs[1].set_xlabel('Time (ns)')
                axs[1].set_ylabel('Amplitude')
                axs[2].set_title('Electric Field Trace Reco')

                # Plotting frequency spectrum
                
                axs[2].plot(signal.freqs,np.abs(post_sr_mean_freq[:, 0]), 'coral', alpha=0.8, label="phi pol")
                axs[2].fill_between(signal.freqs,np.abs(post_sr_std_freq[:, 0]) + np.abs(post_sr_mean_freq[:, 0]), np.abs(post_sr_mean_freq[:, 0]) - np.abs(post_sr_std_freq[:, 0]), color='coral', alpha=0.3)
                #axs[1].fill_between(range(len(post_sr_mean_filtered[:, 0])), post_sr_mean_filtered[:, 0] + post_sr_mean_filtered[:, 0], post_sr_mean_filtered[:, 0] - post_sr_mean_filtered[:, 0], color='m', alpha=0.3)
                axs[2].plot(signal.freqs,np.abs(post_sr_mean_freq[:, 1]), 'lightseagreen', alpha=0.8, label="theta pol")
                axs[2].fill_between(signal.freqs, np.abs(post_sr_mean_freq[:, 1]) + np.abs(post_sr_std_freq[:, 1]), np.abs(post_sr_mean_freq[:, 1]) - np.abs(post_sr_std_freq[:, 1]), color='lightseagreen', alpha=0.3)
                #axs[1].fill_between(range(len(post_sr_mean_filtered[:, 1])), post_sr_mean_filtered[:, 1] + post_sr_mean_filtered[:, 1], post_sr_mean_filtered[:, 1] - post_sr_mean_filtered[:, 1], color='k', alpha=0.3)
                axs[2].legend()
                axs[2].set_xlim(30, 80)

                index = np.where(signal.freqs == 30)[0]
                max_y = max(np.abs(post_sr_std_freq[index, 0]) + np.abs(post_sr_mean_freq[index, 0]), np.abs(post_sr_std_freq[index, 1]) + np.abs(post_sr_mean_freq[index, 1]))

                axs[2].set_ylim(0, max_y)
                axs[2].set_xlabel('Freq (MHz)')
                axs[2].set_ylabel('Amplitude')
                axs[2].set_title('Electric Field Spectrum Reco')
                

                plt.tight_layout()
                plt.show()


    def end(self):
        pass
