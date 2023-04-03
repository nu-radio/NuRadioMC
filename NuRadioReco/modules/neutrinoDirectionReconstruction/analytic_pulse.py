from functools import lru_cache
from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalGen import askaryan as signalgen
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from radiotools import coordinatesystems as cstrans
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.electric_field import ElectricField
from NuRadioReco.utilities import trace_utilities, bandpass_filter
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator
import NuRadioReco.modules.io.eventReader
from NuRadioReco.framework.parameters import stationParameters as stnp
from radiotools import helper as hp
import numpy as np
import logging
import copy

logger = logging.getLogger("analytic_pulse")
logger.setLevel(logging.INFO)

from NuRadioReco.detector import antennapattern
eventreader = NuRadioReco.modules.io.eventReader.eventReader()

hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

class simulation():

	def __init__(self, template = False, vertex = [0,0,-1000], distances = None):
		"""
		Class used to return a modelled pulse

		This class is used to generate a modeled Askaryan pulse, and forward-fold it with
		the expected propagation effects and detector response, in order to use it in 
		a forward-folding fit.

		Other Parameters
		----------------
		template: bool, default: False
			Whether to use template-based fitting. This will currently
			raise a ``NotImplementedError``. If no template is used, the
			Askaryan model specified in ``self.simulation`` is used.
		
		"""
		self._template = template
		self.antenna_provider = antennapattern.AntennaPatternProvider()
		self._raytracing = dict()
		self._launch_vectors = None
		self._launch_vector = None
		self._viewingangle = None
		self._pol = None

		if self._template: ## I tried fitting with templates, but this is not better than fitting ARZ with Alvarez.
			raise NotImplementedError('Fit using templates has not been implemented yet')
			# if distances is None:
			# 	distances = [200, 300, 400, 500, 600, 700,800, 900, 1000, 1100,1143, 1200, 1300,1400,  1500, 1600, 1800, 2100, 2200, 2500, 3000, 300, 3500, 4000]
			# self._templates_path = '/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/inIceMCCall/Uncertainties/templates'
			# distance_event = np.sqrt(vertex[0]**2 + vertex[1]**2 + vertex[2]**2) ## assume station is at 000
			# print("distance event", distance_event)

			# R = distances[np.abs(np.array(distances) - distance_event).argmin()]
			# print("selected distance", R)
			# my_file = Path("/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/inIceMCCall/Uncertainties/templates/templates_{}.pkl".format(R, R))
			# if my_file.is_file():
			# 	f = NuRadioReco.utilities.io_utilities.read_pickle('{}'.format(my_file))
			# 	self._templates = f
			# 	self._templates_energies = f['header']['energies']
			# 	self._templates_viewingangles = f['header']['viewing angles']
			# 	self._templates_R = f['header']['R']
			# else:
			# 	## open look up tables
			# 	viewing_angles = np.arange(40, 70, .2)
			# 	self._header = {}
			# 	self._templates = { 'header': {'energies': 0, 'viewing angles': 0, 'R': 0, 'n_indes': 0} }
			# 	self._templates_viewingangles = []
			# 	for viewing_angle in viewing_angles:
			# 		if viewing_angle not in self._templates.keys():
			# 			try:
			# 				print("viewing angle", round(viewing_angle, 2))
			# 				f = NuRadioReco.utilities.io_utilities.read_pickle('{}/templates_ARZ2020_{}_1200.pkl'.format(self._templates_path,int(viewing_angle*10))) #### in future 10 should be removed.
			# 				if 1:#f['header']['R'] == 1500:
			# 					self._templates[np.round(viewing_angle, 2)] = f
			# 					self._templates_viewingangles.append(np.round(viewing_angle,2 ))
			# 					self._templates_R = f['header']['R']
			# 					print('done')
			# 					self._templates['header']['R'] = self._templates_R
			# 					self._templates['header']['energies'] = f['header']['energies']
			# 					print("HEADER", self._templates['header']['R'])

			# 			except:
			# 				print("template for viewing angle {} does not exist".format(int(viewing_angle*10)))
			# 	self._templates_energies = f['header']['energies']
			# 	print("template energies", self._template_energies)
			# 	self._templates['header']['viewing angles'] = self._templates_viewingangles
			# 	with open('{}/templates_343.pkl'.format(self._templates_path), 'wb') as f: #### this should be args.viewingangle/10
			# 		pickle.dump(self._templates, f)
		return

	def begin(
			self, det, station, use_channels, raytypesolution,
			reference_channel, passband = None,
			ice_model=None, att_model = None, askaryan_model = 'Alvarez2009',
			propagation_module="analytic", propagation_config=None,
			shift_for_xmax=False, systematics = None):
		""" 
		Initialize settings for the simulated pulse

		Used to specify the settings to use for the simulated pulse.
		
		Parameters
		----------
		det : Detector
		station : Station
			The Station for which to run the reconstruction. Used here only to
			obtain the appropriate sampling rate
		use_channels : list of ints
			the channel ids for which to simulate pulses.
		raytypesolution : int
			In addition to the simulated pulse, a launch vector and polarization
			can also be computed. These depend on the raytypesolution. Should generally
			be either 0 (lower/direct ray) or 1 (upper/reflected ray)
		reference_channel : int
			channel_id of the channel to use as the 'reference' channel
			(for launch vector, polarization, ...). Should be the upper Vpol in the phased
			array for an RNO-G-type detector
		passband : None | list | dict
			passband to use for the simulated pulses (after applying detector effects)
			If None, no passband is applied. If a list, should be of the form [highpass, lowpass].
			If a dict, the keys should be the channel ids, and the entries 
			should be lists of the form [highpass, lowpass].
		propagation_module : string, default: "analytic"
			Which propagation module to use for the propagation effects
		propagation_config : string | dict
			The propagation config to use for the reconstruction (usually the same 
			as used for the simulation, if applicable).
		
		
		Other Parameters
		----------------
		shift_for_xmax : bool, default: False
			If True, shift the viewing angle by an energy-dependent parametrization
			for the expected distance between neutrino vertex and shower maximum for 
			hadronic showers.
		ice_model : string | medium.IceModel instance | None (default)
			If not None, use this ice model rather than the one specified in the
			``propagation_config``
		att_model : string | None (default)
			If not None, use this attenuation model rather than the one specified
			in ``propagation_config``
		
		
		"""

		self._systematics = systematics
		self._ch_Vpol = reference_channel
		sim_to_data = True
		self._raytypesolution= raytypesolution
		channl = station.get_channel(use_channels[0])
		self._sampling_rate = channl.get_sampling_rate()
		time_trace = 200 #ns
		self._dt = 1./self._sampling_rate
		self._n_samples = int(time_trace * self._sampling_rate) ## templates are 800 samples long. The analytic models can be longer.
		self._first_iteration = True

		if ice_model is None:
			ice_model = propagation_config['propagation']['ice_model']
		if isinstance(ice_model, str):
			self._ice_model = medium.get_ice_model(ice_model)
		else:
			self._ice_model = ice_model
		if att_model is None:
			att_model = propagation_config['propagation']['attenuation_model']
		self._att_model = att_model
		self._askaryan_model = askaryan_model
		self._prop = propagation.get_propagation_module(propagation_module)
		self._prop_config = propagation_config
		self._shift_for_xmax = shift_for_xmax

        #### define bandpass filters and amplifier response
		self._ff = np.fft.rfftfreq(self._n_samples, self._dt)
		if not isinstance(passband, dict):
			passband = {channel_id:passband for channel_id in use_channels}

		self._h = dict()
		for channel_id in use_channels:
			passband_i = passband[channel_id] #TODO - move to single bandpass filter by default?
			if passband_i is None:
				self._h[channel_id] = 1
			else:
				filter_response_1 = bandpass_filter.get_filter_response(self._ff, [passband_i[0], 1150*units.MHz], 'butter', 8)
				filter_response_2 = bandpass_filter.get_filter_response(self._ff, [0*units.MHz, passband_i[1]], 'butter', 10)
				self._h[channel_id] = filter_response_1 * filter_response_2 

		self._amp = {}
		for channel_id in use_channels:
			self._amp[channel_id] = {}
			self._amp[channel_id] = det.get_amplifier_response(station_id=station.get_id(), channel_id=channel_id, frequencies=self._ff) #hardwareResponseIncorporator.get_filter(self._ff, station.get_id(), channel_id, det, sim_to_data = sim_to_data)

		pass


	def _calculate_polarization_vector(self, channel_id, iS):
		raytracing = self._raytracing
		polarization_direction = np.cross(raytracing[channel_id][iS]["launch vector"], np.cross(self._shower_axis, raytracing[channel_id][iS]["launch vector"]))
		polarization_direction /= np.linalg.norm(polarization_direction)
		cs = cstrans.cstrafo(*hp.cartesian_to_spherical(*raytracing[channel_id][iS]["launch vector"]))
		return cs.transform_from_ground_to_onsky(polarization_direction)

	def _theta_to_thetaprime(self, theta, xmax, R):
		b = R*np.sin(theta)
		a = R*np.cos(theta) - xmax
		return np.arctan2(b, a)

	def _xmax(self, energy):
		return 0.25 * np.log(energy) - 2.78

	@lru_cache(maxsize=128)
	def _raytracer(self, x1_x, x1_y, x1_z, x2_x, x2_y, x2_z):
		r = self._prop(self._ice_model, self._att_model, config=self._prop_config)
		r.set_start_and_end_point([x1_x, x1_y, x1_z], [x2_x, x2_y, x2_z])
		r.find_solutions()
		return copy.deepcopy(r)

	def simulation(
			self, det, station, vertex_x, vertex_y, vertex_z, nu_zenith,
			nu_azimuth, energy, use_channels,
			first_iteration = False,
			starting_values = False):
		"""
		Generate simulated pulses

		Generate the simulated pulses (usually, in order to compare them
		to our 'actual' voltage traces).

		Parameters
		----------
		det
		station
		vertex_x
		vertex_y
		vertex_z
		nu_zenith
		nu_azimuth
		energy
		use_channels

		Other Parameters
		----------------
		first_iteration
		starting_values

		Returns
		-------
		traces 
		timing
		launch_vector 
		viewingangle
		raytype
		pol
		
		"""
		ice = self._ice_model

		vertex = np.array([vertex_x, vertex_y, vertex_z])
		self._shower_axis = -1 * hp.spherical_to_cartesian(nu_zenith, nu_azimuth)
		n_index = ice.get_index_of_refraction(vertex)

		raytracing = self._raytracing # dictionary to store ray tracing properties
		if (self._first_iteration or first_iteration): # we run the ray tracer only on the first iteration

			launch_vectors = []
			polarizations = []
			viewing_angles = []
			chid = self._ch_Vpol
			x2 = det.get_relative_position(station.get_id(), chid) + det.get_absolute_position(station.get_id())
			# r = prop( ice, self._att_model, config=self._prop_config)
			# r.set_start_and_end_point(vertex, x2)

			# r.find_solutions()
			r = self._raytracer(*vertex, *x2)
			for iS in range(r.get_number_of_solutions()):
				if iS == self._raytypesolution:
					launch = r.get_launch_vector(iS)

					receive_zenith = hp.cartesian_to_spherical(*r.get_receive_vector(iS))[0]

			# logger.debug("Solving for channels {}".format(use_channels))
			for channel_id in use_channels:
				# logger.debug("Obtaining ray tracing info for channel {}".format(channel_id))
				raytracing[channel_id] = {}
				x2 = det.get_relative_position(station.get_id(), channel_id) + det.get_absolute_position(station.get_id())
				# r = prop( ice,self._att_model, config=self._prop_config)
				# r.set_start_and_end_point(vertex, x2)
				# r.find_solutions()
				r = self._raytracer(*vertex, *x2)
				if(not r.has_solution()):
					logger.warning(f"warning: no solutions for channel {channel_id}")
					continue

				# loop through all ray tracing solution

				for soltype in range(r.get_number_of_solutions()):

					iS = soltype

					raytracing[channel_id][iS] = {}
					self._launch_vector = r.get_launch_vector(soltype)
					raytracing[channel_id][iS]["launch vector"] = self._launch_vector
					R = r.get_path_length(soltype)
					raytracing[channel_id][iS]["trajectory length"] = R
					T = r.get_travel_time(soltype)  # calculate travel time
					if (R == None or T == None):
						continue
					raytracing[channel_id][iS]["travel time"] = T
					receive_vector = r.get_receive_vector(soltype)
					zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
					raytracing[channel_id][iS]["receive vector"] = receive_vector
					raytracing[channel_id][iS]["zenith"] = zenith
					raytracing[channel_id][iS]["azimuth"] = azimuth

					# we create a dummy efield to obtain the propagation effects from the ray tracer
					# this includes attenuation, focussing and reflection, depending on self._prop_config
					efield = ElectricField([channel_id])
					efield.set_frequency_spectrum(np.ones((3, len(self._ff)), dtype=complex), self._sampling_rate)
					efield = r.apply_propagation_effects(efield, iS)
					raytracing[channel_id][iS]["propagation_effects"] = efield.get_frequency_spectrum()
					raytracing[channel_id][iS]["raytype"] = r.get_solution_type(soltype)
					zenith_reflections = np.atleast_1d(r.get_reflection_angle(soltype))
					raytracing[channel_id][iS]["reflection angle"] = zenith_reflections
					viewing_angle = hp.get_angle(self._shower_axis,raytracing[channel_id][iS]["launch vector"])
					if channel_id == self._ch_Vpol:
						launch_vectors.append(self._launch_vector)
						viewing_angles.append(viewing_angle)

			self._raytracing = raytracing

		raytype = {}
		traces = {}
		timing = {}
		viewingangles = np.zeros((len(use_channels), 2))
		polarizations = []
		polarizations_antenna = []

		for ich, channel_id in enumerate(use_channels):
			raytype[channel_id] = {}
			traces[channel_id] = {}
			timing[channel_id] = {}

			for i_s, iS in enumerate(raytracing[channel_id]):

				raytype[channel_id][iS] = {}
				traces[channel_id][iS] = {}
				timing[channel_id][iS] = {}
				viewing_angle = hp.get_angle(self._shower_axis,raytracing[channel_id][iS]["launch vector"])
				if self._template:



					template_viewingangle = self._templates_viewingangles[np.abs(np.array(self._templates_viewingangles) - np.rad2deg(viewing_angle)).argmin()] ### viewing angle template which is closest to wanted viewing angle
					self._templates[template_viewingangle]
					template_energy = self._templates_energies[np.abs(np.array(self._templates_energies) - energy).argmin()]

					spectrum = self._templates[template_viewingangle][template_energy]
					spectrum = np.array(list(spectrum)[0])
					spectrum *= self._templates_R
					spectrum /= raytracing[channel_id][iS]["trajectory length"]

					spectrum *= energy#template_energy
					spectrum /= template_energy

					spectrum= fft.time2freq(spectrum, 1/self._dt)

				else:
					if self._shift_for_xmax:
						xmax = self._xmax(energy)
						theta_prime = self._theta_to_thetaprime (viewing_angle, xmax, raytracing[channel_id][iS]["trajectory length"])
					else:
						theta_prime = viewing_angle
					spectrum = signalgen.get_frequency_spectrum(
						energy , theta_prime, self._n_samples,
						self._dt, "HAD", n_index,
						raytracing[channel_id][iS]["trajectory length"],self._askaryan_model)

				viewingangles[ich,i_s] = viewing_angle

				polarization_direction_onsky = self._calculate_polarization_vector(channel_id, iS)
				cs_at_antenna = cstrans.cstrafo(*hp.cartesian_to_spherical(*raytracing[channel_id][iS]["receive vector"]))
				polarization_direction_at_antenna = cs_at_antenna.transform_from_onsky_to_ground(polarization_direction_onsky)
				#print("polarization direction at antenna", hp.cartesian_to_spherical(*polarization_direction_at_antenna))
				logger.debug('receive zenith {:.0f} azimuth {:.0f} polarization on sky {:.2f} {:.2f} {:.2f}, on ground @ antenna {:.2f} {:.2f} {:.2f}'.format(
					raytracing[channel_id][iS]["zenith"] / units.deg, raytracing[channel_id][iS]["azimuth"] / units.deg, polarization_direction_onsky[0],
					polarization_direction_onsky[1], polarization_direction_onsky[2],
					*polarization_direction_at_antenna))
				spectrum_3d = np.outer(polarization_direction_onsky, spectrum)

				if channel_id == self._ch_Vpol:
					polarizations.append( self._calculate_polarization_vector(self._ch_Vpol, iS))
					polarizations_antenna.append(polarization_direction_at_antenna)

				## apply ray tracing corrections:
				spectrum_3d *= raytracing[channel_id][iS]['propagation_effects']
				eR, eTheta, ePhi = spectrum_3d

                #### get antenna respons for direction
				zen = raytracing[channel_id][iS]["zenith"]
				az = raytracing[channel_id][iS]["azimuth"]
				efield_antenna_factor = trace_utilities.get_efield_antenna_factor(station, self._ff, [channel_id], det, zen,  az, self.antenna_provider)

                ### convolve efield with antenna reponse

				if isinstance(self._systematics, dict):
					C = self._systematics["antenna response"]["gain"][channel_id]
					new_efield = C*efield_antenna_factor[0]
					shift = self._systematics["antenna response"]["shift"][channel_id]
					samples = shift/((self._ff[1]-self._ff[0])*1000)
					new_efield = np.roll(C*efield_antenna_factor[0], int(samples))

					analytic_trace_fft = np.sum(new_efield * np.array([eTheta, ePhi]), axis = 0)
				elif starting_values:
					analytic_trace_fft = np.sum(efield_antenna_factor[0] * np.array([spectrum,np.zeros(len(spectrum))]), axis = 0)
				else:
					analytic_trace_fft = np.sum(efield_antenna_factor[0] * np.array([eTheta, ePhi]), axis = 0)

                ### apply bandpass filters
				analytic_trace_fft *= self._h[channel_id]

            	#### apply amplifier response
				analytic_trace_fft *= self._amp[channel_id]

				analytic_trace_fft[0] = 0

				### currently, we roll the trace back by 1/4 the trace length to approximately centre the pulse
				#TODO - come up with something more sensible
				#import matplotlib.pyplot as plt
				#fig = plt.figure()
				#ax = fig.add_subplot(111)
				#ax.plot(fft.freq2time(analytic_trace_fft, self._sampling_rate))
				#ax.plot(np.roll(fft.freq2time(analytic_trace_fft, self._sampling_rate), -int(np.argmax(abs(fft.freq2time(analytic_trace_fft, self._sampling_rate)))-0.5*len(fft.freq2time(analytic_trace_fft, self._sampling_rate)))))
				#fig.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/full_reco/Penalty/test_2.pdf")
                                ## shift such that maximum is in middle

				traces[channel_id][iS] = np.roll(fft.freq2time(analytic_trace_fft, self._sampling_rate), -int(self._n_samples /4))#-int(np.argmax(abs(fft.freq2time(analytic_trace_fft, self._sampling_rate)))-0.5*len(fft.freq2time(analytic_trace_fft, self._sampling_rate))))# np.roll(fft.freq2time(analytic_trace_fft, self._sampling_rate), int(self._n_samples / 4))

				timing[channel_id][iS] =raytracing[channel_id][iS]["travel time"]
				raytype[channel_id][iS] = raytracing[channel_id][iS]["raytype"]
		# logger.debug("Found solutions for channels {}".format(raytracing.keys()))

		if (self._first_iteration or first_iteration):
			for i, iS in enumerate(raytracing[self._ch_Vpol]):
				if iS == self._raytypesolution:
					self._launch_vector = launch_vectors[i]
					self._viewingangle = viewing_angles[i]
					self._pol = polarizations[i]
			self._first_iteration = False
		
		if self._pol is None:
			logger.warning((
				"No ray tracing solution exists for ch_Vpol with type {}."
				"Therefore no viewing angle or polarization could be returned."
			).format(self._raytypesolution))


		return traces, timing, self._launch_vector, self._viewingangle, raytype, self._pol











