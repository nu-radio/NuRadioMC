from functools import lru_cache
from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalGen import askaryan as signalgen
from NuRadioReco.detector import detector
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from radiotools import coordinatesystems as cstrans
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.electric_field import ElectricField
from scipy import signal
from NuRadioReco.utilities import trace_utilities, bandpass_filter
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator
import NuRadioReco.modules.io.eventReader
from NuRadioReco.framework.parameters import stationParameters as stnp
from radiotools import helper as hp
import numpy as np
from pathlib import Path
import logging
import pickle
from NuRadioMC.SignalGen import ARZ
import copy

logger = logging.getLogger("sim")
# logger.setLevel(logging.DEBUG)
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.parameters import showerParameters as shp
eventreader = NuRadioReco.modules.io.eventReader.eventReader()

hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

class simulation():

	def __init__(self, template = False, vertex = [0,0,-1000], distances = [200, 300, 400, 500, 600, 700,800, 900, 1000, 1100,1143, 1200, 1300,1400,  1500, 1600, 1800, 2100, 2200, 2500, 3000, 300, 3500, 4000]):
		self._template = template
		self.antenna_provider = antennapattern.AntennaPatternProvider()
		self._raytracing = dict()
		self._launch_vectors = None
		self._launch_vector = None
		self._viewingangle = None
		self._pol = None

		if self._template: ## I tried fitting with templates, but this is not better than fitting ARZ with Alvarez.
			self._templates_path = '/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/inIceMCCall/Uncertainties/templates'
			distance_event = np.sqrt(vertex[0]**2 + vertex[1]**2 + vertex[2]**2) ## assume station is at 000
			print("distance event", distance_event)

			R = distances[np.abs(np.array(distances) - distance_event).argmin()]
			print("selected distance", R)
			my_file = Path("/lustre/fs22/group/radio/plaisier/software/simulations/TotalFit/first_test/inIceMCCall/Uncertainties/templates/templates_{}.pkl".format(R, R))
			if my_file.is_file():
				f = NuRadioReco.utilities.io_utilities.read_pickle('{}'.format(my_file))
				self._templates = f
				self._templates_energies = f['header']['energies']
				self._templates_viewingangles = f['header']['viewing angles']
				self._templates_R = f['header']['R']
			else:
				## open look up tables
				viewing_angles = np.arange(40, 70, .2)
				self._header = {}
				self._templates = { 'header': {'energies': 0, 'viewing angles': 0, 'R': 0, 'n_indes': 0} }
				self._templates_viewingangles = []
				for viewing_angle in viewing_angles:
					if viewing_angle not in self._templates.keys():
						try:
							print("viewing angle", round(viewing_angle, 2))
							f = NuRadioReco.utilities.io_utilities.read_pickle('{}/templates_ARZ2020_{}_1200.pkl'.format(self._templates_path,int(viewing_angle*10))) #### in future 10 should be removed.
							if 1:#f['header']['R'] == 1500:
								self._templates[np.round(viewing_angle, 2)] = f
								self._templates_viewingangles.append(np.round(viewing_angle,2 ))
								self._templates_R = f['header']['R']
								print('done')
								self._templates['header']['R'] = self._templates_R
								self._templates['header']['energies'] = f['header']['energies']
								print("HEADER", self._templates['header']['R'])

						except:
							print("template for viewing angle {} does not exist".format(int(viewing_angle*10)))
				self._templates_energies = f['header']['energies']
				print("template energies", self._template_energies)
				self._templates['header']['viewing angles'] = self._templates_viewingangles
				with open('{}/templates_343.pkl'.format(self._templates_path), 'wb') as f: #### this should be args.viewingangle/10
					pickle.dump(self._templates, f)
		return

	def begin(
			self, det, station, use_channels, raytypesolution = False,
			ch_Vpol = None, Hpol_channels = None,
			Hpol_lower_band = 50, Hpol_upper_band = 700,
			passband = [96 * units.MHz, 1000 * units.MHz],
			ice_model="greenland_simple", att_model = 'GL1',
			propagation_module="analytic", propagation_config=None):
		""" initialize filter and amplifier """
		self._ch_Vpol = ch_Vpol
		sim_to_data = True
		self._raytypesolution= raytypesolution
		channl = station.get_channel(use_channels[0])
		self._sampling_rate = channl.get_sampling_rate()
		time_trace = 200 #ns
		self._dt = 1./self._sampling_rate
		self._n_samples = int(time_trace * self._sampling_rate) ## templates are 800 samples long. The analytic models can be longer.
		self._att_model = att_model
		self._ice_model = medium.get_ice_model(ice_model)
		self._prop = propagation.get_propagation_module(propagation_module)
		self._prop_config = propagation_config

        #### define bandpass filters and amplifier response
		self._ff = np.fft.rfftfreq(self._n_samples, self._dt)
		if not isinstance(passband, dict):
			passband = {channel_id:passband for channel_id in use_channels}

		self._h = dict()
		for channel_id in use_channels:
			passband_i = passband[channel_id]
			filter_response_1 = bandpass_filter.get_filter_response(self._ff, [passband_i[0], 1150*units.MHz], 'butter', 8)
			filter_response_2 = bandpass_filter.get_filter_response(self._ff, [0*units.MHz, passband_i[1]], 'butter', 10)
			self._h[channel_id] = filter_response_1 * filter_response_2 #fb * fa

		self._amp = {}
		for channel_id in use_channels:
			self._amp[channel_id] = {}
			self._amp[channel_id] = hardwareResponseIncorporator.get_filter(self._ff, station.get_id(), channel_id, det, sim_to_data = sim_to_data)

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
			nu_azimuth, energy, use_channels, fit = 'seperate',
			first_iter = False, model = 'Alvarez2009',
			starting_values = False, pol_angle = None):
		ice = self._ice_model

		vertex = np.array([vertex_x, vertex_y, vertex_z])
		self._shower_axis = -1 * hp.spherical_to_cartesian(nu_zenith, nu_azimuth)
		n_index = ice.get_index_of_refraction(vertex)

		raytracing = self._raytracing # dictionary to store ray tracing properties
		if(first_iter): # we run the ray tracer only on the first iteration

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
				if r.get_solution_type(iS) == self._raytypesolution:
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
					print("warning: no solutions", channel_id)
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
					
				
					xmax = self._xmax(energy)
					theta_prime = self._theta_to_thetaprime (viewing_angle, xmax, raytracing[channel_id][iS]["trajectory length"])
					spectrum = signalgen.get_frequency_spectrum(
						energy , theta_prime, self._n_samples,
						self._dt, "HAD", n_index,
						raytracing[channel_id][iS]["trajectory length"],model)

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
				if starting_values:
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
				
				traces[channel_id][iS] = np.roll(fft.freq2time(analytic_trace_fft, self._sampling_rate), int(self._n_samples /4))#-int(np.argmax(abs(fft.freq2time(analytic_trace_fft, self._sampling_rate)))-0.5*len(fft.freq2time(analytic_trace_fft, self._sampling_rate))))# np.roll(fft.freq2time(analytic_trace_fft, self._sampling_rate), int(self._n_samples / 4))

				timing[channel_id][iS] =raytracing[channel_id][iS]["travel time"]
				raytype[channel_id][iS] = raytracing[channel_id][iS]["raytype"]
		# logger.debug("Found solutions for channels {}".format(raytracing.keys()))

		if(first_iter):
			for i, iS in enumerate(raytracing[self._ch_Vpol]):
				if raytype[self._ch_Vpol][iS] == self._raytypesolution:
					self._launch_vector = launch_vectors[i]
					self._viewingangle = viewing_angles[i]
					self._pol = polarizations[i]

		if self._pol is None:
			if not first_iter:
				logger.warning(
					"Possibly not all relevant quantities were calculated - try running with first_iter=True")
			else:
				logger.warning((
					"No ray tracing solution exists for ch_Vpol with type {}."
					"Therefore no viewing angle or polarization could be returned."
				).format(self._raytypesolution))


		return traces, timing, self._launch_vector, self._viewingangle, raytype, self._pol











