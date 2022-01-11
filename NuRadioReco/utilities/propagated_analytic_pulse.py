from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalGen import askaryan as signalgen
from NuRadioReco.detector import detector
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from radiotools import coordinatesystems as cstrans
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.detector import antennapattern
from scipy import signal
from NuRadioReco.utilities import trace_utilities
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator
import NuRadioReco.modules.io.eventReader
import datetime
from NuRadioReco.framework.parameters import stationParameters as stnp
from radiotools import helper as hp
import numpy as np
from pathlib import Path
import logging
import pickle
from NuRadioMC.SignalGen import ARZ
import NuRadioMC




receive_pickle, launch_pickle, solution_pickle, zenith_vertex_pickle = pickle.load(open('/lustre/fs22/group/radio/plaisier/software/simulations/planeWaveFit/receive_launch.pkl', 'rb')) ## i used this to play around with a four parameter fit (R, E, theta, phi)
#(librabry = '/lustre/fs22/group/radio/plaisier/software/NuRadioMC/NuRadioMC/SignalGen/ARZ/average.pkl') ## average ARZ model. In practice this is way too slow.

logger = logging.getLogger("sim")
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.parameters import showerParameters as shp
raytracing = {}
eventreader = NuRadioReco.modules.io.eventReader.eventReader()

ice = medium.get_ice_model('greenland_simple')
prop = propagation.get_propagation_module('analytic')
attenuate_ice = True


hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

class simulation():
	
	def __init__(self, template = False, vertex = [0,0,-1000], distances = [200, 300, 400, 500, 600, 700,800, 900, 1000, 1100,1143, 1200, 1300,1400,  1500, 1600, 1800, 2100, 2200, 2500, 3000, 300, 3500, 4000]):
		self._template = template
		self.antenna_provider = antennapattern.AntennaPatternProvider()
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
	
	def begin(self, det, station, use_channels, raytypesolution = False, ch_Vpol = None):
		""" initialize filter and amplifier """
		self._ch_Vpol = ch_Vpol
		sim_to_data = True
		self._raytypesolution= raytypesolution
		channl = station.get_channel(use_channels[0])
		self._n_samples = 800 ## templates are 800 samples long. The analytic models can be longer. 
		self._sampling_rate = channl.get_sampling_rate()
		self._dt = 1./self._sampling_rate		
	
		self._ff = np.fft.rfftfreq(self._n_samples, self._dt)
		tt = np.arange(0, self._n_samples * self._dt, self._dt)
        
		mask = self._ff > 0
		order = 8
		passband = [50* units.MHz, 1150 * units.MHz]
		b, a = signal.butter(order, passband, 'bandpass', analog=True)
		w, ha = signal.freqs(b, a, self._ff[mask])
		order = 10
		passband = [0* units.MHz, 700 * units.MHz]
		b, a = signal.butter(order, passband, 'bandpass', analog=True)
		w, hb = signal.freqs(b, a, self._ff[mask])
		fa = np.zeros_like(self._ff, dtype=np.complex)
		fa[mask] = ha
		fb = np.zeros_like(self._ff, dtype = np.complex)
		fb[mask] = hb
		self._h = fb*fa

		self._amp = {}
		for channel_id in use_channels:
			self._amp[channel_id] = {}
			
			self._amp[channel_id] = hardwareResponseIncorporator.get_filter(self._ff, station.get_id(), channel_id, det, sim_to_data = sim_to_data)
				
		order = 8
		passband = [20* units.MHz, 1150 * units.MHz]
		b, a = signal.butter(order, passband, 'bandpass', analog=True)
		w, hc = signal.freqs(b, a, self._ff[mask])
		fc = np.zeros_like(self._ff, dtype=np.complex)
		fc[mask] = hc
	#	self._f = fc

		pass


	def _calculate_polarization_vector(self, channel_id, iS):
		polarization_direction = np.cross(raytracing[channel_id][iS]["launch vector"], np.cross(self._shower_axis, raytracing[channel_id][iS]["launch vector"]))
		polarization_direction /= np.linalg.norm(polarization_direction)
		cs = cstrans.cstrafo(*hp.cartesian_to_spherical(*raytracing[channel_id][iS]["launch vector"]))
		return cs.transform_from_ground_to_onsky(polarization_direction)
	
	def simulation(self, det, station, vertex_x, vertex_y, vertex_z, nu_zenith, nu_azimuth, energy, use_channels, fit = 'seperate', first_iter = False, model = 'Alvarez2009', starting_values = False):
		ice = medium.get_ice_model('greenland_simple')
		prop = propagation.get_propagation_module('analytic')
		attenuate_ice = True
		polarization = True
		reflection = True

		vertex = [vertex_x, vertex_y, vertex_z]
		self._shower_axis = -1 * hp.spherical_to_cartesian(nu_zenith, nu_azimuth)
		n_index = ice.get_index_of_refraction(vertex)
		cherenkov_angle = np.arccos(1. / n_index)

		
		global raytracing ## define dictionary to store the ray tracing properties	
		global launch_vectors
		global launch_vector
		global viewingangle
		global pol
		if(first_iter):


			ice = medium.get_ice_model('greenland_simple')
			prop = propagation.get_propagation_module('analytic')
			attenuate_ice = True


			launch_vectors = []
			polarizations = []
			viewing_angles = []
			polarization_antenna = []
			chid = self._ch_Vpol
			x2 = det.get_relative_position(station.get_id(), chid) + det.get_absolute_position(station.get_id())
			r = prop( ice, 'GL1')
			r.set_start_and_end_point(vertex, x2)

			r.find_solutions()
			for iS in range(r.get_number_of_solutions()):
				if r.get_solution_type(iS) == self._raytypesolution:
					launch = r.get_launch_vector(iS)
     
					receive_zenith = hp.cartesian_to_spherical(*r.get_receive_vector(iS))[0]

			x1 = vertex

			for channel_id in use_channels:
                                
				raytracing[channel_id] = {}
				x2 = det.get_relative_position(station.get_id(), channel_id) + det.get_absolute_position(station.get_id())
				r = prop( ice, 'GL1')
				r.set_start_and_end_point(x1, x2)

				r.find_solutions()
				if(not r.has_solution()):
					print("warning: no solutions")
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
				
					attn = r.get_attenuation(soltype, self._ff)
				
					raytracing[channel_id][iS]["attenuation"] = attn
					raytracing[channel_id][iS]["raytype"] = r.get_solution_type(soltype)		
					zenith_reflections = np.atleast_1d(r.get_reflection_angle(soltype))
					raytracing[channel_id][iS]["reflection angle"] = zenith_reflections
					viewing_angle = hp.get_angle(self._shower_axis,raytracing[channel_id][iS]["launch vector"])
					if channel_id == self._ch_Vpol:
						launch_vectors.append( self._launch_vector)
						viewing_angles.append(viewing_angle)
					
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
                    
					spectrum *= template_energy ### this needs to be added otherwise energy is wrongly determined
					spectrum /= energy
					spectrum= fft.time2freq(spectrum, 1/self._dt)
				
				else:
		
					spectrum = signalgen.get_frequency_spectrum(energy , viewing_angle, self._n_samples, self._dt, "HAD", n_index, raytracing[channel_id][iS]["trajectory length"],model)

				# apply frequency dependent attenuation
				viewingangles[ich,i_s] = viewing_angle
				if attenuate_ice:
					spectrum *= raytracing[channel_id][iS]["attenuation"]
					
				if polarization:
	
					polarization_direction_onsky = self._calculate_polarization_vector(channel_id, iS)
				
					cs_at_antenna = cstrans.cstrafo(*hp.cartesian_to_spherical(*raytracing[channel_id][iS]["receive vector"]))
					polarization_direction_at_antenna = cs_at_antenna.transform_from_onsky_to_ground(polarization_direction_onsky)
					logger.debug('receive zenith {:.0f} azimuth {:.0f} polarization on sky {:.2f} {:.2f} {:.2f}, on ground @ antenna {:.2f} {:.2f} {:.2f}'.format(
						raytracing[channel_id][iS]["zenith"] / units.deg, raytracing[channel_id][iS]["azimuth"] / units.deg, polarization_direction_onsky[0],
						polarization_direction_onsky[1], polarization_direction_onsky[2],
						*polarization_direction_at_antenna))
				eR, eTheta, ePhi = np.outer(polarization_direction_onsky, spectrum)
			
		
				if channel_id == self._ch_Vpol:
					polarizations.append( self._calculate_polarization_vector(self._ch_Vpol, iS))
					polarizations_antenna.append(polarization_direction_at_antenna)
				## correct for reflection 
				r_theta = None
				r_phi = None
				
				n_surface_reflections = np.sum(raytracing[channel_id][iS]["reflection angle"] != None)
				if reflection:
					x2 = det.get_relative_position(station.get_id(), channel_id) + det.get_absolute_position(station.get_id())
					for zenith_reflection in raytracing[channel_id][iS]["reflection angle"]:  # loop through all possible reflections
							if(zenith_reflection is None):  # skip all ray segments where not reflection at surface happens
								continue
							r_theta = geo_utl.get_fresnel_r_p(
								zenith_reflection, n_2=1., n_1=ice.get_index_of_refraction([x2[0], x2[1], -1 * units.cm]))
							r_phi = geo_utl.get_fresnel_r_s(
								zenith_reflection, n_2=1., n_1=ice.get_index_of_refraction([x2[0], x2[1], -1 * units.cm]))

							eTheta *= r_theta
							ePhi *= r_phi
							logger.debug("ray hits the surface at an angle {:.2f}deg -> reflection coefficient is r_theta = {:.2f}, r_phi = {:.2f}".format(zenith_reflection / units.deg,
								r_theta, r_phi))

                ##### Get filter (this is the filter used for the trigger for RNO-G)
				
                #### get antenna respons for direction
				
				efield_antenna_factor = trace_utilities.get_efield_antenna_factor(station, self._ff, [channel_id], det, raytracing[channel_id][iS]["zenith"],  raytracing[channel_id][iS]["azimuth"], self.antenna_provider)
				
                ### convolve efield with antenna reponse
				if starting_values: analytic_trace_fft = np.sum(efield_antenna_factor[0] * np.array([spectrum,np.zeros(len(spectrum))]), axis = 0)
				if not starting_values: analytic_trace_fft = np.sum(efield_antenna_factor[0] * np.array([eTheta, ePhi]), axis = 0)
				
                ### filter the trace

				analytic_trace_fft *=self._h
				
		#### add amplifier

				analytic_trace_fft *= self._amp[channel_id]

				analytic_trace_fft[0] = 0
                         
		#### filter becuase of amplifier response
				#analytic_trace_fft *= self._f
#				analytic_trace_fft *= self._h
                ### store traces
				## rotate trace such that 
				traces[channel_id][iS] = np.roll(fft.freq2time(analytic_trace_fft, 1/self._dt), -500)
                ### store timing
				timing[channel_id][iS] =raytracing[channel_id][iS]["travel time"]
				raytype[channel_id][iS] = raytracing[channel_id][iS]["raytype"]
                     
			
		if(first_iter): ## seelct viewing angle due to channel with largest amplitude 
		     
			maximum_channel = 0
			
			for i, iS in enumerate(raytracing[self._ch_Vpol]):
				maximum_trace = max(abs(traces[self._ch_Vpol][iS])) ## maximum due to channel 6 (phased array)
			
				if raytype[self._ch_Vpol][iS] == self._raytypesolution:
					launch_vector = launch_vectors[i]
					viewingangle = viewing_angles[i]
					pol = polarizations[i]
                       
		return traces, timing, launch_vector, viewingangle,raytype, pol      	











