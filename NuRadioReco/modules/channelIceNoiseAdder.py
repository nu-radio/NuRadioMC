import functools
import sys
sys.path.append('/home/masha/Desktop/MyWork/Functions')
import math
from signal_characteristics import*
import scipy
from radiotools import helper as hp

from NuRadioReco.utilities import units
from NuRadioMC.utilities import attenuation as attenuation_util
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import analyticraytracing as ray
import logging
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.detector.antennapattern
from scipy.interpolate import interp1d

logger = logging.getLogger('NuRadioReco.channelIceNoiseAdder')

class channelIceNoiseAdder:
    """
    Class that simulates the noise produced by ice radio emission.
    Assuming that ice in thermodynamic equilibrium radiates as black body.
    """

    def Rayleigh_Jeans_BB(self, freq,T):
        """
        Returns blackbody radiation according to the Rayleigh-Jeans law

        Parameters
        --------------
        freq:  float
            frequencies
        T: float
            Temperature of blackbody

        """
        T = T*units.kelvin
        k = (scipy.constants.Boltzmann * units.joule / units.kelvin)
        c = (scipy.constants.c * units.m / units.s)
        S = 2. * k * freq**2 / c**2 * T
        return S
    def get_temperature_ice(self,depth,model = "SP1"):
        """
        Returns the temperature in Kelvin as a function of depth for South Pole or Greenland

        Parameters
        ----------
        depth: float
            depth in default units
        model: "SP1","GL1", "GL2"
            "SP1" - South Pole
            "GL1", "GL2" -Greenland
        """
        if  model == "SP1":
            T = attenuation_util.get_temperature(depth)
        if (model == "GL1" or  model == "GL2"):
            folder = '/user/rcamphyn/envs/IceNoise/'
            df = pd.read_csv(folder+'temp_greenland.txt',header=0,sep = '\s+')
            f2 = interp1d(df['Depth(m)'],df['Temperature(C)'], kind='cubic',fill_value='extrapolate')
            T = f2(depth/units.m)
        T+=273.15
        return T


    def spectrum_piece_of_ice(self, depth,dr,freq, model = "SP1"):
        """
        Returns spectral power radiation of  piece of ice

        Parameters
        ----------
        depth: float
            depth of the piece of ice
        dr: float
            radial width of the piece of ice
        freq:  float
            frequencies
        model: "SP1","GL1", "GL2"
            "SP1" - South Pole
            "GL1", "GL2" -Greenland

        """
        T = self.get_temperature_ice(depth,model)
        k = ([1./attenuation_util.get_attenuation_length(-depth, frequency  = ff, model = model) for ff in freq])
        epsilon_nu = self.Rayleigh_Jeans_BB(freq,T)*k
        return epsilon_nu*dr

    def E_from_intensity(self, I):
        """
        Returns electromagnetic field strength

        Parameters
        ----------
        I: float
            Intensity of electromagnetic waves I= dP/dS
        """
        c = scipy.constants.c * units.m / units.s
        e0 = scipy.constants.epsilon_0 * (units.coulomb / units.V / units.m)
        E = np.sqrt(I/ (c * e0))
        return E

    def check_solution(self, x2,x1,ice,freq, fmax = 1*units.GHz):
        attn_isnan = False
        direct_refrac_sol = False
        type_sol = 0
        attn = np.zeros(len(freq))
        r = ray.ray_tracing(ice, use_cpp=True)
        r.set_start_and_end_point(x1, x2)
        r.find_solutions()
        iS_r = -1
        if(r.has_solution()):
            for iS in range(r.get_number_of_solutions()):
                if(r.get_solution_type(iS) >2):
                    continue
                direct_refrac_sol = True
                attn = r.get_attenuation(iS, freq , fmax)
                iS_r = iS
                if(math.isnan(attn[0])):
                    attn_isnan = True
                    attn = np.zeros(len(freq))
                break
        return direct_refrac_sol,attn_isnan,attn,r,iS_r

    def Add_attenuation_piece_of_ice(self, theta,d_theta ,radius,step,z_antenna,freq,ice,fmax):
        print("starting attenuation")
        z = z_antenna+np.cos(theta/units.rad)*radius
        x2 = np.array([np.sin(theta/units.rad)*radius, 0., z]) #efield
        x1 = np.array([0., 0., z_antenna])  # antenna
        direct_refrac_sol,attn_isnan,attn,r,iS_r = self.check_solution(x2,x1,ice,freq,fmax)
        radius_new = radius
        while(direct_refrac_sol and attn_isnan and (radius_new<radius+(step)/2)):
            radius_new+=(step)/20
            z_new = z_antenna+np.cos(theta/units.rad)*radius_new
            x2_new = np.array([np.sin(theta/units.rad)*radius_new, 0., z_new]) #efield
            direct_refrac_sol,attn_isnan,attn,r,iS_r = self.check_solution(x2_new,x1,ice,freq,fmax)
        radius_new = radius
        while(direct_refrac_sol and attn_isnan and (radius_new>radius-(step)/2)):
            radius_new-=(step)/20
            z_new = z_antenna+np.cos(theta/units.rad)*radius_new
            x2_new = np.array([np.sin(theta/units.rad)*radius_new, 0., z_new]) #efield
            direct_refrac_sol,attn_isnan,attn,r,iS_r = self.check_solution(x2_new,x1,ice,freq,fmax)
        theta_new = theta
        while(direct_refrac_sol==False and theta_new<(theta+d_theta/2)):
            theta_new+=d_theta/50
            theta_new = round(theta_new/units.degree,1)*units.degree
            z_new = z_antenna+np.cos(theta_new/units.rad)*radius
            x2_new = np.array([np.sin(theta_new/units.rad)*radius, 0., z_new]) #efield
            direct_refrac_sol,attn_isnan,attn,r,iS_r = self.check_solution(x2_new,x1,ice,freq,fmax)
        theta_new = theta
        while(direct_refrac_sol==False and theta_new>(theta-d_theta/2)):
            theta_new-=d_theta/50
            theta_new = round(theta_new/units.degree,1)*units.degree
            z_new = z_antenna+np.cos(theta_new/units.rad)*radius
            x2_new = np.array([np.sin(theta_new/units.rad)*radius, 0., z_new]) #efield
            direct_refrac_sol,attn_isnan,attn,r,iS_r = self.check_solution(x2_new,x1,ice,freq,fmax)
        zenith = -1
        sol_type = 0
        if(direct_refrac_sol):
            receive_vector = r.get_receive_vector(iS_r)
            zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
            sol_type = r.get_solution_type(iS_r)
        #check annt
        for i in range(len(attn)):
            if(math.isnan(attn[i])):
                attn[i] = 0
        print("finished one attenuation")
        return attn,zenith,sol_type


    def IceNoise_vs_Angle(self, theta, d_theta,z_antenna,freq,R = 2200*units.m,n_r = 100,r0 = 20*units.cm,ice = medium.southpole_2015(),model_ice = "SP1",fmax = 1*units.GHz):
        """
        Calculates the intensity of the ice radiation coming into the antenna at a certain angle

        Parameters
        ----------
        theta: float
            Arrival angle
        d_theta: float
            interval of signal arrival angles
        z_antenna: float
            depth of the antenna (<0)
        freq:  float
            frequencies
        R: float, default: 2200*units.m
            R - max distance from antenna.
            Only radiation from ice in sphere with radius R is calculating
        r0: float, default: 20*units.cm
            r0 - min distance from antenna.
            Radiation from ice in sphere with radius r0 is  not calculating
        n_r: int, default: 500
            The n_r parameter  - number of pieces of ice between distance r0 and R
        ice: medium, default: medium.southpole_2015()
        model_ice: "SP1","GL1", "GL2", default: "SP1"
            "SP1" - South Pole
            "GL1", "GL2" -Greenland
        fmax: float, default: 1*units.GHz
            maximum frequency
        """

        d_phi = d_theta
        radial_d = np.linspace(float(r0/units.m),float(R/units.m), n_r)*units.m
        step = np.abs(radial_d[1]-radial_d[2])
        radial_d +=step/2
        z_bottom = -2.7*units.km #!!!!!!!!!!
        d_f = freq[1]-freq[0]

        spectrum_att =np.zeros((3, freq.shape[0]), dtype=complex)
        flux_sum = np.zeros(len(freq))
        zeniths = np.array([])

        # parameters of solution
        N_sol_direct = 0
        N_sol_refrac = 0
        N_big_dteta = 0
        N_no_sol = 0
        N_sol_refl = 0
        N_air = 0
        N_bottom = 0
        N_good = 0

        for ir in range(0,len(radial_d )):
            radius = radial_d[ir]
            z = z_antenna+np.cos(theta/units.rad)*radius
            if(z>0):
                N_air += 1
                continue
            if (z<z_bottom):
                N_bottom +=1
                continue
            depth = -z

            # calculate spectral radiance of radio signal using rayleigh-jeans law
            S =  self.spectrum_piece_of_ice(depth,step,freq,model_ice)

            solid_angle = np.abs(np.sin(theta/units.rad)*np.sin(d_theta/2/units.rad)*2*d_phi)
            S*=solid_angle
            # calculate radiance per energy bin
            S_per_bin = S * d_f

            #add attanuation
            attn,zenith,sol_type  = self.Add_attenuation_piece_of_ice(theta,d_theta ,radius,step,z_antenna,freq,ice,fmax)
            S_per_bin*=attn

            zeniths = np.append(zeniths,zenith)
            #analize solution
            if (sol_type==0):
                N_no_sol += 1
                continue
            if (sol_type==1):N_sol_direct += 1
            if (sol_type==2):N_sol_refrac += 1
            # skip big theta difference
            if( np.abs(zenith/units.degree-theta/units.degree)>45):
                N_big_dteta +=1
                continue
            N_good += 1
            flux_sum+=S_per_bin

        sol_param = (N_sol_direct,N_sol_refrac,N_sol_refl,N_no_sol,N_big_dteta,N_air,N_bottom,N_good)
        return flux_sum, zeniths,sol_param


    def __init__(self):
        self.__debug = None
        self.__R = None
        self.__r0 = None
        self.__n_r = None
        self.__d_theta = None
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
        self.begin()

    def begin(
        self,
        debug=False,
        R = 2200*units.m,
        r0 = 20*units.cm,
        n_r = 500,
        d_theta = 5*units.degree

    ):
        """
        Set up important parameters for the module
        
        Parameters
        ---------------
        debug: bool, default: False
            It True, debug plots will be shown
        R: float, default: 2200*units.m
            R - max distance from antenna.
            Only radiation from ice in sphere with radius R is calculating
        r0: float, default: 20*units.cm
            r0 - min distance from antenna.
            Radiation from ice in sphere with radius r0 is  not calculating
        n_r: int, default: 500
            The n_r parameter  - number of pieces of ice between distance r0 and R
        d_theta: float, default: 5*units.degree
            d_theta - a step of zenith angle, using to calculate a solid angle
        """
        self.__debug = debug
        self.__R = R
        self.__r0 = r0
        self.__n_r = n_r
        self.__d_theta = d_theta


    @register_run()
    def run(
        self,
        event,
        station,
        detector,
        ice,
        model_ice,
        passband=None
    ):
        """
        Adds noise resulting from ice radio emission to the channel traces

        Parameters
        --------------
        event: Event object
            The event containing the station to whose channels noise shall be added
        station: Station object
            The station whose channels noise shall be added to
        detector: Detector object
            The detector description
        ice: medium, default: medium.southpole_2015()
        model_ice: "SP1","GL1", "GL2", default: "SP1"
            "SP1" - South Pole
            "GL1", "GL2" -Greenland
        passband: list of float
            Lower and upper bound of the frequency range in which noise shall be
            added
        """
        if passband is None:
            passband = [10 * units.MHz, 1200 * units.MHz]
        if ice is None:
            ice  = medium.southpole_2015()
            model_ice
        if model_ice is None:
            model_ice  = "SP1"

        thetas = np.arange(self.__d_theta/2,180*units.degree,self.__d_theta)
        phis = np.arange(0*units.degree,360*units.degree,self.__d_theta)
        station_id = station.get_id()

        if self.__debug:
            plt.close('all')
            fig = plt.figure(figsize=(12, 8))
            ax_1 = fig.add_subplot(211)
            ax_2 = fig.add_subplot(212)
            ax_1.grid()
            ax_2.grid()

            ax_1.set_xlabel("depth, m")
            ax_2.set_xlabel("depth, m")
            ax_1.set_ylabel("Ice-Temperature, K")
            ax_2.set_ylabel('Lattn [km]')
            depth0 = np.linspace(0,2.7*units.km)
            ax_1.plot(depth0 / units.m, self.get_temperature_ice(depth0,model_ice))
            ax_2.plot(depth0 / units.m, attenuation_util.get_attenuation_length(-depth0, frequency  = 200*units.MHz, model =model_ice)/units.km,label = '200 MHz')
            plt.legend()
            plt.show()


        for channel in station.iter_channels():
            test_i = 0
            print(test_i)
            test_i += 1
            antenna_pattern = self.__antenna_pattern_provider.load_antenna_pattern(detector.get_antenna_model(station.get_id(), channel.get_id()))
            self.freq = channel.get_frequencies()
            #passband_filter = (freqs > passband[0]) & (freqs < passband[1])
            #freq = freqs[passband_filter]
            cid = channel.get_id()
            z_antenna = detector.get_relative_position(station_id, cid)[2]
            sampling_rate = channel.get_sampling_rate()
            noise_spec_sum = np.zeros((3, self.freq.shape[0]), dtype=complex)
            channel_spectrum = np.zeros(len(self.freq), dtype=complex)
            antenna_orientation = detector.get_antenna_orientation(station_id, cid)
            fmax = sampling_rate/2
            # Add ice noise
            for theta in thetas:
                flux_sum, zeniths ,sol_param = self.IceNoise_vs_Angle(theta, self.__d_theta,z_antenna,self.freq,self.__R ,self.__n_r,self.__r0 ,ice,model_ice,fmax)

                # calculate electric field per energy bin from the radiance per bin
                E = self.E_from_intensity(flux_sum)#/ (d_f)
                for azimuth in phis:
                    # assign random phases and polarizations to electric field
                    noise_spectrum = np.zeros((3, self.freq.shape[0]), dtype=complex)
                    polarizations = np.random.uniform(0, 2. * np.pi, len(flux_sum))
                    phases = np.random.uniform(0, 2. * np.pi, len(flux_sum))
                    noise_spectrum[1] = np.exp(1j * phases) * E * np.cos(polarizations)
                    noise_spectrum[2] = np.exp(1j * phases) * E * np.sin(polarizations)
                    # fold electric field with antenna response
                    antenna_response = self.get_cached_antenna_response(antenna_pattern, theta, azimuth, *antenna_orientation)
                    channel_noise_spectrum = antenna_response['theta'] * noise_spectrum[1] + antenna_response['phi'] * noise_spectrum[2]
                    channel_spectrum += channel_noise_spectrum
                    noise_spec_sum += noise_spectrum


            E_trace = np.fft.irfft(channel_spectrum , axis=-1)*(len(channel_spectrum))
            new_trace = E_trace+channel.get_trace()
            channel.set_trace(new_trace, sampling_rate)

    @functools.lru_cache(maxsize=1024 * 32)
    def get_cached_antenna_response(self, antenna_pattern, zen, azi, *ant_orient):
        return antenna_pattern.get_antenna_response_vectorized(self.freq, zen, azi, *ant_orient)

