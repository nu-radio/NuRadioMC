Event Generation
==========
    .. Important:: This document has been written for the master branch available in July 2020. It must be updated when the new internal looping is approved and merged.


The first thing to do when using NuRadioMC for simulating a realistic experiment is generating events. The module ``generator.py`` contained in the folder EvtGen allows the user to create input files for different detector geometries and physical assumptions, and constitutes the most important module for event generation.

Events in a cylindrical volume
----------

Generating input events for calculating effective volumes can be done just by importing the function ``generate_eventlist_cylinder`` and calling it with the minimum number of arguments requested:

    .. code-block:: Python

        from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
        from NuRadioReco.utilities import units
        filename = 'input_file.hdf5'
        n_events = int(1e5)
        Emin = 1e18 * units.eV
        Emax = 2e18 * units.eV
        fiducial_rmin = 0 * units.km
        fiducial_rmax = 4 * units.km
        fiducial_zmin = -3 * units.km
        fiducial_zmax = 0 * units.km
        generate_eventlist_cylinder(filename, n_events, Emin, Emax,
                                    fiducial_rmin, fiducial_rmax, 
                                    fiducial_zmin, fiducial_zmax)

These few lines generate a mixture of all-flavour events (1:1:1) in a fiducial cylinder. Each event constitutes a forced neutrino interaction in our fiducial volume. The event vertex distribution is homogeneous, and the flux in any region of the cylinder is isotropic, which is what would be expected in nature were the radius of the Earth small compared to the interaction length. NuRadioMC then uses weights to account for the probability that the neutrino reaches the vertex position.

Input parameters
__________
We explain here all the different input values for the ``generate_eventlist_cylinder`` function.

    .. code-block:: Python

        def generate_eventlist_cylinder(filename, n_events, Emin, Emax,
                                        fiducial_rmin, fiducial_rmax, 
                                        fiducial_zmin, fiducial_zmax,
                                        full_rmin=None, full_rmax=None, 
                                        full_zmin=None, full_zmax=None,
                                        thetamin=0.*units.rad, thetamax=np.pi * units.rad,
                                        phimin=0.*units.rad, phimax=2 * np.pi * units.rad,
                                        start_event_id=1,
                                        flavor=[12, -12, 14, -14, 16, -16],
                                        n_events_per_file=None,
                                        spectrum='log_uniform',
                                        add_tau_second_bang=False,
                                        tabulated_taus=True,
                                        deposited=False,
                                        proposal=False,
                                        proposal_config='SouthPole',
                                        start_file_id=0):
            """
            Event generator

            Generates neutrino interactions, i.e., vertex positions, neutrino directions,
            neutrino flavor, charged currend/neutral current and inelasticity distributions.
            All events are saved in an hdf5 file.

            Parameters
            ----------
            filename: string
                the output filename of the hdf5 file
            n_events: int
                number of events to generate
            Emin: float
                the minimum neutrino energy (energies are randomly chosen assuming a
                uniform distribution in the logarithm of the energy)
            Emax: float
                the maximum neutrino energy (energies are randomly chosen assuming a
                uniform distribution in the logarithm of the energy)

            fiducial_rmin: float
                lower r coordinate of fiducial volume (the fiducial volume needs to be 
                chosen large enough such that no events outside of it will trigger)
            fiducial_rmax: float
                upper r coordinate of fiducial volume (the fiducial volume needs to be 
                chosen large enough such that no events outside of it will trigger)
            fiducial_zmin: float
                lower z coordinate of fiducial volume (the fiducial volume needs to be 
                chosen large enough such that no events outside of it will trigger)
            fiducial_zmax: float
                upper z coordinate of fiducial volume (the fiducial volume needs to be 
                chosen large enough such that no events outside of it will trigger)
            full_rmin: float (default None)
                lower r coordinate of simulated volume (if None it is set to fiducial_rmin)
            full_rmax: float (default None)
                upper r coordinate of simulated volume (if None it is set to fiducial_rmax)
            full_zmin: float (default None)
                lower z coordinate of simulated volume (if None it is set to fiducial_zmin)
            full_zmax: float (default None)
                upper z coordinate of simulated volume (if None it is set to fiducial_zmax)
            """

These four parameters, ``full_rmin``, ``full_rmax``, ``full_zmax``, and ``full_zmin`` serve to increase the total interaction volume while saving and simulating only event vertices that lie within the cylinder defined by the parameters that start with ``fiducial``, so that the simulation time remains small. This is extremely useful for simulating multiple interactions from a single event. When a tau or muon neutrino interacts via charged current (CC), the resulting lepton can radiate and decay. In the case of a tau lepton, the range can reach 100 km around 100 EeV, which means that our interaction volume must be increased a lot so that we account correctly for these taus that can reach our fiducial volume after being created tens of kilometres away. The total number of events is also increased proportionally to the ratio of full volume vs fiducial volume.

    .. code-block:: Python

        """
        thetamin: float
            lower zenith angle for neutrino arrival direction
        thetamax: float
            upper zenith angle for neutrino arrival direction
        phimin: float
            lower azimuth angle for neutrino arrival direction
        phimax: float
             upper azimuth angle for neutrino arrival direction
        """

These parameters control the arrival directions of the incoming neutrinos. The azimuth distribution is uniform, while the zenith distribution is flat as a function of the cosine of the zenith, which implies an isotropic flux and a constant number of events per solid angle. Please keep in mind that, while the physical flux depends on the projected area of the cylinder, the probability of interaction (assuming a thin volume) depends on the chord length of the neutrino trajectory within the volume. For any direction, the product of the projected area times the average chord length is the total volume, so for any volume that is small with respect to the interaction length, the flux of interacting events must be isotropic (ignoring the probability of previous interaction). This is confirmed by the unforced event generator in ``generate_unforced.py``.

    .. code-block:: Python
        
        """
        start_event: int
            default: 1
            event number of first event
        flavor: array of ints
            default: [12, -12, 14, -14, 16, -16]
            specify which neutrino flavors to generate. A uniform distribution of
            all specified flavors is assumed.
            The neutrino flavor (integer) encoded as using PDF numbering scheme,
            particles have positive sign, anti-particles have negative sign,
            relevant for us are:
            * 12: electron neutrino
            * 14: muon neutrino
            * 16: tau neutrino
        """

The flavour codes follow the PDG conventions. Positive integers indicate neutrinos and negative numbers indicate antineutrinos. Neutrinos and antineutrinos possess a slightly different cross section, and also the lepton created via CC interaction changes, which in turn changes the stochastic losses of the lepton. However, at high energies and for the radio technique, the difference between neutrinos and antineutrinos is negligible for practical purposes. The input ``flavor`` must be a list, from which the neutrino flavour will be randomly drawn. So, the default list ``[12,-12,14,-14,16,-16]`` creates a 1:1:1 flavour ratio. If we want to use a single flavour, e.g. electron neutrino, we can use [12,-12]. Integer ratios are easily created. If we want a 1:2:0 ratio, we can pass [12,-12,14,14,-14,-14] as ``flavor``.

    .. code-block:: Python
        
        """
        n_events_per_file: int or None
            the maximum number of events per output files. Default is None, which
            means that all events are saved in one file. If 'n_events_per_file' is
            smaller than 'n_events' the event list is split up into multiple files.
            This is useful to split up the computing on multiple cores.
        """

When producing effective volumes for a large number of events, it is advisable to split the files in smaller input files so that each
file can be simulated in parallel using a cluster. Due to the Poissonian distribution of the number of triggered events, the relative uncertainty for, let's say, an effective volume goes with :math:`1/N_{triggered}^{0.5}`. However, the number of triggers is unknown a priori because that's precisely one of the things NuRadioMC calculates. A good rule of thumb for most detector configurations is to take around 10\ :sup:`5` ~ 10\ :sup:`6` events per energy bin for low energies (``n_events``), where radio efficiency is low, and for high energies the number of simulated neutrinos can be reduced. The number of events per file will depend on the number of jobs desired.

    .. code-block::
        
        """
        spectrum: string
            defines the probability distribution for which the neutrino energies are generated
            * 'log_uniform': uniformly distributed in the logarithm of energy
            * 'E-?': E to the -? spectrum where ? can be any float
            * 'IceCube-nu-2017': astrophysical neutrino flux measured with IceCube muon sample 
                                 (https://doi.org/10.22323/1.301.1005)
            * 'GZK-1': GZK neutrino flux model from van Vliet et al., 2019, 
                       https://arxiv.org/abs/1901.01899v1 for 10% proton fraction 
                       (see get_GZK_1 function for details)
            * 'GZK-1+IceCube-nu-2017': a combination of the cosmogenic (GZK-1) and 
                                       astrophysical (IceCube nu 2017) flux
        """

If instead of simulating a small energy bin one wishes to simulate a larger part of the neutrino spectrum, one can make use of this feature. By default, the input neutrino energies are drawn from a log-uniform distribution, which is appropriate for bins. An arbitrary power law, the measured IceCube flux, a GZK neutrino model, and a combination of the two previous can be chosen. If a really large ``n_events`` is used and then the file is split into reasonable cluster jobs using an appropriate ``n_events_per_file``, the NuRadioMC output files can be merged after simulation and all the relevant information on the measured events can be obtained. Keep in mind, however, that it is usually more flexible to simulate single bins with a log-uniform distribution and then convolve the results with whatever neutrino flux model we fancy.

    .. code-block:: Python

        """
        add_tau_second_bang: bool
            if True simulate second vertices from tau decays
        """

``add_tau_second_bang`` should be set to True if one wants to simulate tau double bangs with the simple model native to NuRadioMC. It also controls the full radius of the cylinder if ``full_rmax`` is ``None``, and it sets it to the 95% percentile of the tau range for a given energy, where the value is taken from a fit to the range.

If ``add_tau_second_bang`` is `True` and ``proposal`` is ``False``, tau decays will be simulated using a continuous slowing down approximation (CSDA) where the only relevant interaction is the photonuclear interaction. Stochastic losses are ignored. See the `NuRadioMC paper <https://dx.doi.org/10.1140/epjc/s10052-020-7612-8>`__ for more details.

    .. code-block:: Python

        """
        tabulated_taus: bool
            if True the tau decay properties are taken from a table
        """

``tabulated_taus`` controls if the code uses the tabulated results for the photonuclear CSDA. It is recommended to be used if the CSDA is wanted, although using PROPOSAL is the best option, if it's available.

    .. code-block:: Python

        """
        deposited: bool
            if True, generate deposited energies instead of primary neutrino energies
        """

``deposited`` allows us to choose to work with shower energy instead of neutrino energy. This variable should be set to ``True`` only for first neutrino interactions, without lepton propagation. It is useful when the response of the detector as a function of shower energy, which is what is actually measurable, is needed. It has been used for generating the radio effective volumes that are used in Jakob van Santen's `Gen2 framework <https://github.com/IceCubeOpenSource/gen2-analysis.git>`__. Using this feature, the shower energy is randomly drawn using the input ``Emin`` and ``Emax``, as well as the inelasticity, both of which can be used to find out the initial neutrino energy. Then, the file is saved and it can be normally executed using NuRadioMC.

    .. code-block:: Python

        """
        proposal: bool
            if True, the tau and muon secondaries are calculated using PROPOSAL
        """

If ``proposal`` is ``True``, an accurate propagation of muons (if a muon neutrino undergoes a CC interaction) and taus (the same, for a tau neutrino). This propagation is performed using the PROPOSAL code, which must be previously installed on the system. See how to at:

    https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md

NuRadioMC feeds PROPOSAL the lepton properties and PROPOSAL propagates them, storing every interaction along the way as well as the final decay. NuRadioMC then saves only the interactions that produce a shower above some energy threshold and transform them into events. See the NuRadioProposal.py module for more information.

    .. code-block:: Python

        """
        proposal_config: string or path
            The user can specify the path to their own config file or choose among
            the three available options:
            -'SouthPole', a config file for the South Pole (spherical Earth). It
            consists of a 2.7 km deep layer of ice, bedrock below and air above.
            -'MooresBay', a config file for Moore's Bay (spherical Earth). It
            consists of a 576 m deep ice layer with a 2234 m deep water layer below,
            and bedrock below that.
            -'InfIce', a config file with a medium of infinite ice
            -'Greenland', a config file for Summit Station, Greenland (spherical Earth),
            same as SouthPole but with a 3 km deep ice layer.
            IMPORTANT: If these options are used, the code is more efficient if the
            user requests their own "path_to_tables" and "path_to_tables_readonly",
            pointing them to a writable directory
            If one of these three options is chosen, the user is supposed to edit
            the corresponding config_PROPOSAL_xxx.json.sample file to include valid
            table paths and then copy this file to config_PROPOSAL_xxx.json.
        """

PROPOSAL needs a configuration file specifying the geometry to be run. The user can choose among the media listed above or they can specify the path to an own file. Important: the listed media need some input from the user. The files that end with ``.sample`` need to have two writable directories on the user's system to save some tables (which makes PROPOSAL faster), and they need to be renamed by removing the suffix ``.sample``.

    .. code-block:: Python

        """
        start_file_id: int (default 0)
            in case the data set is distributed over several files, 
            this number specifies the id of the first file
            (useful if an existing data set is extended)
            if True, generate deposited energies instead of primary neutrino energies
        """

Data sets and attributes
__________
The function ``generate_eventlist_cylinder`` creates events according to the input parameters and saves all the relevant parameters to a set of HDF5 files. These files created by the event generator consist of a collection of arrays containing the properties of the neutrinos and other secondary particles. The array keys and contents are the following:

    * ``azimuths``, the arrival azimuth angles in radians.
    * ``zeniths``, the arrival zenith angles in radians.
    * ``xx``, ``yy``, and ``zz``, the x, y and z coordinates in metres for the point where the particles interact or decay.
    * ``event_ids``, the event identification numbers
    * ``n_interaction``, the interaction number. 1 indicates a neutrino interaction, 2 and greater indicates decay or interaction of a lepton created after the neutrino interaction.
    
    .. Important:: As of July 2020, the event ids are not unique. This is because we consider as a single event all the shower-inducing interaction created by a neutrino and its secondary particles. One of our open issues is to change this numbering scheme in order to facilitate the splitting of events in different sub-events with a fixed-length trace. See `this pull request <https://github.com/nu-radio/NuRadioMC/pull/208>`__.

    * ``vertex_times``, the time at which the interaction happens. The first neutrino interaction is taken to be equal to zero, and the rest of the interactions are referred to this first interaction and calculated using the time of flight.
    * ``flavors``, neutrino flavours. 12 for electron neutrino, 14 for muon neutrino, and 16 for tau neutrino. Antineutrinos are represented by -12, -14, and -16. A value of 15 indicates a tau lepton. The numbers are following the `PDG standard <http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf>`__, and we have added some more between 80 and 90 to denote the particles and groups of particles created by PROPOSAL. The codes can be consulted in the table below.
    * ``energies``, the particle energies in electronvolts
    * ``interaction_type``, the interaction type. `'cc'` for charged current, and `'nc'` for neutral current. `'tau_had', 'tau_em', 'tau_mu'` indicate the tau decays into the hadronic, electromagnetic and muonic channels respectively, calculated using the CSDA for taus. If the vertex has been created by a secondary particle propagated by PROPOSAL, `had` indicates a hadronic shower, and `em` an electromagnetic shower.
    * ``inelasticities``, the inelasticities for the neutrino interactions and the tau decays. The inelasticity value represents the fraction of the initial energy taken by the product hadronic cascade. This is only relevant for first interactions. For secondary interactions, this field is set to 1.

In these HDF5 files we also save as HDF5 attributes:

    * ``n_events``, the number of events in the present file. If the file is split into smaller files, the number of events is recalculated.
    * ``NuRadioMC_EvtGen_version``
    * ``NuRadioMC_EvtGen_version_hash``
    
The following items have the same meaning as the parameters that are passed to the generator function.

    * ``start_event_id``
    * ``fiducial_rmin``
    * ``fiducial_rmax``
    * ``fiducial_zmin``
    * ``fiducial_zmax``
    * ``rmin``, equivalent to ``full_rmin``
    * ``rmax``, equivalent to ``full_rmax``
    * ``zmin``, equivalent to ``full_zmin``
    * ``zmax``, equivalent to ``full_zmax``
    * ``flavors``, the flavour list
    * ``Emin``
    * ``Emax``
    * ``thetamin``
    * ``thetamax``
    * ``phimin``
    * ``phimax``
    * ``deposited``

The HDF5 data sets outlined here are what constitute a NuRadioMC input file produced its generator module. However, any HDF5 file containing equally named data sets (and, depending on the purpose, also the attributes) can be processed by NuRadioMC to simulate a detector. This is useful in case one wants to compare the effect of a different input generation on the simulation output. However, although these comparisons are necessary for cross-checking, we encourage our NuRadioMC users to contribute to our projects and expand on our event generators if they consider that a different or complementary way of drawing input events is advisable.

    .. csv-table:: Particle codes used in NuRadioMC
            :header: "Name", "Symbol", "Code"
            
            Gamma (photon),:math:`\gamma`, 0
            Electron, :math:`e^-`, 11
            Positron, :math:`e^+`, -11
            Electron neutrino, :math:`\nu_e`, 12
            Electron antineutrino, :math:`\bar{\nu}_e`, -12
            Muon (negative), :math:`\mu^-`, 13
            Antimuon (positive muon), :math:`\mu^+`, -13
            Muon neutrino, :math:`\nu_{\mu}`, 14
            Muon antineutrino, :math:`\bar{\nu}_{\mu}`, -14
            Tau (negative), :math:`\tau^-`, 15
            Antitau (or positive tau), :math:`\tau^+`, -15
            Tau neutrino, :math:`\nu_{\tau}`, 16
            Tau antineutrino, :math:`\bar{\nu}_{\tau}`, -16
            Bremsstrahlung photon, :math:`\gamma_{brems}`, 81
            Ionised electron, :math:`\delta`, 82 
            Electron-positron pair, :math:`e^+e^-`, 83
            Hadron blundle, , 84
            Nuclear interaction products, , 85
            Hadronic Decay bundle, , 86
            Muon pair, :math:`\mu^+\mu^-`, 87
            Continuous loss, , 88
            Weak interaction, , 89
            Compton, , 90
            Pion (neutral), :math:`\pi^0`, 111
            Pion (positive), :math:`\pi^+`, 211
            Pion (negative), :math:`\pi^-`, -211
            Kaon (neutral), :math:`K^0`, 311
            Kaon (positive), :math:`K^+`, 321
            Kaon (negative), :math:`K^-`, -321
            Proton, :math:`p^+`, 2212
            Antiproton, :math:`p^-`, -2212

Atmospheric muons generated on a flat surface
----------
The function ``generate_surface_muons`` generates muons (leptons, NOT muon neutrinos) at ``z=0`` and propagates them using PROPOSAL. Please be aware that a functioning installation of PROPOSAL is needed for using this function. 

    .. code-block: Python
        def generate_surface_muons(filename, n_events, Emin, Emax,
                                   fiducial_rmin, fiducial_rmax, 
                                   fiducial_zmin, fiducial_zmax,
                                   full_rmin=None, full_rmax=None, 
                                   full_zmin=None, full_zmax=None,
                                   thetamin=0.*units.rad, thetamax=np.pi * units.rad,
                                   phimin=0.*units.rad, phimax=2 * np.pi * units.rad,
                                   start_event_id=1,
                                   plus_minus='mix',
                                   n_events_per_file=None,
                                   spectrum='log_uniform',
                                   start_file_id=0,
                                   config_file='SouthPole'):
            """
            Event generator for surface muons


            Generates muons at the surface for the atmospheric muon acceptance studies.
            All events are saved in an hdf5 file.
            """

Most of the arguments are the same as the ones used for ``generate_eventlist_cylinder``, although there are fewer and ``spectrum`` can only be ``"log-uniform"`` and ``"E-X"``. The only additional option for ``generate_surface_muons`` is ``plus_minus``.

    .. code-block:: Python

        """
        plus_minus: string
            if 'plus': generates only positive muons
            if 'minus': generates only negative muons
            else generates positive and negative muons randomly
        """

This parameter controls if the generated muons are negative, positive, or an equal mixture.

This function takes a thin disk of 10 cm of height near z=0 and randomly generates muons using an isotropic flux. The radius of this cylinder is equal to ``full_rmax`` and given by the user. Then, NuRadioMC calls PROPOSAL and saves all the shower-inducing interactions muons have undergone in a cylindrical region defined by ``rmin``, ``rmax``, ``zmin```, and ``zmax``. Although the generation of events is done in a flat surface near z=0, the curvature of the Earth is correctly considered by PROPOSAL provided the configuration file defines a spherical medium.

The main purpose of the ``generate_surface_muons`` is to generate muons coming from air showers to know how many of
these muons are detected by an in-ice array and therefore constitute a physical background for neutrino detection. This has been
explored in one of our papers (`lepton paper <https://arxiv.org/abs/2003.13442>`__). The user can choose the incoming zenith angle of these atmospheric muons as well, and they should be careful to use only the upper half of the sky (from 0 to 90 degrees in a standard simulation) to simulate muons coming from above.

For a realistic background simulation for which high statistics are needed, we recommend the use of a cluster to execute this function in parallel to create one or more files for each energy bin. The user should be aware of the following pitfall: if two input files with the same input parameters are going to be used for the same analysis, these two files should not have overlapping event IDs. The number of events for each file is an input parameter, so one can know a priori how many event IDs are going to be in a file. Unlike when using the ``generate_eventlist_cylinder``, increasing the full volume does not change the total number of events. The user can use the parameter ``n_events_per_file`` to split the file, but sometimes it is more convenient and faster to execute the function in parallel than calling it once and splitting the file.

NuRadioProposal as a standalone module
----------
The module ``NuRadioProposal.py`` in EvtGen can also be used standalone to study the propagation of leptons created by neutrino interactions and the radio-detectable showers they produce. For instance, for propagating 1000 taus having 1 EeV of energy in infinite ice, we can use the following code.

    .. code-block:: Python

        from NuRadioMC.EvtGen.NuRadioProposal import ProposalFunctions
        from NuRadioReco.utilities import units
        proposal_functions = ProposalFunctions(config_file='InfIce')

        N_taus = 1000
        energy_leptons = [units.EeV] * N_taus
        tau_codes = [15] * N_taus

        secondaries_array = proposal_functions.get_secondaries_array(energy_leptons,
                                                 lepton_codes,
                                                 config_file='InfIce',
                                                 low_nu=0.1*units.PeV,
                                                 propagate_decay_muons=True)

The minimum energy when the lepton propagation stops can be controlled with ``low_nu``, and the minimal energy of the shower-inducing secondaries can be controlled with ``min_energy_loss_nu``. ``get_secondaries_array`` returns a 2D list containing all the secondary particles that induce a shower. The first dimension indicates the primary lepton and the second dimension navigates through the secondaries produced by that primary. Each one of the elements is a member of the SecondaryProperties class, which has the distance from the lepton creation, its energy, the shower type, the PDG code and the name of the secondary particle as class properties.

    .. code-block:: Python

        class SecondaryProperties:
            """
            This class stores the properties from secondary particles that are
            relevant for NuRadioMC, namely:
            - distance, the distance to the first interaction vertex
            - energy, the particle energy
            - shower_type, whether the shower they induce is hadronic or electromagnetic
            - name, its name according to the particle_name dictionary on this module

            Distance and energy are expected to be in NuRadioMC units
            """
            def __init__(self,
                         distance,
                         energy,
                         shower_type,
                         code,
                         name):
                self.distance = distance
                self.energy = energy
                self.shower_type = shower_type
                self.code = code
                self.name = name

So, each property can be retrieved just by taking a SecondaryProperties object and accessing its properties. If the particle code lies between 80 and 90 (except for 86, hadronic decay bundle), the particle has been created upon an interaction before decaying. Any other particle is a product of decay. If there is more than one hadron created during decay, the NuRadioProposal module groups them into a hadronic decay bundle and adds their energies, so that the final shower is hadronic and with the sum of the energies.

If the user is only interested in decay energy and distance, 
the function ``get_decays`` can be used in a similar way, and it returns a list of (distance, energy) tuples.