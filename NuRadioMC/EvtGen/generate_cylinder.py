from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
from NuRadioReco.utilities import units
import argparse
import logging
logger = logging.getLogger("EventGen")
logging.basicConfig()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate input data files using forced event generator for cylinder geometry')
    parser.add_argument('filename', type=str,
                        help='the output filename of the hdf5 file')
    parser.add_argument('n_events', type=int,
                        help='number of events to generate')
    parser.add_argument('Emin', type=float,
                        help='the minimum neutrino energy')
    parser.add_argument('Emax', type=float,
                        help='the maximum neutrino energy')
    parser.add_argument('fiducial_rmin', type=float,
                        help='lower r coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)')
    parser.add_argument('fiducial_rmax', type=float,
                        help='upper r coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)')
    parser.add_argument('fiducial_zmin', type=float,
                        help='lower z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)')
    parser.add_argument('fiducial_zmax', type=float,
                        help='upper z coordinate of fiducial volume (the fiducial volume needs to be chosen large enough such that no events outside of it will trigger)')
    parser.add_argument('--full_rmin', type=float, default=None,
                        help='lower r coordinate of simulated volume (if None it is set to 1/3 of the fiducial volume, if second vertices are not activated it is set to the fiducial volume)')
    parser.add_argument('--full_rmax', type=float, default=None,
                        help='upper r coordinate of simulated volume (if None it is set to 5x the fiducial volume, if second vertices are not activated it is set to the fiducial volume)')
    parser.add_argument('--full_zmin', type=float, default=None,
                        help='lower z coordinate of simulated volume (if None it is set to 1/3 of the fiducial volume, if second vertices are not activated it is set to the fiducial volume)')
    parser.add_argument('--full_zmax', type=float, default=None,
                        help='upper z coordinate of simulated volume (if None it is set to 5x the fiducial volume, if second vertices are not activated it is set to the fiducial volume)')
    parser.add_argument('--thetamin', type=float, default=0,
                        help='lower zenith angle for neutrino arrival direction (default 0deg)')
    parser.add_argument('--thetamax', type=float, default=180 * units.deg,
                        help='upper zenith angle for neutrino arrival direction (default 180deg)')
    parser.add_argument('--phimin', type=float, default=0 * units.deg,
                        help='lower azimuth angle for neutrino arrival direction')
    parser.add_argument('--phimax', type=float, default=360 * units.deg,
                        help='upper azimuth angle for neutrino arrival direction')
    parser.add_argument('--start_event_id', type=int, default=1,
                        help='event number of first event')
    parser.add_argument('--flavor', nargs='+', type=int, default=[12, -12, 14, -14, 16, -16],
                        help="""specify which neutrino flavors to generate. A uniform distribution of
                                all specified flavors is assumed.
                                The neutrino flavor (integer) encoded as using PDF numbering scheme,
                                particles have positive sign, anti-particles have negative sign,
                                relevant for us are:
                                * 12: electron neutrino
                                * 14: muon neutrino
                                * 16: tau neutrino""")
    parser.add_argument('--n_events_per_file', type=int, default=None,
                        help='the maximum number of events per output files. Default is None, which means that all events are saved in one file. If `n_events_per_file` is smaller than `n_events` the event list is split up into multiple files. This is useful to split up the computing on multiple cores.')
    parser.add_argument('--spectrum', type=str, default="log_uniform",
                        help="""defines the probability distribution for which the neutrino energies are generated
                            * 'log_uniform': uniformly distributed in the logarithm of energy
                            * 'E-?': E to the -? spectrum where ? can be any float
                            * 'IceCube-nu-2017': astrophysical neutrino flux measured with IceCube muon sample (https://doi.org/10.22323/1.301.1005)
                            * 'GZK-1': GZK neutrino flux model from van Vliet et al., 2019, https://arxiv.org/abs/1901.01899v1 for
                                       10 percent proton fraction (see get_GZK_1 function for details)
                            * 'GZK-1+IceCube-nu-2017': a combination of the cosmogenic (GZK-1) and astrophysical (IceCube nu 2017) flux""")
    parser.add_argument('--deposited', type=bool, default=False,
                        help='if True, generate deposited energies instead of primary neutrino energies')
    parser.add_argument('--proposal', default=False, action='store_true',
                        help='if integer, PROPOSAL generates a number of propagations equal to resample and then reuses them. Only to be used with a single kind of lepton (muon or tau)')
    parser.add_argument('--proposal_config', type=str, default="SouthPole",
                        help="""The user can specify the path to their own config file or choose among
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
        table paths and then copy this file to config_PROPOSAL_xxx.json.""")
    parser.add_argument('--start_file_id', type=int, default=0,
                        help="in case the data set is distributed over several files, this number specifies the id of the first file (useful if an existing data set is extended)")
    args = parser.parse_args()

    generate_eventlist_cylinder(args.filename, args.n_events, args.Emin, args.Emax,
                                args.fiducial_rmin, args.fiducial_rmax, args.fiducial_zmin, args.fiducial_zmax,
                                args.full_rmin, args.full_rmax, args.full_zmin, args.full_zmax,
                                args.thetamin, args.thetamax,
                                args.phimin, args.phimax,
                                args.start_event_id,
                                args.flavor,
                                args.n_events_per_file,
                                args.spectrum,
                                args.deposited,
                                args.proposal,
                                args.proposal_config,
                                args.start_file_id)
