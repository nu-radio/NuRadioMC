import NuRadioReco.utilities.units as units
import numpy as np

config = dict(
    sampling_rate = 5.0*units.GHz,
#    no_sim_station = True
)

vertex = dict(
    distance_step_3d=10 * units.m,
    z_step_3d=5 * units.m,
    widths_3d=np.arange(-60, 61, 5),
    distances_2d = np.arange(100, 4000, 200),#np.logspace(np.log10(100), np.log10(6000), 30),
    passband=[96*units.MHz, 300*units.MHz],
    azimuths_2d = np.linspace(0, 2*np.pi, 360, endpoint=False),
    use_maximum_filter = False,
    min_antenna_distance = 5 * units.m
)

direction = dict(
    passband=[50*units.MHz, 700*units.MHz],
    window_Vpol = [-10, 50],
    window_Hpol=[10, 40],
    icemodel='greenland_simple',
    att_model='GL1',
    grid_spacing = [.5*units.deg, 5*units.deg, .2],
    brute_force=False,
    use_fallback_timing=True,
    fit_shower_type=False,
)
config['vertex'] = vertex
config['direction'] = direction

run_params = dict(
    debug_vertex = True,
    debug_direction = True,
    noise_level_before_amp = 15.8 * units.microvolt,

    ilse_vertex_reco = False,
    sjoerd_vertex_reco = True,
    save_nur_output = False,
    restricted_input = False,

    add_noise = False,
    add_noise_only = False,
    no_sim_station = False,
    apply_bandpass_filters = True
)

config.update(run_params)
