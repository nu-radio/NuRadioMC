import numpy as np
import matplotlib.pyplot as plt

from NuRadioReco.modules.channelSignalReconstructor import (
    channelSignalReconstructor,
)
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
import warnings, logging, scipy.ndimage
import pandas as pd
from NuRadioReco.utilities.logging import set_general_log_level
from reconstruction import _build_travel_time_map
from reconstruction import _run_deep_reco, get_SNRs
from travel_time_maps import load_map, load_map_non_interp, map_from_dict
set_general_log_level(logging.CRITICAL)
warnings.filterwarnings("ignore")

def plot_skymap(outpath, corrmap, zen, az):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize = (5, 4), layout = "constrained")
    gs = GridSpec(1, 1, figure = fig)
    ax = fig.add_subplot(gs[0], projection="polar")

    cmax = np.max(np.abs(corrmap))

    im = ax.pcolormesh(az, zen, corrmap,
                       cmap='bwr', rasterized=True, vmin=-cmax, vmax=cmax)
    
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad = 0.03)
        # Run peak finder

    max_zenith_index, max_azimuth_index = np.unravel_index(
        np.argmax(corrmap), corrmap.shape
    )

    ax.scatter(
        az[max_azimuth_index],
        zen[max_zenith_index],
        edgecolor="green",
        facecolor="none",
        s=100,
        label="Max Correlation",
        zorder=10,
    )
    ax.annotate(
        f"{corrmap[max_zenith_index, max_azimuth_index]:.2f}",
        (az[max_azimuth_index], zen[max_zenith_index]),
        textcoords="offset points",
        xytext=(0, 5),
        ha='center',
        fontsize=8,
        color="green",
        zorder=11,
    )
    
    data_max = scipy.ndimage.maximum_filter(corrmap, size=10)
    mask = (corrmap == data_max) & (corrmap > 0.9 * cmax)
    y, x = mask.nonzero()
    # remove maxima from secondary peaks
    max_mask = (y == max_zenith_index) & (x == max_azimuth_index)
    x = x[~max_mask]
    y = y[~max_mask]
    
    if len(x) != len(y):
        print(f"Warning: Found {len(x)} secondary peaks but {len(y)} zenith indices. This should not happen.")
        return 
    
    value = corrmap[y, x]
    
    
    print(f"Found {len(x)} peaks at 2-sigma")
    
    ax.scatter(az[x], zen[y], edgecolor="black", facecolor="none", s=50, label="Secondary Peaks $2\\sigma$", zorder=10)
    for i in range(len(x)):
        ax.annotate(f"{value[i]:.2f}", (az[x[i]], zen[y[i]]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color="black", zorder=11)
    
    fig.legend(loc="outside upper right", fontsize=10)
    

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(
        ["E (0°)", "NE (45°)", "N (90°)", "NW (135°)",
         "W (180°)", "SW (225°)", "S (270°)", "SE (315°)"],
        fontsize=10, 
    )
    ax.set_rlim(0, np.pi / 2)
    rticks = [np.deg2rad(r) for r in (0, 15.1, 30.1, 45.1, 60.1, 75.1, 90)]
    ax.set_rticks(rticks)
    
    ax.set_yticklabels([f"{int(np.degrees(r))}°" for r in rticks],
                       color="black", fontsize=10, zorder=10)

    cbar.set_label("Correlation")

    fig.savefig(outpath)
    
    plt.close()


if __name__ == "__main__":

    provider = dataProviderRNOG()
    signal_reconstructor = channelSignalReconstructor()

    RUN_NR = 4710
    EVENT_ID = 3475

    detector = Detector(select_stations=11)
    detector.update(pd.to_datetime("2024-01-02T00:00:00"))

    run_folder = f'/pnfs/ifh.de/acs/radio/diskonly/data/full/root/station11/run{RUN_NR}'
    provider.begin(files=run_folder, det=detector,
                reader_kwargs={
                    "mattak_kwargs": {"backend": "uproot"},
                    "apply_baseline_correction": None,
                })

    event = provider.reader.get_event(run_nr=RUN_NR, event_id=EVENT_ID)
    station = event.get_station()

    provider.channelBlockOffsetFitter.run(event, station, detector)
    provider.channelGlitchDetector.run(event, station, detector)
    provider.channelCableDelayAdder.run(event, station, detector, mode="subtract")
    signal_reconstructor.run(event, station, det=detector)

    deep_channels = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]

    ice_model = medium.greenland_3exp_layered()
            
    tt_maps = _build_travel_time_map(
        detector, ice_model, station.get_id(), deep_channels,
        zeniths = np.linspace(0.01, np.pi/2, 90),
        azimuths = np.linspace(0, 2*np.pi, 360),
    )

    tt_maps = map_from_dict(tt_maps)
    SNRs = get_SNRs(station, channels=deep_channels)

    config_dict = {
        "upsample_factor": 5,
        "post_process": "norm-square",
        "snr_min": 5.0,
        "imp_min": 0.1,
        "bandpass_lo_hi": [0.05, 0.5],
        "bandpass_order": 8,
    }

    station = event.get_station()
    trigger_type = station.get_first_trigger().get_name()
    station_time = station.get_station_time()

    (zenith, azimuth, correlation), corr_map = _run_deep_reco(
        deep_channels, SNRs, station, detector, tt_maps,
        RUN_NR, EVENT_ID, trigger_type, config_dict=config_dict,
    )
    skymap_path = "event_skymap.pdf"


    plot_skymap(skymap_path, corr_map, tt_maps["zeniths"], tt_maps["azimuths"])



