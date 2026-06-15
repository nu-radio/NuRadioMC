import argparse

parser = argparse.ArgumentParser(description='feature_extraction')
parser.add_argument('--input_sim', type=str, default=None)
parser.add_argument('--output_sim', type=str, default=None)
parser.add_argument('--input_burn', type=str, default=None)
parser.add_argument('--output_burn', type=str, default=None)
args = parser.parse_args()

import pandas as pd
import os
from NuRadioReco.modules.io import eventReader
from NuRadioReco.framework.parameters import eventParametersRNOG as ep
import csv

features = ["csw_snr_PA", "csw_snr_PS", "csw_snr_ALL", "csw_rpr_PA", "csw_rpr_PS", "csw_rpr_ALL", "csw_hilbert_snr_PA", "csw_hilbert_snr_PS", "csw_hilbert_snr_ALL", "csw_peak_PA", "csw_peak_PS", "csw_peak_ALL", "csw_power_PA", "csw_power_PS", "csw_power_ALL", "impulsivity_PA", "R2_PA", "slope_PA", "intercept_PA", "impulsivity_PS", "R2_PS", "slope_PS", "intercept_PS", "impulsivity_ALL", "R2_ALL", "slope_ALL", "intercept_ALL", "ks_PA", "ks_PS", "ks_ALL", "avg_snr", "avg_rpr", "theta", "phi", "r", "z", "max_corr", "surf_corr_ratio", "max_surf_corr", "surf_r", "surf_z", "readout_times", "trigger_times", "event_id", "has_glitch", "run_num", "energy"]

sim  = {f: [] for f in features}
data = {f: [] for f in features}

def extract_features(evt, store):

    try:

        csw_snr = evt.get_parameter(ep.csw_snr)
        csw_rpr = evt.get_parameter(ep.csw_rpr)
        csw_hilbert = evt.get_parameter(ep.csw_hilbert_snr)
        csw_peak = evt.get_parameter(ep.csw_peak)
        csw_power = evt.get_parameter(ep.csw_power)
        impulsivity = evt.get_parameter(ep.csw_impulsivity)
        energy = evt.get_parameter(ep.energy)

        avg_snr = evt.get_parameter(ep.avg_snr)
        avg_rpr = evt.get_parameter(ep.avg_rpr)

        coords = evt.get_parameter(ep.max_corr_coords)
        max_corr = evt.get_parameter(ep.max_corr)

        surf_corr_ratio = evt.get_parameter(ep.surf_corr_ratio)
        max_surf_corr = evt.get_parameter(ep.max_surf_corr)
        surf_pos = evt.get_parameter(ep.max_surf_corr_pos)

        theta = coords["elevation"]
        phi = coords["azimuth"]
        r = coords["r"]
        z = coords["z"]

        surf_r, surf_z = surf_pos[0]
        readout_times = evt.get_parameter(ep.readout_times)
        trigger_times = evt.get_parameter(ep.trigger_times)
        run_num = evt.get_parameter(ep.run_num)
        combos = ["PA","PS","ALL"]

        for c in combos:
            store[f"csw_snr_{c}"].append(csw_snr[c][0])
            store[f"csw_rpr_{c}"].append(csw_rpr[c])
            store[f"csw_hilbert_snr_{c}"].append(csw_hilbert[c])
            store[f"csw_peak_{c}"].append(csw_peak[c])
            store[f"csw_power_{c}"].append(csw_power[c])

            store[f"impulsivity_{c}"].append(impulsivity[c]["impulsivity"])
            store[f"R2_{c}"].append(impulsivity[c]["r_value"])
            store[f"slope_{c}"].append(impulsivity[c]["slope"])
            store[f"intercept_{c}"].append(impulsivity[c]["intercept"])
            store[f"ks_{c}"].append(impulsivity[c]["ks"])

        store["avg_snr"].append(avg_snr['peak_2_peak_amplitude'])
        store["avg_rpr"].append(avg_rpr)

        store["theta"].append(coords["elevation"])
        store["phi"].append(coords["azimuth"])
        store["r"].append(coords["r"])
        store["z"].append(coords["z"])

        store["max_corr"].append(max_corr)
        store["surf_corr_ratio"].append(surf_corr_ratio)
        store["max_surf_corr"].append(max_surf_corr)

        store["surf_r"].append(surf_pos[0][0])
        store["surf_z"].append(surf_pos[0][1])
        store["readout_times"].append(readout_times)
        store["trigger_times"].append(trigger_times)
        store["event_id"].append(evt.get_id())
        store["has_glitch"].append(evt.has_glitch())
        store["run_num"].append(run_num)
        store["energy"].append(energy)

    except Exception:
        return

def read_folder(indir, store):

    file_count = 0
    event_count = 0
    file_event_map = {}
    file_event_map[indir] = []
    if os.path.getsize(indir) == 0:
        print(f"{indir} is empty, skipping")
        return None
    reader = eventReader.eventReader()
    reader.begin(indir)
    for evt in reader.run():
        if (True == True):
            evt.add_parameter_type(ep)

            extract_features(evt, store)
            file_event_map[indir].append(evt.get_id())

    return file_event_map

sim_dir = args.input_sim
data_dir = args.input_burn
if (sim_dir is not None):
    file_event_map_sim = read_folder(sim_dir, sim)
    sim_df = pd.DataFrame(sim)
    sim_df.to_csv(args.output_sim, index=False)
if (data_dir is not None):
    file_event_map_data = read_folder(data_dir, data)
    data_df = pd.DataFrame(data)
    data_df.to_csv(args.output_burn, index=False)

