"""Summarize per-stage timing from merged HDF5 reco results.

Reads timing keys written by interferometric_reco_3d_advanced.py and prints
a markdown-formatted table of statistics per reconstruction stage.
"""

import argparse
import h5py
import numpy as np


HW_TIMING_KEYS = [
    ('preproc_time', 'Preprocessing'),
    ('coarse_time', 'Coarse scan'),
    ('refine_time', 'Refine scan'),
    ('opt_time', 'Optimizer'),
    ('post_time', 'Post-processing'),
    ('raw_refine_time', 'Raw refine'),
]

RXTX_TIMING_KEYS = [
    ('p1_preproc_time', 'P1 preproc'),
    ('p1_coarse_time', 'P1 coarse'),
    ('p1_refine_time', 'P1 refine'),
    ('p1_opt_time', 'P1 optimizer'),
    ('p1_total_time', 'P1 total'),
    ('p2_dedisp_time', 'P2 dedispersion'),
    ('p2_coarse_time', 'P2 coarse'),
    ('p2_refine_time', 'P2 refine'),
    ('p2_opt_time', 'P2 optimizer'),
    ('p2_reco_time', 'P2 reco'),
]


def summarize(reco_file, mode):
    """Print per-stage timing statistics from a merged HDF5 results file.

    Args:
        reco_file: path to merged_reco_results.h5
        mode: "hw", "rx", or "rxtx"
    """
    with h5py.File(reco_file, 'r') as f:
        if mode is None:
            mode = f.attrs.get('mode', None)
            if mode is None:
                for k in f['results']:
                    if k.startswith('p1_'):
                        mode = 'rxtx'
                        break
                else:
                    mode = 'hw'
            if isinstance(mode, bytes):
                mode = mode.decode()

        results = f['results']
        n_events = results['rho'].shape[0]

        if mode in ('rx', 'rxtx'):
            key_list = RXTX_TIMING_KEYS
        else:
            key_list = HW_TIMING_KEYS

        available = []
        for hdf5_key, label in key_list:
            if hdf5_key in results:
                arr = results[hdf5_key][:]
                if np.any(arr > 0):
                    available.append((label, arr))

        if mode in ('rx', 'rxtx'):
            total_keys = ['p1_preproc_time', 'p1_total_time',
                          'p2_dedisp_time', 'p2_reco_time']
            total = np.zeros(n_events)
            for k in total_keys:
                if k in results:
                    total += results[k][:]
        else:
            total = np.zeros(n_events)
            for k in ['preproc_time', 'coarse_time', 'refine_time',
                       'opt_time', 'post_time', 'raw_refine_time']:
                if k in results:
                    total += results[k][:]

    print(f"File: {reco_file}")
    print(f"Mode: {mode}, Events: {n_events}")
    print()
    print("| Stage | Median (s) | Mean (s) | 25th (s) | 75th (s) | 90th (s) |")
    print("|-------|-----------|---------|---------|---------|---------|")

    for label, arr in available:
        print(f"| {label} | {np.median(arr):.2f} | {np.mean(arr):.2f} "
              f"| {np.percentile(arr, 25):.2f} | {np.percentile(arr, 75):.2f} "
              f"| {np.percentile(arr, 90):.2f} |")

    print(f"| **Total** | **{np.median(total):.2f}** | **{np.mean(total):.2f}** "
          f"| **{np.percentile(total, 25):.2f}** | **{np.percentile(total, 75):.2f}** "
          f"| **{np.percentile(total, 90):.2f}** |")


def main():
    """Parse arguments and run timing summary."""
    parser = argparse.ArgumentParser(
        description='Summarize per-stage timing from merged HDF5 reco results')
    parser.add_argument('--reco-file', required=True,
                        help='Path to merged_reco_results.h5')
    parser.add_argument('--mode', choices=['hw', 'rx', 'rxtx'], default=None,
                        help='Reco mode (auto-detected if not specified)')
    args = parser.parse_args()
    summarize(args.reco_file, args.mode)


if __name__ == '__main__':
    main()
