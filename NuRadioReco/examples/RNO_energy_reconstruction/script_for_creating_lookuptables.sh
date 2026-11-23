#!/bin/bash

# Lookup tables for the vertex channels of station 11, covering the full T00 volume.
# The tables assume the ice surface at z=0, so the antenna depths are the true depths below the surface:
# relative channel z + station z (-3.783 m). Use them with use_absolute_depth=True in T02.
# Each table takes a few hours; on the cluster, submit them as Condor jobs instead of running this locally.
output_path=${1:-./lookup_tables_absdepth}
mkdir -p "$output_path"

# Depths of channels 0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23
z_values=(99.998 98.957 97.966 96.938 82.904 62.914 43.140 86.733 85.723 99.443 100.453)

printf '%s\n' "${z_values[@]}" | xargs -P "${#z_values[@]}" -I {} \
    python create_lookup_table.py {} --r_min 10 --r_max 4000 --z_min 2800 --z_max 2 --d_r 2 --d_z 2 --output_path "$output_path"
