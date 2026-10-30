#!/bin/bash

# Lookup tables for the vertex channels of station 11, restricted to the range of the
# two triggered events in sim.nur (station-relative r = 915 m and 2101 m, z = -352 m and -1102 m).
output_path=${1:-./lookup_tables_two_events}
mkdir -p "$output_path"

# Depths of channels 0, 1, 2, 3, 5, 6, 7, 9, 10, 23, 22
z_values=(96.215 95.174 94.183 93.155 79.121 59.131 39.357 82.95 81.94 96.67 95.66)

printf '%s\n' "${z_values[@]}" | xargs -P "${#z_values[@]}" -I {} \
    python create_lookup_table.py {} --r_min 600 --r_max 2400 --z_min 1400 --z_max 2 --d_r 5 --d_z 5 --output_path "$output_path"
