set -e
cd NuRadioMC/examples/RNO_G_trigger_simulation

# The seed is chosen such that (at least) 2 events trigger. One of them with a SNR ~ 8-9,
# i.e., it should trigger independent of the (unseeded) noise realization.
python3 simulate.py --station_id 11 -e 1e19 -n 100 --seed 1 --data_dir ci_test_data

python3 - <<'EOF'
import sys
import h5py

filename = "ci_test_data/station_11/nu_all_ccnc/all_ccnc_1e19.00eV_00000000.hdf5"
with h5py.File(filename, "r") as f:
    n_triggered = int(f["triggered"][:].sum()) if "triggered" in f else 0

print(f"Number of triggered events: {n_triggered}")
if n_triggered == 0:
    sys.exit("No events triggered, but at least one was expected for this fixed seed.")
EOF

rm -rf ci_test_data
