#!/bin/bash
# Example SLURM submission for 3D interferometric reconstruction.
#
# Splits input NUR files across N_CHUNKS parallel jobs. Each chunk processes
# a subset of files and writes results to a separate HDF5 file. After all
# chunks complete, a merge job concatenates them into merged_reco_results.h5.
#
# Usage:
#   bash submit_reco3d_example.sh <config.yaml> <data_dir> <output_dir> [mode]
#
# Arguments:
#   config.yaml  - Reco config (e.g., configs/reco3d_neutrino_gzk.yaml)
#   data_dir     - Directory containing input .nur files
#   output_dir   - Where to write per-chunk HDF5 results
#   mode         - hw, rx, or rxtx (default: hw)
#
# Adjust ACCOUNT, PARTITION, N_CHUNKS, WALLTIME, and MEM for your cluster.

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: submit_reco3d_example.sh <config.yaml> <data_dir> <output_dir> [mode]"
    exit 1
fi

CONFIG=$1
DATA_DIR=$2
OUT_DIR=$3
MODE=${4:-hw}

# Cluster settings (adjust for your environment)
ACCOUNT="your_account"
PARTITION="your_partition"
N_CHUNKS=100
MEM="4GB"

if [ "$MODE" = "hw" ]; then
    WALLTIME="00:30:00"
elif [ "$MODE" = "rx" ]; then
    WALLTIME="03:00:00"
elif [ "$MODE" = "rxtx" ]; then
    WALLTIME="03:00:00"
else
    echo "Unknown mode: $MODE (use hw, rx, or rxtx)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRIVER="${SCRIPT_DIR}/interferometric_reco_3d_example.py"

mkdir -p "${OUT_DIR}/slurm_outputs"

# Collect all NUR files and split into chunks
mapfile -t ALL_FILES < <(find "$DATA_DIR" -name "*.nur" -type f | sort)
N_FILES=${#ALL_FILES[@]}

if [ "$N_FILES" -eq 0 ]; then
    echo "No .nur files found in $DATA_DIR"
    exit 1
fi

if [ "$N_CHUNKS" -gt "$N_FILES" ]; then
    N_CHUNKS=$N_FILES
fi

echo "Found $N_FILES NUR files, splitting across $N_CHUNKS chunks"
echo "Mode: $MODE, Walltime: $WALLTIME, Memory: $MEM"

CHUNK_DIR="${OUT_DIR}/chunk_filelists"
mkdir -p "$CHUNK_DIR"

# Split files round-robin into chunk lists
for ((i=0; i<N_CHUNKS; i++)); do
    : > "${CHUNK_DIR}/chunk_${i}.txt"
done
for ((i=0; i<N_FILES; i++)); do
    chunk_idx=$((i % N_CHUNKS))
    echo "${ALL_FILES[$i]}" >> "${CHUNK_DIR}/chunk_${chunk_idx}.txt"
done

# Submit array of jobs
JOB_IDS=""
for ((i=0; i<N_CHUNKS; i++)); do
    FILES_ARG=$(tr '\n' ' ' < "${CHUNK_DIR}/chunk_${i}.txt")
    if [ -z "$FILES_ARG" ]; then
        continue
    fi

    JID=$(sbatch --parsable \
        --job-name="reco3d_${i}" \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 \
        --mem="${MEM}" --time="${WALLTIME}" \
        --output="${OUT_DIR}/slurm_outputs/slurm_%j_chunk${i}.out" \
        --wrap="python ${DRIVER} --config ${CONFIG} --mode ${MODE} -i ${FILES_ARG} -o ${OUT_DIR}/chunk_${i}.h5")

    JOB_IDS="${JOB_IDS}:${JID}"
done

echo "Submitted $N_CHUNKS chunk jobs"

MERGE_CMD="python3 -c \"
import glob, h5py, numpy as np, sys
files = sorted(glob.glob('${OUT_DIR}/chunk_*.h5'))
if not files:
    print('No chunk files found'); sys.exit(1)
merged = {}
for f in files:
    with h5py.File(f, 'r') as h:
        for k in h['results']:
            arr = h['results'][k][:]
            merged[k] = np.concatenate([merged[k], arr]) if k in merged else arr.copy()
with h5py.File('${OUT_DIR}/merged_reco_results.h5', 'w') as h:
    g = h.create_group('results')
    for k, v in merged.items():
        g.create_dataset(k, data=v)
print(f'Merged {len(files)} chunks, {len(merged.get(\"rho\", []))} events')
\""

MERGE_JID=$(sbatch --parsable \
    --job-name="reco3d_merge" \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem="2GB" --time="00:10:00" \
    --dependency="afterany${JOB_IDS}" \
    --output="${OUT_DIR}/slurm_outputs/slurm_%j_merge.out" \
    --wrap="$MERGE_CMD")

echo "Merge job: $MERGE_JID (runs after all chunks complete)"
echo "Output: ${OUT_DIR}/merged_reco_results.h5"
