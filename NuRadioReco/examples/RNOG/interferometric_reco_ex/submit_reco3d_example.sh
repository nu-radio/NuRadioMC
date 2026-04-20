#!/bin/bash
# SLURM submission for 3D interferometric reconstruction.
#
# Splits input NUR files across parallel jobs. Each chunk processes
# a subset of files and writes results to a separate HDF5 file. After all
# chunks complete, a merge job concatenates them into merged_reco_results.h5.
#
# Usage:
#   bash submit_reco3d_example.sh --config <config.yaml> --data-dir <dir> \
#       --output-dir <dir> --account <acct> [options]
#
# Required:
#   --config       Reco config YAML (e.g., configs/reco3d_neutrino_gzk.yaml)
#   --data-dir     Directory containing input .nur files
#   --output-dir   Where to write per-chunk HDF5 results
#   --account      SLURM account
#
# Optional:
#   --mode         hw, rx, or rxtx (default: hw)
#   --n-chunks     Number of parallel jobs (default: 100)
#   --partition    SLURM partition (default: cluster default)
#   --mem          Memory per job (default: 4GB)
#   --walltime     Override walltime (default: 30min for hw, 3h for rx/rxtx)

set -euo pipefail

# Defaults
MODE="hw"
N_CHUNKS=100
PARTITION=""
MEM="4GB"
WALLTIME=""

# Parse named arguments
CONFIG=""
DATA_DIR=""
OUT_DIR=""
ACCOUNT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)     CONFIG="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUT_DIR="$2"; shift 2 ;;
        --account)    ACCOUNT="$2"; shift 2 ;;
        --mode)       MODE="$2"; shift 2 ;;
        --n-chunks)   N_CHUNKS="$2"; shift 2 ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --mem)        MEM="$2"; shift 2 ;;
        --walltime)   WALLTIME="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: submit_reco3d_example.sh --config <yaml> --data-dir <dir> --output-dir <dir> --account <acct> [options]"
            exit 1 ;;
    esac
done

if [ -z "$CONFIG" ] || [ -z "$DATA_DIR" ] || [ -z "$OUT_DIR" ] || [ -z "$ACCOUNT" ]; then
    echo "Missing required arguments. Need: --config, --data-dir, --output-dir, --account"
    exit 1
fi

if [ -z "$WALLTIME" ]; then
    case "$MODE" in
        hw)   WALLTIME="00:30:00" ;;
        rx)   WALLTIME="03:00:00" ;;
        rxtx) WALLTIME="03:00:00" ;;
        *)
            echo "Unknown mode: $MODE (use hw, rx, or rxtx)"
            exit 1 ;;
    esac
fi

# Build common sbatch args
SBATCH_ARGS=(--account "${ACCOUNT}" --nodes 1 --ntasks 1 --cpus-per-task 1)
if [ -n "$PARTITION" ]; then
    SBATCH_ARGS+=(--partition "${PARTITION}")
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
DRIVER="${SCRIPT_DIR}/interferometric_reco_3d_advanced.py"

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
echo "Mode: $MODE, Walltime: $WALLTIME, Memory: $MEM, Account: $ACCOUNT"

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
        "${SBATCH_ARGS[@]}" \
        --mem="${MEM}" --time="${WALLTIME}" \
        --output="${OUT_DIR}/slurm_outputs/slurm_%j_chunk${i}.out" \
        --wrap="export PYTHONPATH=${REPO_ROOT}; export NURADIO_TABLE_DIR=${NURADIO_TABLE_DIR:-/path/to/tables}; export NURADIO_DETECTOR_DIR=${NURADIO_DETECTOR_DIR:-/path/to/detector}; python ${DRIVER} --config ${CONFIG} --mode ${MODE} -i ${FILES_ARG} -o ${OUT_DIR}/chunk_${i}.h5")

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
print('Merged %d chunks, %d events' % (len(files), len(merged.get('rho', []))))
\""

MERGE_JID=$(sbatch --parsable \
    --job-name="reco3d_merge" \
    "${SBATCH_ARGS[@]}" \
    --mem="2GB" --time="00:10:00" \
    --dependency="afterany${JOB_IDS}" \
    --output="${OUT_DIR}/slurm_outputs/slurm_%j_merge.out" \
    --wrap="$MERGE_CMD")

echo "Merge job: $MERGE_JID (runs after all chunks complete)"
echo "Output: ${OUT_DIR}/merged_reco_results.h5"
