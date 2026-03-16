#!/bin/bash
#SBATCH --array=0-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --account=open
#SBATCH -o ./slurm_outputs/slurm-tables-%A_%a.out

export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $NUMBA_CACHE_DIR

# All VPOL channels used in 3D reco
CHANNELS=(0 1 2 3 5 6 7 9 10 22 23)
CH=${CHANNELS[$SLURM_ARRAY_TASK_ID]}

STATION=${1:-23}
MODE=${2:-"all"}
DET_DATE=${3:-"2022-10-01"}
OUTPUT_DIR=${4:-"./output/station${STATION}"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/rz_lookup_table_creator_inice.py"

echo "Station $STATION channel $CH mode $MODE using $SLURM_CPUS_PER_TASK threads"
echo "Detector date: $DET_DATE"
echo "Output: $OUTPUT_DIR"

python3 $SCRIPT \
    --station $STATION \
    --channel $CH \
    --mode $MODE \
    --num_threads $SLURM_CPUS_PER_TASK \
    --output-dir $OUTPUT_DIR \
    --det-date $DET_DATE
