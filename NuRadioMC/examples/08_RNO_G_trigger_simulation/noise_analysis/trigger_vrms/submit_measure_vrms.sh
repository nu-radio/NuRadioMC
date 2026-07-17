#!/bin/bash
#SBATCH --job-name=vrms_meas
#SBATCH --account=your_account
#SBATCH --partition=your_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH -o vrms_%j.out

# Measure full-pool trigger-path Vrms for one station.
# Edit the env block for your site, then:
#   sbatch --export=ALL,STATION=23,MODE=ft submit_measure_vrms.sh
# Required env: STATION; FT_NOISE_DIR (mode ft) or BURN_ROOT (mode burn); optional
# CLEAN_MASK, DETECTOR_FILE (omit to query MongoDB), MODE (default ft).

ENV_SETUP=${ENV_SETUP:-""}
PYTHONPATH_ADD=${PYTHONPATH_ADD:-""}
[ -n "$ENV_SETUP" ] && eval "$ENV_SETUP"
[ -n "$PYTHONPATH_ADD" ] && export PYTHONPATH="$PYTHONPATH_ADD:${PYTHONPATH:-}"

cd "$(dirname "$0")"
MODE=${MODE:-ft}
date
echo "host: $(hostname)  cpus: $SLURM_CPUS_PER_TASK  station: $STATION  mode: $MODE"

ARGS="--station $STATION --mode $MODE --n_jobs ${SLURM_CPUS_PER_TASK:-20}"
[ -n "$FT_NOISE_DIR" ] && ARGS="$ARGS --ft_noise_dir $FT_NOISE_DIR"
[ -n "$BURN_ROOT" ]    && ARGS="$ARGS --burn_root $BURN_ROOT"
[ -n "$CLEAN_MASK" ]   && ARGS="$ARGS --clean_mask $CLEAN_MASK"
[ -n "$DETECTOR_FILE" ] && ARGS="$ARGS --detector_file $DETECTOR_FILE"

python3 measure_trigger_vrms_full.py $ARGS
echo "exit: $?  end: $(date -Is)"
