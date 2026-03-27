#!/bin/bash
#SBATCH --job-name="gen_intf_data"
#SBATCH --time=1-00:00:00
#SBATCH -o /vol/astro5/lofar/tgottmer/logs/%A_%x_%a.out
#SBATCH -e /vol/astro5/lofar/tgottmer/logs/%A_%x_%a.log
#SBATCH --chdir=/vol/astro5/lofar/tgottmer/
#SBATCH --mem=10G
#SBATCH --arr=0-123

WORK_DIR=/vol/astro5/lofar/tgottmer/

source /vol/astro5/lofar/tgottmer/nrr-venv/bin/activate
python NuRadioMC/gen_lofar_data.py --chunk_id $SLURM_ARRAY_TASK_ID

deactivate
exit 0
