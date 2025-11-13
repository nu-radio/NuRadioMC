#!/bin/bash
#SBATCH --job-name="gen_intf_data"
#SBATCH --time=0-6:00:00
#SBATCH -N 1 -n 14
#SBATCH -o /vol/astro5/lofar/tgottmer/logs/%j_%x_%a.out
#SBATCH -e /vol/astro5/lofar/tgottmer/logs/%j_%x_%a.log
#SBATCH --chdir=/vol/astro5/lofar/tgottmer/
#SBATCH --mail-type=END
#SBATCH --mail-user=tjibbegottmer@ru.nl
#SBATCH --mem=96G

WORK_DIR=/vol/astro5/lofar/tgottmer/

source /vol/astro5/lofar/tgottmer/nrr-venv/bin/activate
python gen_data.py

deactivate
exit 0
