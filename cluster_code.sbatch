#!/usr/bin/env bash
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --constrain='balrog|belegost|gondor|khazadum|rivendell|shelob|shire|smaug'

source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate env_torch

python3 /Midgard/home/areichlin/OfflineRL/simple_script.py --offline ${OFFLINE} --kde ${KDE} --beta ${BETA} --offset ${OFFSET} --temperature ${T}