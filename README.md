# regp_paper_experiments

Use python>=3.8.

## Install

```
## Create & activate virtual environnement
python3 -m venv regp
source ./regp/bin/activate

## Alternative: using conda
# conda create --prefix `pwd`/regp python=3.11
# source activate ./regp

## Install troch
pip3 install torch

## Install gmpm fork
git clone -b regp_paper_experiments git@github.com:relaxedGP/gpmp.git
pip3 install -e gpmp

## Install gpmp-contrib fork
git clone -b regp_paper_experiments git@github.com:relaxedGP/gpmp-contrib.git
pip3 install -e gpmp-contrib
```

## Run benchmarks

The script for launching slurm is `run_allmethods.sh`. E.g.,
```
bash run_allmethods.sh goldsteinprice 30
```
This launchs 30 repetitions of the "Concentration", "Spatial", "Constant", and "None" heuristics on the Goldstein-Price function. Results and logs are written in `results`.
