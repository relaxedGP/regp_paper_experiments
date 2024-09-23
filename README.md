# regp_paper_experiments

This repository contains the scripts used for the numerical bencharmks of the article:

Sébastien J. Petit (§), Julien Bect (†), and Emmanuel Vazquez (†)
**Relaxed Gaussian Process Interpolation: a Goal-Oriented Approach to Bayesian Optimization**
([arXiv:2206.03034](https://arxiv.org/abs/2206.03034))

It relies on code available on two other repositories: https://github.com/relaxedGP/gpmp.git and https://github.com/relaxedGP/gpmp-contrib.git.

(§) Laboratoire National de Métrologie et d’Essais, 78197 Trappes Cedex, France
(†) Université Paris-Saclay, CNRS, CentraleSupélec,
[Laboratoire des signaux et systèmes](https://l2s.centralesupelec.fr/),
Gif-sur-Yvette, France.  

![Illustration](images/regp_illustration.png)

## Requirements

### Installation for Python

Use Python>=3.8.

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
git clone -b regp_paper_experiments https://github.com/relaxedGP/gpmp.git
pip3 install -e gpmp

## Install gpmp-contrib fork
git clone -b regp_paper_experiments https://github.com/relaxedGP/gpmp-contrib.git
pip3 install -e gpmp-contrib
```

### Reproducing the benchmarks

The script for launching slurm is `run_allmethods.sh`. E.g.,
```
bash run_allmethods.sh goldsteinprice 30 EI
```
This launchs 30 repetitions of the Expected improvement algorithm with the "Concentration", "Spatial", "Constant", and "None" heuristics on the Goldstein-Price function. Results and logs are written in `results`. Other possible algorithms are "straddle" and UCB with, for instance, "UCB10" for q = 0.1.
