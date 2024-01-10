# regp_paper_experiments

Use python=3.8.

Installation :
```
pip3 install torch

git clone git@github.com:johncoltrane1/gpmp.git
cd gpmp
git checkout 16_10_2023_regp_paper_experiments
pip3 install -e .

cd ..

git clone git@github.com:johncoltrane1/gpmp-contrib.git
cd gpmp-contrib
git checkout 19_12_2023_regp_paper_experiments
pip3 install -e .

cd ..
```

The script for launching slurm is `run_ego.sh`. E.g.,
```
bash run_ego.sh concentration_run goldsteinprice Concentration 30
```
This launchs 30 repetitions of the "Concentration" heuristic on the Goldstein-Price function. Results and logs are written in `concentration_run`.