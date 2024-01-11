#!/bin/bash

mkdir -p $1/report

## Number of runs
NB_RUNS=$4
echo NB_RUNS=$NB_RUNS

## Task IDs
TASK_ID_MIN=0
TASK_ID_MAX=$((NB_RUNS - 1))

echo "#!/bin/bash" > jobscript.sh
echo "#SBATCH --job-name=regp_bench" >> jobscript.sh
echo "#SBATCH -o $1/report/output.%a.out" >> jobscript.sh
echo "#SBATCH -e $1/report/error.%a.err" >> jobscript.sh

echo "export OUTPUT_DIR=$1" >> jobscript.sh
echo "export PROBLEM=$2" >> jobscript.sh
echo "export STRATEGY=$3" >> jobscript.sh
echo "export SMC_METHOD=restart" >> jobscript.sh

if [ -n "$MODULE_LOAD" ]; then
    echo "module load $MODULE_LOAD" >> jobscript.sh
fi

if [ -f "./regp/bin/activate" ]; then
    ## without conda
    echo "source ./regp/bin/activate" >> jobscript.sh
else
    ## with conda
    echo "source activate ./regp" >> jobscript.sh
fi

echo "python3 -u ./run/bench_optim.py" >> jobscript.sh

## Submit the job
if command -v sbatch &> /dev/null
then
    ## Run with the SLURM scheduler
    sbatch -a $TASK_ID_MIN-$TASK_ID_MAX jobscript.sh
else
    ## Run locally
    for a in $(seq $TASK_ID_MIN 1 $TASK_ID_MAX)
    do
	export SLURM_ARRAY_TASK_ID=$a
	bash jobscript.sh
    done
fi

rm jobscript.sh
