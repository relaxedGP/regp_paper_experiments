#!/bin/bash
set -e

## $1 = output directory
if [ -z "$1" ]; then
    echo Argument \$1 missing: output directory
    exit 1
fi
OUTPUT_DIR=$1
echo OUTPUT_DIR=$OUTPUT_DIR

## $2 = problem name
if [ -z "$2" ]; then
    echo Argument \$2 missing: problem name
    exit 1
fi
PROBLEM=$2
echo PROBLEM=$PROBLEM

## $3 = strategy
if [ -z "$3" ]; then
    echo Argument \$3 missing: strategy
    exit 1
fi
STRATEGY=$3
echo STRATEGY=$STRATEGY

## $4 = number of runs
if [ -z "$4" ]; then
    echo Argument \$4 missing: number of runs
    exit 1
fi
NB_RUNS=$4
echo NB_RUNS=$NB_RUNS

## Task IDs
TASK_ID_MIN=0
TASK_ID_MAX=$((NB_RUNS - 1))

## SMC flavour
SMC_METHOD=restart
echo SMC_METHOD=$SMC_METHOD

mkdir -p $OUTPUT_DIR/report

echo "#!/bin/bash" > jobscript.sh
echo "#SBATCH --job-name=regp_bench" >> jobscript.sh
echo "#SBATCH -o $OUTPUT_DIR/report/output.%a.out" >> jobscript.sh
echo "#SBATCH -e $OUTPUT_DIR/report/error.%a.err" >> jobscript.sh

echo "export OUTPUT_DIR=$OUTPUT_DIR" >> jobscript.sh
echo "export PROBLEM=$PROBLEM" >> jobscript.sh
echo "export STRATEGY=$STRATEGY" >> jobscript.sh
echo "export SMC_METHOD=$SMC_METHOD" >> jobscript.sh

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
