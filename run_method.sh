set -e

## $1 = output directory
if [ -z "$1" ]; then
    echo Argument \$1 missing: output directory
    exit 1
fi
OUTPUT_ROOT=$1
echo OUTPUT_ROOT=$OUTPUT_ROOT

## $2 = problem name
if [ -z "$2" ]; then
    echo Argument \$2 missing: problem name
    exit 1
fi
PROBLEM=$2
echo PROBLEM=$PROBLEM

## $3 = method
if [ -z "$3" ]; then
    echo Argument \$3 missing: strategy
    exit 1
fi
METHOD=$3
echo METHOD=$METHOD

## $4 = number of runs
if [ -z "$4" ]; then
    echo Argument \$4 missing: number of runs
    exit 1
fi
NB_RUNS=$4
echo NB_RUNS=$NB_RUNS

## $5 = sequential strategy
if [ -z "$5" ]; then
    echo Argument \$5 missing: sequential strategy
    exit 1
fi
SEQUENTIAL_STRATEGY=$5
echo SEQUENTIAL_STRATEGY=$SEQUENTIAL_STRATEGY

## Task IDs
TASK_ID_MIN=0
TASK_ID_MAX=$((NB_RUNS - 1))

## SMC flavour
if [ $SEQUENTIAL_STRATEGY = "EI" ]; then
    SMC_METHOD=restart
else
    SMC_METHOD=subset
fi


echo SMC_METHOD=$SMC_METHOD

OUTPUT_DIR=$OUTPUT_ROOT/$SEQUENTIAL_STRATEGY/$PROBLEM/$METHOD
echo OUTPUT_DIR=$OUTPUT_DIR

mkdir -p $OUTPUT_DIR

echo "#!/bin/bash" > jobscript.sh
echo "#SBATCH --job-name=regp_bench_$PROBLEM-$METHOD" >> jobscript.sh
echo "#SBATCH --output=$OUTPUT_DIR/output.%a.out" >> jobscript.sh
echo "#SBATCH --error=$OUTPUT_DIR/error.%a.err" >> jobscript.sh

# Adapt these SLURM directives to your setting:
# echo "#SBATCH --account=XXX"        >> jobscript.sh
# echo "#SBATCH --partition=cpu_long" >> jobscript.sh
# echo "#SBATCH --time=19:59:00"      >> jobscript.sh
echo "#SBATCH --nodes=1" >> jobscript.sh
echo "#SBATCH --ntasks=1" >> jobscript.sh
echo "#SBATCH --cpus-per-task=1" >> jobscript.sh

echo "export OUTPUT_DIR=$OUTPUT_DIR" >> jobscript.sh
echo "export PROBLEM=$PROBLEM" >> jobscript.sh
echo "export STRATEGY=$METHOD" >> jobscript.sh
echo "export SMC_METHOD=$SMC_METHOD" >> jobscript.sh
echo "export ALGO=$SEQUENTIAL_STRATEGY" >> jobscript.sh

if [ -n "$MODULE_LOAD" ]; then
    echo "module load $MODULE_LOAD" >> jobscript.sh
fi

## activate virtual env located at ./regp, if there is one
if [ -d "./regp/" ]; then
    if [ -f "./regp/bin/activate" ]; then
	## without conda
	echo "source ./regp/bin/activate" >> jobscript.sh
    else
	## with conda
	echo "source activate ./regp" >> jobscript.sh
    fi
fi

echo "export LANG=C" >> jobscript.sh
echo 'echo Starting at: `date`' >> jobscript.sh
# FIXME: use modules?
echo "PYTHONPATH=utils python3 -u ./run/bench.py" >> jobscript.sh
echo 'echo Finishing at: `date`' >> jobscript.sh

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
