#!/bin/bash

mkdir -p $1/report

echo "#!/bin/bash" > jobscript.sh
echo "#SBATCH --job-name=regp_bench" >> jobscript.sh
echo "#SBATCH -o $1/report/output.%a.out" >> jobscript.sh
echo "#SBATCH -e $1/report/error.%a.err" >> jobscript.sh

echo "export OUTPUT_DIR=$1" >> jobscript.sh
echo "export PROBLEM=$2" >> jobscript.sh
echo "export STRATEGY=$3" >> jobscript.sh
echo "export SMC_METHOD=restart" >> jobscript.sh

echo "source ./regp/bin/activate" >> jobscript.sh
echo "python3 -u ./run/bench_optim.py" >> jobscript.sh

## Submit the job
if command -v sbatch &> /dev/null
then
    ## Run with the SLURM scheduler
    sbatch -a 0-$4 jobscript.sh
else
    ## Run locally
    for a in $(seq 0 1 $4)
    do
	export SLURM_ARRAY_TASK_ID=$a
	bash jobscript.sh
    done
fi

rm jobscript.sh
