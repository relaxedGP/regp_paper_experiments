set -e

# export MODULE_LOAD=anaconda3/2022.10/gcc-11.2.0

CASE=$1
REP=$2
RUN_EGO="bash run_method.sh"
RESULTS="results"

for METHOD in None Constant Concentration Spatial
do
  $RUN_EGO $RESULTS $CASE $METHOD $REP
done

