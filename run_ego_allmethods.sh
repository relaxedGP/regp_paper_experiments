set -e

# export MODULE_LOAD=anaconda3/2022.10/gcc-11.2.0

CASE=$1
REP=$2
RUN_EGO="bash run_ego.sh"

METHOD=None
$RUN_EGO results $CASE $METHOD $REP

METHOD=Constant
$RUN_EGO results $CASE $METHOD $REP

METHOD=Concentration
$RUN_EGO results $CASE $METHOD $REP
