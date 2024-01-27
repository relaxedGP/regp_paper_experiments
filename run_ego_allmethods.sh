set -e

CASE=$1
REP=$2
RUN_EGO=bash run_ego.sh

METHOD=None
$RUN_EGO ego_results/$METHOD $CASE $METHOD $REP

METHOD=Constant
$RUN_EGO ego_results/$METHOD $CASE $METHOD $REP

METHOD=Concentration
$RUN_EGO ego_results/$METHOD $CASE $METHOD $REP
