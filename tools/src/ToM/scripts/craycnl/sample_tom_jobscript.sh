#!/bin/bash
#PBS -A CSC023
#PBS -N PFLOTRAN
#PBS -j oe
#PBS -l walltime=1:00:00,size=4488

export BIN_NAME=pflotran-mrnet
export EXPR_NAME=pflotran_2b
export OUT_NAME=Results_ToM

export EXPR_HOME=$LUSTRE_HOME/experiments/PFLOTRAN-TAU/$EXPR_NAME

cd $EXPR_HOME

export RESULT_NAME=Results.$PBS_NNODES.$PBS_JOBID
mkdir -p $OUT_NAME/$RESULT_NAME

export PROFILEDIR=$EXPR_HOME/$OUT_NAME/$RESULT_NAME
echo $PROFILEDIR

./startToM_craycnl.sh $PROFILEDIR 4488 ToM_FE 4104 12 374 ./$BIN_NAME -file_output no -flow_mat_type aij -flow_ksp_type ibcgs -flow_ksp_lag_norm -tran_ksp_type ibcgs -log_summary