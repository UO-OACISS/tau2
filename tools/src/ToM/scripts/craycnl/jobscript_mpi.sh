#!/bin/bash
#PBS -A CSC023
#PBS -N PFLOTRAN
#PBS -j oe
#PBS -l walltime=1:00:00

# -l size will be determined by the script that submits the job scripts

export BIN_NAME=pflotran-collate
export EXPR_NAME=pflotran_2b
export OUT_NAME=Results_MonMPI

export EXPR_HOME=$LUSTRE_HOME/experiments/PFLOTRAN-TAU/$EXPR_NAME

cd $EXPR_HOME

export RESULT_NAME=Results.$PBS_NNODES.$PBS_JOBID
mkdir -p $OUT_NAME/$RESULT_NAME

export PROFILEDIR=$EXPR_HOME/$OUT_NAME/$RESULT_NAME

aprun -n $PBS_NNODES ./$BIN_NAME -file_output no -flow_mat_type aij -flow_ksp_type ibcgs -flow_ksp_lag_norm -tran_ksp_type ibcgs -log_summary
