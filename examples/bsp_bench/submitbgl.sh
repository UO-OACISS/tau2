#!/bin/sh -x
#Meant to be run on BG/L for evaluation of noise-estimation analysis. There we run using "Selfish" as the noise-source. Needs hacking.
numphases=10000
comptime=0
collective=BARRIER
I=1000
P=1
D=80

DIR=strong.C$comptime.Ph$numphases.I$I.P$P.D$D.$collective

#setup env vars
TAUENV=COUNTER1=BGL_TIMERS::COUNTER2=CPU_TIME
BSPENV=BSP_SELFISH=1::BSP_COMP_TIME=$comptime::BSP_PHASES=$numphases::BSP_STRONG=1::BSP_COLLECTIVE=$collective
SELFISHENV=SELFISH_I=$I::SELFISH_P=$P::SELFISH_D=$D
#SELFISHENV=x=y

time=1200

obj=#full path to location of the object file to run>

mkdir -p $DIR
pushd $DIR

for C in 2048 1024 512 256 128 64 32
do

	mkdir -p N$C;
	pushd N$C;

	N=512
	if [ $C -gt 512 ]; then
		N=$C
	fi;

	env=$TAUENV::$BSPENV::$SELFISHENV

	cqsub -t$time -n$C -c$C -e$env $obj

	popd
done;

popd;
