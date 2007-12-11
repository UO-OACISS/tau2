#!/bin/bash -x

#export PROFILEDIR=/var/taudata
#export TRACEDIR=/var/taudata

#TAU envs
export COUNTER1=LINUX_TIMERS
export COUNTER2=KTAU_schedule
export COUNTER3=KTAU_smp_apic_timer_interrupt
export COUNTER4=KTAU_INCL_smp_apic_timer_interrupt
export COUNTER5=KTAU_NUM_smp_apic_timer_interrupt

#export COUNTER6=KTAU_schedule_vol

#sicortex specific
#export COUNTER7=KTAU_sc1000_slow_interrupt
#export COUNTER8=KTAU_sc1000_dma_interrupt
#export COUNTER9=KTAU_sc1000_perfctr_interrupt



#BSP envs
numphases=100
#numphases=10000
comptime=4000
collective=BARRIER
strong=1
pin=8
timed=0

export BSP_TIMED_COMP=$timed
export BSP_PIN=$pin
export BSP_COMP_TIME=$comptime
export BSP_PHASES=$numphases
export BSP_STRONG=$strong
export BSP_COLLECTIVE=$collective

mpirun -np $1 ./bsp

#sicortex/slurm
###############
#srun -p sf0 -n $1 ./bsp
#trial clock sync
#srun -p sf0 -n 12 -N 4 --ntasks-per-node=3 ./bsp


