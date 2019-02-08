TAU_MAKEFILE ?=../../include/Makefile

include $(TAU_MAKEFILE)

MCC              = TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cc.sh 

CFLAGS = -O2   

OBJS =  main.o 

run: run_bcast
	mpirun -np 4 ./run_bcast 2 2

run_bcast:   ${OBJS}
	${MCC} ${CFLAGS} ${OPTFLAGS} -o  $@ ${OBJS}

main.o:		main.c
	${MCC} ${CFLAGS} ${OPTFLAGS} -c $?


clean:
	- rm -rf *.o *.bin core run_bcast a.out silc* *.tar *.pdb *.inst.* profile* MULT*
