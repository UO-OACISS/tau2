include $(HOME)/tau2/x86_64/lib/Makefile.tau-pdt
#CC=tau_cc.sh -tau_makefile=$(HOME)/tau2/x86_64/lib/Makefile.tau-pdt -tau_options='-optTauSelectFile=select.tau -optVerbose -optKeepFiles '
CC=$(TAU_COMPILER) gcc 

APP=loop_test
$(APP): $(APP).o
	$(CC) $(APP).o -o $@ 
$(APP).o: $(APP).c
	$(CC) -c $(APP).c 
clean:
	/bin/rm -f *.o *.pdb *.inst.* $(APP) profile.*
