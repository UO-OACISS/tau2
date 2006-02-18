CC=tau_cc.sh -tau_makefile=../../../include/Makefile -tau_options='-optTauSelectFile=select.tau -optVerbose -optKeepFiles '

APP=loop_test
$(APP): $(APP).o
	$(CC) $(APP).o -o $@ 
$(APP).o: $(APP).c
	$(CC) -c $(APP).c 
clean:
	/bin/rm -f *.o *.pdb *.inst.* $(APP) profile.*
