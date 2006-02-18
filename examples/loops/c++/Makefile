CC=tau_cxx.sh -tau_makefile=../../../include/Makefile -tau_options='-optTauSelectFile=select.tau -optVerbose -optKeepFiles '

APP=loop_test
$(APP): $(APP).o
	$(CC) $(APP).o -o $@ 
$(APP).o: $(APP).cpp
	$(CC) -c $(APP).cpp 
clean:
	/bin/rm -f *.o *.pdb *.inst.* $(APP) profile.*
