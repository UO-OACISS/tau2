default: offload

sample_native.o: sample.c
	tau_cc.sh -openmp -mmic -c $< -o $@

sample_host.o: sample.c
	tau_cc.sh -openmp -no-offload -c $< -o $@

sample.o: sample.c
	tau_cc.sh -openmp -c $< -o $@

native: sample_native.o
	tau_cc.sh -openmp -mmic sample_native.o -o $@
	
host: sample_host.o
	tau_cc.sh -openmp -no-offload sample_host.o -o $@

offload: sample.o
	tau_cc.sh -tau_options='-optVerbose -optMICOffload' -openmp sample.o -o $@

clean:
	rm -f *.o native host offload
