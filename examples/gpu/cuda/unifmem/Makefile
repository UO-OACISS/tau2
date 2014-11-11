all: aplusb

aplusb: aplusb.o
	nvcc -o $@ $<

aplusb.o: aplusb.cu
	nvcc -g -c $< -o $@

clean: 
	rm -rf aplusb aplusb.o

run:
	tau_exec -T serial -cuda ./aplusb

