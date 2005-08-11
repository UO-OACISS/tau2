CXX=gcc
INCLUDE=-I/opt/osshpc/mpich-1.2.5/32/ch_shmem/include/
LIB=-L/opt/osshpc/mpich-1.2.5/32/ch_shmem/lib -lmpich

computePi: computePi.cpp
	$(CXX) $(INCLUDE)  computePi.cpp -o computePi $(LIB)

clean:
	rm computePi
