MPI_OPTS=$(shell ./mpiopts.sh)
# Put your own mpi -I -L -l options here. 

HIPCC=hipcc 

SOURCES = vectoradd_hip.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./vectoradd_hip.exe

.PHONY: test


all: $(EXECUTABLE) 

CXXFLAGS =-g

CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS) 
	$(HIPCC) $(OBJECTS) -o $@ ${MPI_OPTS}

%.o: %.cpp
	$(HIPCC) -c $< -o $@ ${MPI_OPTS}


run: $(EXECUTABLE)
	mpirun -np 3  $(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o profile*


