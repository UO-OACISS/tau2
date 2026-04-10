HIPCC=hipcc

TARGET=hcc

SOURCES = vectoradd_hip.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./vectoradd_hip.exe

SOURCES2 = vectoradd_hip_multi.cpp
OBJECTS2 = $(SOURCES2:.cpp=.o)

EXECUTABLE2=./vectoradd_hip_multi.exe

.PHONY: test


all: $(EXECUTABLE) $(EXECUTABLE2)

CXXFLAGS =-g

CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS) 
	$(HIPCC) $(OBJECTS) -o $@

$(EXECUTABLE2): $(OBJECTS2)
	$(HIPCC) $(OBJECTS2) -o $@

run: $(EXECUTABLE)
	$(EXECUTABLE)

run_multi: $(EXECUTABLE2)
	$(EXECUTABLE2)

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE2)
	rm -f $(OBJECTS) $(OBJECTS2)
	rm -f $(HIP_PATH)/src/*.o profile*


