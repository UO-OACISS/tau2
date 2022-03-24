HIPCC=hipcc

TARGET=hcc

SOURCES = vectoradd_hip.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./vectoradd_hip.exe

.PHONY: test


all: $(EXECUTABLE) 

CXXFLAGS =-g

CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS) 
	$(HIPCC) $(OBJECTS) -o $@


run: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o profile*


