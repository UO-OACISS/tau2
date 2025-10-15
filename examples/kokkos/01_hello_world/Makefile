KOKKOS_PATH ?=

ifeq ($(KOKKOS_PATH),)
    KOKKOS_COMPILER_PATH := $(shell which kokkos_launch_compiler)

    ifeq ($(KOKKOS_COMPILER_PATH),)
        $(error KOKKOS_PATH is not set and 'kokkos_launch_compiler' could not be found in your PATH.)
    else
        KOKKOS_PATH := $(shell dirname $(shell dirname $(KOKKOS_COMPILER_PATH)))
        $(info KOKKOS_PATH automatically set to: $(KOKKOS_PATH))
    endif
endif

KOKKOS_LIB64_EXISTS := $(wildcard $(KOKKOS_PATH)/lib64)

ifeq ($(KOKKOS_LIB64_EXISTS),)
    KOKKOS_LIBDIR := $(KOKKOS_PATH)/lib
    $(info KOKKOS library directory detected as: $(KOKKOS_PATH)/lib)
else
    KOKKOS_LIBDIR := $(KOKKOS_PATH)/lib64
    $(info KOKKOS library directory detected as: $(KOKKOS_PATH)/lib64)
endif

KOKKOS_LIBS = -L$(KOKKOS_LIBDIR) -lkokkoscore -ldl

SRC = $(wildcard *.cpp)
vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

OBJ = $(notdir $(SRC:.cpp=.o))
EXE = $(notdir $(SRC:.cpp=.exe))
LIB =
LINK=g++
CXX=g++
KOKKOS_CXXFLAGS=-I$(KOKKOS_PATH)/include -std=c++17 -fopenmp
KOKKOS_LDFLAGS=-std=c++17 -fopenmp

build: $(EXE)

test: $(EXE)
	./$(EXE)

$(EXE): $(OBJ)
	$(LINK) $(KOKKOS_LDFLAGS) $(LDFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean:
	rm -f *.o *.exe

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)
