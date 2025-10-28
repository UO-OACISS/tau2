# Allow KOKKOS_PATH to be set from the environment or command line
KOKKOS_PATH ?=

# I. --- Find Kokkos Installation ---
# If KOKKOS_PATH is not set, try to find it using kokkos_launch_compiler
ifeq ($(KOKKOS_PATH),)
    KOKKOS_COMPILER_PATH := $(shell which kokkos_launch_compiler)

    ifeq ($(KOKKOS_COMPILER_PATH),)
        $(error KOKKOS_PATH is not set and 'kokkos_launch_compiler' could not be found in your PATH.)
    else
        # Infer KOKKOS_PATH from the location of the compiler script
        KOKKOS_PATH := $(shell dirname $(shell dirname $(KOKKOS_COMPILER_PATH)))
        $(info KOKKOS_PATH automatically set to: $(KOKKOS_PATH))
    endif
endif

# Check if the installation uses /lib or /lib64
KOKKOS_LIB64_EXISTS := $(wildcard $(KOKKOS_PATH)/lib64)
ifeq ($(KOKKOS_LIB64_EXISTS),)
    KOKKOS_LIBDIR := $(KOKKOS_PATH)/lib
    $(info KOKKOS library directory detected as: $(KOKKOS_PATH)/lib)
else
    KOKKOS_LIBDIR := $(KOKKOS_PATH)/lib64
    $(info KOKKOS library directory detected as: $(KOKKOS_PATH)/lib64)
endif

# II. --- Backend Detection ---
# Introspect the KokkosCore_config.h header to determine the primary backend.
# The order of checks is important: CUDA -> HIP -> OpenMP -> Serial.

# Default to a standard C++ compiler. These will be overridden if a backend is detected.
CXX := g++
LINK := g++
BACKEND_LDFLAGS :=

# Path to the configuration header
KOKKOS_CONFIG_H := $(KOKKOS_PATH)/include/KokkosCore_config.h

KOKKOS_HAVE_CUDA := $(shell grep -q "^#define KOKKOS_ENABLE_CUDA" $(KOKKOS_CONFIG_H) && echo "yes")
KOKKOS_HAVE_HIP := $(shell grep -q "^#define KOKKOS_ENABLE_HIP" $(KOKKOS_CONFIG_H) && echo "yes")

ifeq ($(KOKKOS_HAVE_CUDA), yes)
    $(info Detected Kokkos CUDA backend)
    CXX  := $(KOKKOS_PATH)/bin/nvcc_wrapper
    LINK := $(CXX)

    # Allow user to specify CUDA location, otherwise use a common default
    CUDA_HOME ?= /usr/local/cuda
    BACKEND_LDFLAGS := -L$(CUDA_HOME)/lib64 -lcudart

    # Add rpath to help the executable find shared libraries at runtime
    KOKKOS_LDFLAGS += -Wl,-rpath,$(KOKKOS_LIBDIR) -Wl,-rpath,$(CUDA_HOME)/lib64

else ifeq ($(KOKKOS_HAVE_HIP), yes)
    $(info Detected Kokkos ROCm/HIP backend)
    CXX  := hipcc
    LINK := $(CXX)

    # Allow user to specify ROCm location, otherwise use a common default
    ROCM_HOME ?= /opt/rocm
    BACKEND_LDFLAGS := -L$(ROCM_HOME)/lib -lamdhip64

    # Add rpath to help the executable find shared libraries at runtime
    KOKKOS_LDFLAGS += -Wl,-rpath,$(KOKKOS_LIBDIR) -Wl,-rpath,$(ROCM_HOME)/lib

endif

    # If not a GPU backend, check for OpenMP
    KOKKOS_HAVE_OPENMP := $(shell grep -q "^#define KOKKOS_ENABLE_OPENMP" $(KOKKOS_CONFIG_H) && echo "yes")
    ifeq ($(KOKKOS_HAVE_OPENMP), yes)
        $(info Detected Kokkos OpenMP backend)
        KOKKOS_CXXFLAGS += -fopenmp
        KOKKOS_LDFLAGS  += -fopenmp
    else
        $(info Detected Kokkos Serial backend)
    endif


KOKKOS_LIBS := -L$(KOKKOS_LIBDIR) -lkokkoscore -ldl
KOKKOS_LIBS += $(BACKEND_LDFLAGS)

SRC := $(wildcard *.cpp)
OBJ := $(notdir $(SRC:.cpp=.o))
EXE := $(notdir $(SRC:.cpp=.exe))

KOKKOS_CXXFLAGS += -I$(KOKKOS_PATH)/include -std=c++17

default: build

build: $(EXE)

test: build
	./$(EXE)

$(EXE): $(OBJ)
	$(LINK) $(KOKKOS_LDFLAGS) $(OBJ) $(KOKKOS_LIBS) -o $@

%.o: %.cpp
	$(CXX) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) -c $< -o $(notdir $@)

clean:
	rm -f *.o *.exe profile.*
