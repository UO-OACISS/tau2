KOKKOS_PATH = /storage/users/jalcaraz/spackkkk/spack/opt/spack/linux-ubuntu20.04-cascadelake/gcc-9.4.0/kokkos-4.3.00-sdwsar4ewnadahzp6g3oyz44yftjxrwv
SRC = $(wildcard *.cpp)
vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

OBJ = $(notdir $(SRC:.cpp=.o))
EXE = $(notdir $(SRC:.cpp=.exe))
LIB =
LINK=g++
CXX=g++
KOKKOS_LIBS=-L$(KOKKOS_PATH)/lib64 -lkokkoscore -ldl
KOKKOS_CXXFLAGS=-I$(KOKKOS_PATH)/include -std=c++14 -fopenmp
KOKKOS_LDFLAGS=-std=c++14 -fopenmp

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
