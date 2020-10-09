CXX      := g++
CXXFLAGS := -pedantic-errors -Wall -Wextra -Werror -std=c++11 -pthread -g -O3
LDFLAGS  :=
TARGET   := maptest
INCLUDE  := -I.
SRC      := $(wildcard *.cpp)
HEADERS  := $(wildcard *.h)

OBJECTS  := $(SRC:%.cpp=%.o)

all: $(TARGET)

%.o: %.cpp $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@ $(LDFLAGS)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

.PHONY: all clean test

clean:
	-@rm -rvf $(OBJECTS) $(TARGET)

test: $(TARGET)
	./$(TARGET)