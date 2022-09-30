CFLAGS += $(shell pkg-config --cflags starpu-1.3) -g
LDFLAGS += $(shell pkg-config --libs starpu-1.3) -g

mult:  mult.o

clean:
	rm -f mult *.o
