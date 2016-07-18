RM = /bin/rm -rf

CC = oshcc
CFLAGS = -Wall

LD = $(CC)
LDFLAGS =

OBJ = rotput.o

LIBS =

ROTPUT = rotput

##############################################

.PHONY: all clean run

all: $(ROTPUT)

run: $(ROTPUT)
	oshrun -np 4 tau_exec -T serial,shmem,pdt -shmem ./$(ROTPUT)
	pprof

$(ROTPUT): $(OBJ)
	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o: %.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJ) $(ROTPUT)
	$(RM) *.trc *.edf profile* MULT* *.slog2
