RM = /bin/rm -rf

CC = oshcc
CFLAGS = -Wall

LD = tau_cc.sh -optLinkOnly
LDFLAGS =

OBJ = rotput.o

LIBS =

ROTPUT = rotput

##############################################

.PHONY: all clean run

all: $(ROTPUT)

run: $(ROTPUT)
	oshrun -np 4 ./$(ROTPUT)
	pprof

$(ROTPUT): $(OBJ)
	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o: %.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJ) $(ROTPUT)
	$(RM) *.trc *.edf profile* MULT* *.slog2
