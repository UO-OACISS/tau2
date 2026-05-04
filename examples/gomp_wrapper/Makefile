TAU_MAKEFILE ?=../../include/Makefile

CC=tau_cc.sh
LDFLAGS         = -g $(TAU_OPENMP_OPTION)
CFLAGS          = -g $(TAU_OPENMP_OPTION)
TARGET=matmult

OBJS=$(TARGET).o 
$(TARGET): $(OBJS)
	$(CC) -g $(TARGET).o -o $@ $(LDFLAGS) 

.c.o:
	$(CC) $(CFLAGS) -c $<

test: $(TARGET)
	./matmult

clean:
	/bin/rm -rf $(OBJS) $(TARGET) profile.* MULT* core.*
