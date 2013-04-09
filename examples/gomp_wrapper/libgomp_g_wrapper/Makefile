TAU_MAKEFILE ?=../../../include/Makefile
include $(TAU_MAKEFILE)

TARGET=libgomp_g_wrap$(TAU_APPLE_SHLIBX)
CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS)  $(TAU_INCLUDE) $(TAU_MPI_INCLUDE)
EXTRA_FLAGS=-I. -fPIC -g -fopenmp

$(TARGET): libgomp_g_wrap.o omp_collector_util.o
	$(CC) $(TAU_SHFLAGS) $@ *.o $(TAU_SHLIBS) -ldl 

libgomp_g_wrap.o: wr.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@

omp_collector_util.o: omp_collector_util.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@

clean:
	/bin/rm -f libgomp_g_wrap.o omp_collector_util.o $(TARGET)

