#ifndef TAU_SOS_H
#define TAU_SOS_H

#define TAU_SOS_INTERRUPT_PERIOD 2 // two seconds

#ifdef TAU_SOS
#include "VMPI.h"
static inline MPI_Comm TAU_SOS_MAP_COMMUNICATOR(MPI_Comm _c) {
    if (_c==MPI_COMM_WORLD && __vmpi_status.numpartitions != 0) {
        return VMPI_Get_partition_comm();
    } else if( _c==MPI_COMM_UNIVERSE) {
        return MPI_COMM_WORLD;
    }
}
#else
#define TAU_SOS_MAP_COMMUNICATOR(arg) arg
#endif

#ifdef __cplusplus
extern "C" {  // export a C interface for C++ codes
#else
#include <stdbool.h> // import bool support for C codes
#endif
void TAU_SOS_send_data(void);
void TAU_SOS_init(int * argc, char *** argv, bool threaded);
void TAU_SOS_stop_worker(void);
void TAU_SOS_finalize(void);
void TAU_SOS_send_data(void);
#ifdef __cplusplus
}
#endif

#endif // TAU_SOS_H
