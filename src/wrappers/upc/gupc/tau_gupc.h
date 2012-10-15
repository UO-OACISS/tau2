#ifdef __UPC_PUPC_INST__
#undef __UPC_PUPC_INST__
#define NEED__UPC_PUPC_INST__
#endif

#pragma pupc off

#include <upc.h>

#ifdef NEED__UPC_PUPC_INST__
#define __UPC_PUPC_INST__
#endif
