#ifndef __TAU_BG_HWP_COUNTERS_H__
#define __TAU_BG_HWP_COUNTERS_H__

#include <stdint.h>

#define COUNTER_CONFIGURATION_ERROR -1
#define COUNTER_START_ERROR         -2
#define COUNTER_READ_ERROR          -3
#define COUNTER_INACTIVE_ERROR      -4
#define COUNTER_REDUCE_ERROR        -5
#define COUNTER_IO_ERROR            -6

#define COUNTER_MODE_NOCORES         1
#define COUNTER_MODE_CORES01         2
#define COUNTER_MODE_CORES23         3
#define COUNTER_MODE_INVALID         4

#define COUNTER_ARRAY_LENGTH       256

#define OUTPUT_FILE        counters.dat

#define QUOTE_(x) #x
#define QUOTE(x) QUOTE_(x)

#endif /* __TAU_BG_HWP_COUNTERS_H__  */
