#ifndef __BSP_H
#define __BSP_H

#ifdef BGL_TIMERS
/* header files for BlueGene/L */
#include <bglpersonality.h>
#include <rts.h>
#endif // BGL_TIMERS

#ifdef __cplusplus
extern "C" {
#endif
//now what? only bgl for now...
double bsp_rdtsc();
double bsp_get_tsc();
#ifdef __cplusplus
}
#endif

#endif /* __BSP_H */
