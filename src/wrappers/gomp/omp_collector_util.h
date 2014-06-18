/* 
 * Collector API implementation for the GOMP library,
 * based on the Collector API implementation for Open64,
 * a research compiler at the University of Houston.
 * For more information:
 * Open64 - http://www.cs.uh.edu/~hpctools
 * TAU    - http://tau.uoregon.edu/
 * GOMP   - http://www.gnu.org/projects/GOMP
 *
 */

#ifndef _OMP_COLLECTOR_UTIL_H
#define _OMP_COLLECTOR_UTIL_H

#include "omp_collector_api.h"

#ifdef __cplusplus
extern "C" {
#endif

extern char OMP_EVENT_NAME[35][50];
extern char OMP_STATE_NAME[11][50];

int __omp_collector_api(void *arg);

void __omp_collector_init(void);

OMP_COLLECTOR_API_THR_STATE __ompc_set_state(OMP_COLLECTOR_API_THR_STATE state);
void __ompc_event_callback(OMP_COLLECTORAPI_EVENT event);

void incr_current_region_id(void);

#ifdef __cplusplus
}
#endif

#endif  /* _OMP_COLLECTOR_UTIL_H */
