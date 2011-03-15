#ifndef _TAU_MON_H
#define _TAU_MON_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

  /* Headers for Monitoring API calls. Must always be implemented. */
  void Tau_mon_onlineDump();

  /* Internal common Monitoring Framework supporting function headers.
     Do not define in non-monitoring configurations of TAU.
   */
#ifdef TAU_MONITORING
  /* Some monitoring transport will implement these functions */
  void Tau_mon_internal_onlineDump();
  void Tau_mon_connect();
  void Tau_mon_disconnect();
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_MON_H */
