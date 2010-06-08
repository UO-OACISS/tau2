#ifndef _TAU_MON_H
#define _TAU_MON_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifdef TAU_MONITORING
  /* some monitoring module will define these functions */
  void Tau_mon_connect();
  void Tau_mon_disconnect();
#else
  /* default - macro them away */
#define Tau_mon_connect()
#define Tau_mon_disconnect()
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_MON_H */
