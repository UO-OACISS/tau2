#ifndef TAU_PTHREAD_GLOBAL_H
#define TAU_PTHREAD_GLOBAL_H

/* These functions are useful for using pthread functions to
 * emulate Thread Local Storage. 
 *
 * This class is used in the OpenMPLayer.cpp and TauCAPI.cpp 
 * files when thread local storage is unavailable (PGI for example).
 * */

#include "pthread.h"

/* This structure holds thread-specific global data structures */
struct _tau_global_data {
  int insideTAU;  // prevents re-entry into TAU
  int threadID;   // the thread's TAU ID
  int lightsOut;  // Application has exited, TAU is finalizing
};

class TauGlobal {
private:
  /* The pointer to the thread-specific global data */
  struct _tau_global_data* _value;
  /* The pthread key for getting/setting the data */
  static pthread_key_t* _key;
  /* This class is a process-level singleton, this is the object pointer */
  static TauGlobal* _instance;
  /* This method will initialize the data for a new thread */
  struct _tau_global_data* tau_set_specific();
  /* private constructor to prevent multiple instances */
  TauGlobal() { tau_set_specific(); }
public:
  /* public access to the singleton object */
  static TauGlobal getInstance() { 
    if (_instance == NULL) 
      _instance = new TauGlobal();
    return *_instance; 
  };
  /* public access to the thread-specific global data */
  struct _tau_global_data* getValue();
  /* public access to the key for destruction/cleanup (internal use only) */
  pthread_key_t* getKey() {return _key;};
};

#endif //TAU_PTHREAD_GLOBAL_H
