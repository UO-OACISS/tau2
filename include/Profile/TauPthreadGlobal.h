/* These functions are useful for using pthread functions to
 * emulate Thread Local Storage. */

#include "pthread.h"

struct _tau_global_data {
  int insideTAU;
  int threadID;
};

class TauGlobal {
private:
  struct _tau_global_data* _value;
  static pthread_key_t* _key;
  static TauGlobal* _instance;
  struct _tau_global_data* tau_set_specific();
  TauGlobal() { tau_set_specific(); }
public:
  static TauGlobal getInstance() { 
    if (_instance == NULL) 
      _instance = new TauGlobal();
    return *_instance; 
  };
  struct _tau_global_data* getValue();
  pthread_key_t* getKey() {return _key;};
};

