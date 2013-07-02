/* These functions are useful for using pthread functions to
 * emulate Thread Local Storage. 
 * This class is used in the OpenMPLayer.cpp and TauCAPI.cpp 
 * files when thread local storage is unavailable (PGI for example).
 * */

#include "TauPthreadGlobal.h"
#include "stdlib.h"

/* Initialize the static member variables */
TauGlobal* TauGlobal::_instance = NULL;
pthread_key_t* TauGlobal::_key = NULL;

/* this is the destructor called when the thread exits 
 * Not really necessary, but good practice to do */
void globalDestructor(void *value) {
  free(value);
  pthread_setspecific(*(TauGlobal::getInstance().getKey()), NULL);
}

/* When a new thread is created, this function will be called
 * once, if the _value* member variable is null. The first time
 * it is called, the key is created. */
struct _tau_global_data* TauGlobal::tau_set_specific() { 
  if (_key == NULL) {
    _key = (pthread_key_t*)malloc(sizeof(pthread_key_t));
    pthread_key_create(_key, globalDestructor);
  }
  _value = (struct _tau_global_data*)malloc(sizeof(_tau_global_data));
  _value->insideTAU = 0;
  _value->threadID = -1;
  pthread_setspecific(*_key, _value);
  return _value;
}

/* return the thread-specific global data */
struct _tau_global_data* TauGlobal::getValue() { 
  struct _tau_global_data *tmp = (struct _tau_global_data*)pthread_getspecific(*_key);
  if (tmp == NULL) {
    tmp = tau_set_specific();
  }
  return tmp;
}

