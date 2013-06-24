/* These functions are useful for using pthread functions to
 * emulate Thread Local Storage. */

#include "TauPthreadGlobal.h"
#include "stdlib.h"

TauGlobal* TauGlobal::_instance = NULL;
pthread_key_t* TauGlobal::_key = NULL;

void globalDestructor(void *value) {
  free(value);
  pthread_setspecific(*(TauGlobal::getInstance().getKey()), NULL);
}

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

struct _tau_global_data* TauGlobal::getValue() { 
  struct _tau_global_data *tmp = (struct _tau_global_data*)pthread_getspecific(*_key);
  if (tmp == NULL) {
    tmp = tau_set_specific();
  }
  return tmp;
}

