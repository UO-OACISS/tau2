/////////////////////////////////////////////////
//Class definintion file for the PCL_Layer class.
//
//Author:   Robert Ansell-Bell
//Created:  February 2000
//
/////////////////////////////////////////////////

#ifndef _PAPI_LAYER_H_
#define _PAPI_LAYER_H_

#ifdef TAU_PAPI
extern "C" {
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"
}


  struct ThreadValue{
  int ThreadID;
  long long CounterValue;
  };



class PapiLayer
{

  //No need to define constructors and destructors.
  //The default ones will do.

  public:
    //Default getCounters ... without sychronization of resources.
    static long long getCounters(int tid);
};

#endif /* TAU_PAPI */
#endif /* _PAPI_LAYER_H_ */

/////////////////////////////////////////////////
//
//End PCL_Layer class definition.
//
/////////////////////////////////////////////////




