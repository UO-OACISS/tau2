/**********************************************************************
 *user_event_data.h
 *
 *user_event_data object stores the number of events, max value, min 
 *value, mean value, and the sum square value.  This header file  also
 *declares an empty constructor, a copy constructor, a deconstructor, 
 *and overloaded versions of the = and += operators. 
 *
 *********************************************************************/

#include "tau_platforms.h"

/* Macros for min/max. To avoid <algobase> and gcc-3.0 problems, we define: */
#define TAU_MIN(a,b) (((a)<(b))?(a):(b))
#define TAU_MAX(a,b) (((a)>(b))?(a):(b))

class user_event_data {
  public :
    double     numevents;
    double     maxvalue;
    double     minvalue;
    double     meanvalue;
    double     sumsqr;
    user_event_data(double ne, double maxv, double minv, double meanv, double sumsqrv);
    user_event_data();
    user_event_data(const user_event_data& X);
    user_event_data& operator= (const user_event_data& X);  
    user_event_data& operator+= (const user_event_data& X);
    ~user_event_data();
};

/***************************************************************************
 * $RCSfile: user_event_data.h,v $   $Author: ntrebon $
 * $Revision: 1.2 $   $Date: 2002/07/25 20:50:00 $
 * TAU_VERSION_ID: $Id: user_event_data.h,v 1.2 2002/07/25 20:50:00 ntrebon Exp $
 ***************************************************************************/

