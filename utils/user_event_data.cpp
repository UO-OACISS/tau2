/**********************************************************************
 *user_event_data Class
 *
 *user_event_data object stores the number of events, max value, min 
 *value, mean value, and the sum square value.  The class also defines
 *a constructor, an empty constructor, a copy constructor, a 
 *deconstructor, and overloaded versions of the = and += operators. 
 *
 *********************************************************************/

# include "user_event_data.h"

  user_event_data::user_event_data(double ne, double maxv, double minv, double meanv, double sumsqrv) 
    : numevents(ne), maxvalue(maxv), minvalue(minv), meanvalue(meanv),sumsqr(sumsqrv) { }


  //empty constructor
  user_event_data::user_event_data() {
    numevents = meanvalue = sumsqr = 0;
    maxvalue  = DBL_MIN;
    minvalue  = DBL_MAX;
  }

  //copy constructor
  user_event_data::user_event_data(const user_event_data& X) 
    : numevents(X.numevents), maxvalue(X.maxvalue), minvalue(X.minvalue),
      meanvalue(X.meanvalue), sumsqr(X.sumsqr) { }

  //overloaded = operator.
  user_event_data& user_event_data::operator= (const user_event_data& X)  
  {
    numevents 	= X.numevents;
    maxvalue 	= X.maxvalue;
    minvalue 	= X.minvalue;
    meanvalue	= X.meanvalue;
    sumsqr	= X.sumsqr;
    return *this;
  }

  //overloaded += operator.
  user_event_data& user_event_data::operator+= (const user_event_data& X) {
    maxvalue	= TAU_MAX (maxvalue, X.maxvalue);
    minvalue	= TAU_MIN (minvalue, X.minvalue);
    if (numevents+X.numevents != 0) {
      meanvalue 	= (meanvalue*numevents + X.meanvalue * X.numevents)/(numevents+X.numevents); 
    }
    else 
      meanvalue = 0;
    numevents 	+= X.numevents;
    sumsqr	+= X.sumsqr;

    return *this;
  }

  //empty deconstructor
  user_event_data::~user_event_data() { }


/***************************************************************************
 * $RCSfile: user_event_data.cpp,v $   $Author: ntrebon $
 * $Revision: 1.1 $   $Date: 2002/07/25 20:40:13 $
 * TAU_VERSION_ID: $Id: user_event_data.cpp,v 1.1 2002/07/25 20:40:13 ntrebon Exp $
 ***************************************************************************/

