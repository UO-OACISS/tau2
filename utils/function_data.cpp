/****************************************************************************
 *Class function_data
 *
 *function_data objects hold information for each function.  Namely, the 
 *number of calls, number of subroutines, exclusive time, inclusive time,
 *standard deviation, and name.  The class defines an empty constructor,
 *a copy constructor, deconstructor, and overloaded versions of the = and 
 *+= operators
 *
 ***************************************************************************/

#include "function_data.h"

  //empty constructor.  Set all values to 0.
  function_data::function_data() {
    numcalls = numsubrs = 0;
    excl = incl = stddeviation = 0;
  }

  //copy constructor.  Set all values in new function_data object to values of 
  //function_data object
  function_data::function_data(const function_data& X) {
    numcalls = X.numcalls;
    numsubrs = X.numsubrs;
    excl  = X.excl;
    incl  = X.incl;
    stddeviation = X.stddeviation; 
  }
  
  //overloaded = operator.  
  function_data& function_data::operator= (const function_data& X) {
    numcalls = X.numcalls;
    numsubrs = X.numsubrs;
    excl = X.excl;
    incl = X.incl;
    stddeviation = X.stddeviation;
    return *this;
  }

  //overloaded += operator.  Increment all values and return new object.
  function_data& function_data::operator+= (const function_data& X) {
    numcalls += X.numcalls;
    numsubrs += X.numsubrs;
    excl += X.excl;
    incl += X.incl;
    stddeviation += X.stddeviation;
    return *this;
  }

  //empty deconstructor
  function_data::~function_data() { }

/***************************************************************************
 * $RCSfile: function_data.cpp,v $   $Author: ntrebon $
 * $Revision: 1.1 $   $Date: 2002/07/25 20:40:14 $
 * TAU_VERSION_ID: $Id: function_data.cpp,v 1.1 2002/07/25 20:40:14 ntrebon Exp $
 ***************************************************************************/

