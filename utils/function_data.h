/****************************************************************************
 * function_data.h
 *
 *function_data objects hold information for each function.  Namely, the 
 *number of calls, number of subroutines, exclusive time, inclusive time,
 *standard deviation, and name.  The header file also declares an empty 
 *constructor, a copy constructor, a deconstructor, and overloaded versions 
 *of the = and += operators.  The first constructor is defined based on 
 *whether or no we can use long numbers or the defaul double numbers for
 *the variables numcalls and numsubrs.
 *
 ***************************************************************************/

#include "tau_platforms.h"

class function_data {
  public :
#ifdef USE_LONG 
    long numcalls;
    long numsubrs;
#else // DEFAULT double 
    double     numcalls;
    double     numsubrs;
#endif // USE_LONG
    double excl;
    double incl;
    double stddeviation;
    char *groupNames;
#ifdef USE_LONG 
  function_data(long nc, long ns, double ex, double in, double sigma) 
    : numcalls(nc), numsubrs(ns),  excl(ex), incl(in), stddeviation(sigma) { }
#else // DEFAULT double
  function_data(double nc, double ns, double ex, double in, double sigma) 
    : numcalls(nc), numsubrs(ns),  excl(ex), incl(in), stddeviation(sigma) { }
#endif // USE_LONG 
  
//empty constructor
  function_data();       

  //copy constructor;
  function_data(const function_data& X);

  //overloaded = operator
  function_data& operator= (const function_data& X);

  //overloaded += operator
  function_data& operator+= (const function_data& X);

  //deconstructor
  ~function_data();
};

/***************************************************************************
 * $RCSfile: function_data.h,v $   $Author: ntrebon $
 * $Revision: 1.2 $   $Date: 2002/07/25 20:50:00 $
 * TAU_VERSION_ID: $Id: function_data.h,v 1.2 2002/07/25 20:50:00 ntrebon Exp $
 ***************************************************************************/

