#ifndef _pprof_elem_h
#define _pprof_elem_h

#include "tau_platforms.h"
#include <string>
using std::string;

class pprof_elem{
public:
  pprof_elem();
  void setName(string s);
  string getName();
  void setNumCalls(double d);
  double getNumCalls();
  void setNumSubrs(double d);
  double getNumSubrs();
  void setPercent(double d);
  double getPercent();
  void setUsec(double d);
  double getUsec();
  void setCumusec(double d);
  double getCumusec();
  void setCount(double d);
  double getCount();
  void setTotalCount(double d);
  double getTotalCount();
  void setStdDeviation(double d);
  double getStdDeviation();
  void setUsecsPerCall(double d);
  double getUsecsPerCall();
  void setCountsPerCall(double d);
  double getCountsPerCall();
  void printElem();

protected:
  string name;
  double numcalls;
  double numsubrs;
  double percent;
  double usec;
  double cumusec;
  double count;
  double totalcount;
  double stddeviation;
  double usecspercall;
  double countspercall;
};

#endif

/***************************************************************************
 * $RCSfile: pprof_elem.h,v $   $Author: ntrebon $
 * $Revision: 1.2 $   $Date: 2002/07/25 20:51:00 $
 * TAU_VERSION_ID: $Id: pprof_elem.h,v 1.2 2002/07/25 20:51:00 ntrebon Exp $
 ***************************************************************************/

