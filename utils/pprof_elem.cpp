#include "pprof_elem.h"

pprof_elem::pprof_elem(){
  groupnames="";
}//constructor

void pprof_elem::setName(string s){
  name=s;
}//setName()

string pprof_elem::getName(){
  return name;
}//getName()

void pprof_elem::setNumCalls(double d){
  numcalls=d;
}//setNumCalls()

double pprof_elem::getNumCalls(){
  return numcalls;
}//getNumCalls()

void pprof_elem::setNumSubrs(double d){
  numsubrs=d;
}//setNumSubrs()

double pprof_elem::getNumSubrs(){
  return numsubrs;
}//getNumSubrs()

void pprof_elem::setPercent(double d){
  percent=d;
}//setPercent();

double pprof_elem::getPercent(){
  return percent;
}//getPercent()

void pprof_elem::setUsec(double d){
  usec=d;
}//setUsec()

double pprof_elem::getUsec(){
  return usec;
}//getUsec()

void pprof_elem::setCumusec(double d){
  cumusec=d;
}//setCumusec()

double pprof_elem::getCumusec(){
  return cumusec;
}//getCumusec()

void pprof_elem::setCount(double d){
  count=d;
}//setCount()

double pprof_elem::getCount(){
  return count;
}//getCount()

void pprof_elem::setTotalCount(double d){
  totalcount=d;
}//setTotalCount()

double pprof_elem::getTotalCount(){
  return totalcount;
}//getTotalCount()

void pprof_elem::setStdDeviation(double d){
  stddeviation=d;
}//setStdDeviation()

double pprof_elem::getStdDeviation(){
  return stddeviation;
}//getStdDeviation()

void pprof_elem::setUsecsPerCall(double d){
  usecspercall=d;
}//setUsecsPerCall()

double pprof_elem::getUsecsPerCall(){
  return usecspercall;
}//getUsecsPerCall()

void pprof_elem::setCountsPerCall(double d){
  countspercall=d;
}//setCountsPerCall()

double pprof_elem::getCountsPerCall(){
  return countspercall;
}//getCountsPerCall()

void pprof_elem::setGroupNames(string s){
  groupnames=s;
}//setGroupNames()

string pprof_elem::getGroupNames(){
  return groupnames;
}//getGroupNames()


void pprof_elem::printElem(){
  printf("NAME: %s\n",name.c_str());
  printf("\tNUMCALLS: %lG\tNUMSUBRS: %lG\tPERCENT: %lG\n",numcalls,numsubrs,percent);
  printf("\tUSEC: %lG\tCUMUSEC: %lG\tCOUNT: %lG\n",usec,cumusec,count);
  printf("\tTOTALCOUNT: %lG\tSTDDEV: %lG\n",totalcount, stddeviation);
  printf("\tUSECSPERCALL: %lG\tCOUNTSPERCALL: %lG\n",usecspercall,countspercall);
  printf("\tGROUPNAME(S): %s\n",groupnames.c_str());
}//printElem()


/***************************************************************************
 * $RCSfile: pprof_elem.cpp,v $   $Author: ntrebon $
 * $Revision: 1.2 $   $Date: 2002/08/05 20:19:37 $
 * TAU_VERSION_ID: $Id: pprof_elem.cpp,v 1.2 2002/08/05 20:19:37 ntrebon Exp $
 ***************************************************************************/

