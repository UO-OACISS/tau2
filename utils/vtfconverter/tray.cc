/************************************************************************
************************************************************************/

/************************************************************************
	tray.cc
	Author: Ryan K. Harris
 
	Source file for tray class.
*************************************************************************
 
	Member functions:
		getFunctionID -- returns functionID
		getTimeStamp -- returns time stamp
		getSamples -- returns samples
		getExchangeType -- returns exchange type
		setFunctionID -- sets functionID
		setTimeStamp -- sets timeStamp
		setSamples -- sets the sample data (samples) array
		setExchangeType -- sets exchangeType
 
		getPrevious -- gets tray that came previous to the current one
		getNext -- gets tray that comes after this one
		setPrevious -- set previousTray pointer
		setNext -- set nextTray pointer
		getLastEntrance -- return the 'lastEntrance' var.
		setLastEntrance -- set time of 'lastEntrance' var.
		getLastSample -- return last observed sample data
		setLastSample -- set last observed sample data
************************************************************************/

#include "tray.h"


/************************************************************************
	tray::getFunctionID
		Returns the functionID
 
	Parameters: none
************************************************************************/
int tray::getFunctionID(void)
{
	return(functionID);
}//getFunctionID

/************************************************************************
	tray::getTimeStamp
		Returns the timeStamp
 
	Parameters: none
************************************************************************/
double tray::getTimeStamp(void)
{
	return(timeStamp);
}//getTimeStamp

/************************************************************************
	tray::getExchangeType
		Returns exchangeType
 
	Parameters: none
************************************************************************/
int tray::getExchangeType(void)
{
	return(exchangeType);
}//getExchangeType

/************************************************************************
	tray::setFunctionID
		set the functionID
 
	Parameters: int (pass by value)
************************************************************************/
void tray::setFunctionID(int n_functionID)
{
	functionID = n_functionID;
	return;
}//setFunctionID

/************************************************************************
	tray::setTimeStamp
		set the timeStamp
 
	Parameters: int (pass by value)
************************************************************************/
void tray::setTimeStamp(double n_timeStamp)
{
	timeStamp = n_timeStamp;
	return;
}//setTimeStamp

/************************************************************************
	tray::setExchangeType
		set the exchangeType
 
	Parameters: int (pass by value)
************************************************************************/
void tray::setExchangeType(int n_exchangeType)
{
	exchangeType = n_exchangeType;
	return;
}//setExchangeType

/************************************************************************
	tray::getPrevious
		get the tray that came previous to this one on the stack
 
	Parameters: none
************************************************************************/
tray * tray::getPrevious(void)
{
	return(previousTray);
}//getPrevious

/************************************************************************
	tray::getNext
		returns the tray that comes after this one on the stack
 
	Parameters: none
************************************************************************/
// tray * tray::getNext(void)
// {
// 	return(nextTray);
// }//getNext

/************************************************************************
	tray::setPrevious
		set the previousTray pointer
 
	Parameters: reference to a tray (pass by address)
************************************************************************/
void tray::setPrevious(tray * n_previousTray)
{
  //	delete previousTray;
	previousTray = n_previousTray;
	return;
}//setPrevious

/************************************************************************
	tray::setNext
		set the nextTray pointer
 
	Parameters: reference to a tray pointer ( pass by address)
************************************************************************/
// void tray::setNext(tray * n_nextTray)
// {
//   //	delete nextTray;
// 	nextTray = n_nextTray;
// 	return;
// }//setNext

/************************************************************************
	tray::getLastEntrance
		returns time this function was last entered/returned to, or
		began generating exclusive time again
 
	Parameters: none
************************************************************************/
double tray::getLastEntrance(void)
{
	return(lastEntrance);
}//getLastEntrance

/************************************************************************
	tray::setLastEntrance
		set last time this function was entered/returned to, or
		began generating exclusive time
 
	Parameters: double: time to set (pass by value)
************************************************************************/
void tray::setLastEntrance(double n_lastEntrance)
{
	lastEntrance = n_lastEntrance;
	return;
}//setLastEntrance


/************************************************************************
	tray::setLastSamples
		set last values of this  function's samples
 
	Parameters: int*: values to set (pass by value)
************************************************************************/
void tray::setLastSamples(unsigned long long * a_lastsamples)
{
	for(int i = 0; i<samplesdim; i++)
	{
		lastsamples[i] = a_lastsamples[i];
	}
	return;
}//setLastSamples

/************************************************************************
	tray::getLastSamples
		get last values of this function's samples
 
	Parameters: int*: values to get (pass by value)
************************************************************************/
unsigned long long * tray::getLastSamples(void)
{
	return lastsamples;
}//getLastSamples


/************************************************************************
	tray::getSamples
		get last values of this function's samples
 
	Parameters: int*: values to get (pass by value)
************************************************************************/
unsigned long long * tray::getSamples(void)
{
	return samples;
}//getSamples


/************************************************************************
	tray::setSamples
		set values of this  function's samples
 
	Parameters: int*: values to set (pass by value)
************************************************************************/
void tray::setSamples(unsigned long long * a_samples)
{
	for(int i = 0; i<samplesdim; i++)
	{
		samples[i] = a_samples[i];
	}
	return;
}//setLastSample
