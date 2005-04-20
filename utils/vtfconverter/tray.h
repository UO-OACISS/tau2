/************************************************************************
************************************************************************/

/************************************************************************
	tray.h
	Author: Ryan K. Harris
 
	Definitions for the tray class.
	A tray contains the applicable information for each instance of
	a function to be stored on the call_trace stack.
*************************************************************************
	class tray
 
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
		getLastEntrance -- used to track last time this tray was entered.
		setLastEntrance -- used to set lastEntrance variable.
		getLastSample -- return last observed sample data
		setLastSample -- set last observed sample data
************************************************************************/
#ifndef __TRAY_H
#define __TRAY_H
#include <iostream>
using namespace std;
class tray
{
private:
	tray * previousTray;
	tray * nextTray;

	int functionID;
	double timeStamp;
	unsigned long long * samples;
	unsigned long long * lastsamples;
	int samplesdim;
	int exchangeType;

	double lastEntrance;

public:
	tray(int n_functionID, double n_timeStamp, unsigned long long * a_samples, int n_samplesdim, int n_exchangeType)
	{
		previousTray = 0;
		nextTray = 0;
		functionID = n_functionID;
		timeStamp = n_timeStamp;
		exchangeType = n_exchangeType;
		samplesdim = n_samplesdim;
		
		/*if(samplesdim == 0)
		{
			samples = 0;
			lastsamples = 0;
		}
		else
		{*/
			samples = new unsigned long long[samplesdim];
			lastsamples = new unsigned long long[samplesdim];
		//}
		

		lastEntrance = 0;

		
		for(int i = 0; i < samplesdim; i++)
		{
			samples[i] = a_samples[i];

			lastsamples[i] = 0;
		}
	}

	/* Constructor for the type of tray(s) needed by
		the obsolstack class. */
	tray(int n_functionID, double n_timeStamp, unsigned long long * a_samples, int n_samplesdim)
	{
		previousTray = 0;
		nextTray = 0;
		functionID = n_functionID;
		timeStamp = n_timeStamp;
		exchangeType = 0;
		samplesdim = n_samplesdim;
		samples = a_samples;

		lastEntrance = 0;

		lastsamples = new unsigned long long[samplesdim];
		for(int i = 0; i < samplesdim; i++)
		{
			lastsamples[i] = 0;
		}

	}

	~tray()
	{   //if(*samples != 0)
		delete[] samples;
		//cout << samples << " " << *samples << " " << samples[0] << endl;
		//if(*lastsamples != 0)
		delete[] lastsamples;
	}

	int getFunctionID(void);
	double getTimeStamp(void);
	int getExchangeType(void);

	void setFunctionID(int n_functionID);
	void setTimeStamp(double n_timeStamp);
	void setSamples(unsigned long long * a_samples);
	void setExchangeType(int n_exchangeType);

	tray * getPrevious(void);
	tray * getNext(void);

	void setPrevious(tray * n_previousTray);
	void setNext(tray * n_nextTray);

	double getLastEntrance(void);

	unsigned long long * getLastSamples(void);
	unsigned long long * getSamples(void);

	void setLastEntrance(double n_lastEntrance);

	void setLastSamples(unsigned long long * a_lastsamples);
}
;//tray

#endif
