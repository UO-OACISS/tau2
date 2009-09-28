/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009					   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: UserEvent.h					  **
**	Description 	: TAU Profiling Package				  **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#define TAU_EVENT_DATATYPE  double

class TauContextUserEvent;
class TauUserEvent {
    
  public: 
    TAU_STORAGE(TAU_EVENT_DATATYPE, MinValue);
    TAU_STORAGE(TAU_EVENT_DATATYPE, MaxValue);
    TAU_STORAGE(TAU_EVENT_DATATYPE, SumValue);
    TAU_STORAGE(TAU_EVENT_DATATYPE, SumSqrValue); 
    TAU_STORAGE(TAU_EVENT_DATATYPE, LastValueRecorded);
    TAU_STORAGE(TAU_EVENT_DATATYPE, UserFunctionValue);
    TAU_STORAGE(long, NumEvents);
    bool DisableMin, DisableMax, DisableMean, DisableStdDev, MonotonicallyIncreasing;
    string EventName;
    long EventId;
    TauContextUserEvent *ctxevt;

    void AddEventToDB();
    TauUserEvent();
    TauUserEvent(const char * EName, bool MonotonicallyIncreasing=false);
    TauUserEvent(TauUserEvent& );
    TauUserEvent& operator= (const TauUserEvent& );
    void TriggerEvent(TAU_EVENT_DATATYPE data, int tid = RtsLayer::myThread());
    void TriggerEvent(TAU_EVENT_DATATYPE data, int tid, double timestamp, int use_ts);
    ~TauUserEvent();
    TAU_EVENT_DATATYPE GetMin(int tid = RtsLayer::myThread());
    TAU_EVENT_DATATYPE GetMax(int tid = RtsLayer::myThread());
    TAU_EVENT_DATATYPE GetSumValue(int tid = RtsLayer::myThread());
    TAU_EVENT_DATATYPE GetMean(int tid = RtsLayer::myThread());
    double  GetSumSqr(int tid = RtsLayer::myThread());
    long    GetNumEvents(int tid = RtsLayer::myThread());
    const char *  GetEventName (void) const;
    long GetEventId(void);
    void SetEventName(const char * newname); 
    void SetEventName(string newname); 
    bool GetDisableMin(void);
    bool GetDisableMax(void);
    bool GetDisableMean(void);
    bool GetDisableStdDev(void);
    void SetDisableMin(bool value);
    void SetDisableMax(bool value);
    void SetDisableMean(bool value);
    void SetDisableStdDev(bool value);
    void SetMonotonicallyIncreasing(bool value);
    bool GetMonotonicallyIncreasing(void);
    static void ReportStatistics(bool ForEachThread = false); 
};

class TauContextUserEvent {
  public:
    TauContextUserEvent(const char * EName, bool MonoIncr=false);
    ~TauContextUserEvent(); 
    void SetDisableContext(bool value);
    void TriggerEvent(TAU_EVENT_DATATYPE data, int tid = RtsLayer::myThread());
    void TriggerEvent(TAU_EVENT_DATATYPE data, int tid, double timestamp, int use_ts);
    const char *GetEventName(void); 
    TauUserEvent *contextevent;
  private:
    bool DisableContext;
    TauUserEvent *uevent;
    bool MonotonicallyIncreasing;
};


TAU_STD_NAMESPACE vector<TauUserEvent*>& TheEventDB(void);

/***************************************************************************
 * $RCSfile: UserEvent.h,v $   $Author: amorris $
 * $Revision: 1.17 $   $Date: 2009/09/28 18:28:48 $
 * POOMA_VERSION_ID: $Id: UserEvent.h,v 1.17 2009/09/28 18:28:48 amorris Exp $ 
 ***************************************************************************/
