/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: UserEvent.h					  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**                        -DSGI_TIMERS  for SGI fast nanosecs timer       **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#define TAU_EVENT_DATATYPE  double

class TauUserEvent {
    
  public: 
    TAU_EVENT_DATATYPE MinValue, MaxValue, SumValue, SumSqrValue; 
    TAU_EVENT_DATATYPE LastValueRecorded, UserFunctionValue;
    bool DisableMin, DisableMax, DisableMean, DisableStdDev;
    long NumEvents;
    string EventName;

    void AddEventToDB();
    TauUserEvent();
    TauUserEvent(const char * EName);
    TauUserEvent(TauUserEvent& );
    TauUserEvent& operator= (const TauUserEvent& );
    void TriggerEvent(TAU_EVENT_DATATYPE data);
    ~TauUserEvent();
    TAU_EVENT_DATATYPE GetMin(void);
    TAU_EVENT_DATATYPE GetMax(void);
    TAU_EVENT_DATATYPE GetMean(void);
    double  GetSumSqr(void);
    long    GetNumEvents(void);
    const char *  GetEventName (void) const;
    bool GetDisableMin(void);
    bool GetDisableMax(void);
    bool GetDisableMean(void);
    bool GetDisableStdDev(void);
    void SetDisableMin(bool value);
    void SetDisableMax(bool value);
    void SetDisableMean(bool value);
    void SetDisableStdDev(bool value);
    static void StoreData(void); 
};


vector<TauUserEvent*>& TheEventDB(int threadid = RtsLayer::myThread()); 
/*    
#ifdef PROFILING_ON
#define TAU_REGISTER_EVENT(event, name)  	TauUserEvent event(name);
#define TAU_EVENT(event, data) 		 	(event).TriggerEvent(data);
#define TAU_EVENT_DISABLE_MIN(event) 		(event).SetDisableMin(true);
#define TAU_EVENT_DISABLE_MAX(event) 		(event).SetDisableMax(true);
#define TAU_EVENT_DISABLE_MEAN(event) 		(event).SetDisableMean(true);
#define TAU_EVENT_DISABLE_STDDEV(event) 	(event).SetDisableStdDev(true);
#define TAU_STORE_ALL_EVENTS 			TauUserEvent::StoreData();

#else // PROFILING is disabled
#define TAU_REGISTER_EVENT(event, name)
#define TAU_EVENT(event, data)
#define TAU_EVENT_DISABLE_MIN(event)
#define TAU_EVENT_DISABLE_MAX(event)
#define TAU_EVENT_DISABLE_MEAN(event)
#define TAU_EVENT_DISABLE_STDDEV(event)
#define TAU_STORE_ALL_EVENTS

#endif // PROFILING_ON 
*/

/***************************************************************************
 * $RCSfile: UserEvent.h,v $   $Author: sameer $
 * $Revision: 1.2 $   $Date: 1998/05/14 22:07:59 $
 * POOMA_VERSION_ID: $Id: UserEvent.h,v 1.2 1998/05/14 22:07:59 sameer Exp $ 
 ***************************************************************************/
