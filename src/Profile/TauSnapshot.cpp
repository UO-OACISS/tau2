/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauSnapshot.cpp  				   **
**	Description 	: TAU Profiling Package				   **
**	Author		: Alan Morris					   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : This file contains all the Snapshot routines     **
**                                                                         **
****************************************************************************/

#include <Profiler.h>
#include <TauUtil.h>
#include <TauSnapshot.h>
#include <TauMetaData.h>
#include <TauMetrics.h>
#include <TauXML.h>
#include <TauUnify.h>
#include <vector>

using namespace std;
using namespace tau;

static int Tau_snapshot_writeSnapshot(const char *name, int to_buffer);
static int startNewSnapshotFile(char *threadid, int tid, int to_buffer);
struct SnapshotFileList : vector<Tau_util_outputDevice *>{
    SnapshotFileList (const SnapshotFileList&) = delete;
    SnapshotFileList& operator= (const SnapshotFileList&) = delete;
    SnapshotFileList(){
         //printf("Creating SnapshotFileList at %p\n", this);
      }
     virtual ~SnapshotFileList(){
         //printf("Destroying SnapshotFileList at %p, with size %ld\n", this, this->size());
         Tau_destructor_trigger();
     }
   };

// Static holder for snapshot file handles
static SnapshotFileList & Tau_snapshot_getFiles() {
  static SnapshotFileList snapshotFiles;
  return snapshotFiles;
}

static inline void checkSnapshotFilesVector(int tid){
	while(Tau_snapshot_getFiles().size()<=tid){
        RtsLayer::LockDB();
		Tau_snapshot_getFiles().push_back(NULL);
        RtsLayer::UnLockDB();
	}
}

static inline Tau_util_outputDevice * Tau_snapshot_GetFile(int tid){
    checkSnapshotFilesVector(tid);
	return Tau_snapshot_getFiles()[tid];
}
static inline void Tau_snapshot_SetFile(int tid, Tau_util_outputDevice * value){
    checkSnapshotFilesVector(tid);
	Tau_snapshot_getFiles()[tid]=value;
}


static void writeEventXML(Tau_util_outputDevice *out, int id, FunctionInfo *fi) {
  Tau_util_output (out, "<event id=\"%d\"><name>", id);
  Tau_XML_writeString(out, fi->GetName());
  Tau_XML_writeString(out, " ");
  Tau_XML_writeString(out, fi->GetType());
  Tau_util_output (out, "</name><group>");
  Tau_XML_writeString(out, fi->GetAllGroups());
  Tau_util_output (out, "</group></event>\n");
  return;
}

static void writeUserEventXML(Tau_util_outputDevice *out, int id, TauUserEvent *ue) {
  Tau_util_output (out, "<userevent id=\"%d\"><name>", id);
  Tau_XML_writeString(out, ue->GetName().c_str());
  Tau_util_output (out, "</name></userevent>\n");
  return;
}





extern "C" int Tau_snapshot_initialization() {
  return 0;
}

extern "C" void Tau_snapshot_getBuffer(char *buf) {
	strcpy(buf, Tau_snapshot_GetFile(0)->buffer);
	for (int tid = 1; tid<RtsLayer::getTotalThreads(); tid++) {
		strcat(buf, Tau_snapshot_GetFile(tid)->buffer);
	}
}

extern "C" int Tau_snapshot_getBufferLength() {
	int length = 0;
	for (int tid = 0; tid<RtsLayer::getTotalThreads(); tid++) {
		length +=	Tau_snapshot_GetFile(tid)->bufidx; 
	}
  return length;
}

// Static holder for snapshot event counts
struct snapshotEventCountList: vector<int>{
    snapshotEventCountList(){
        //printf("Creating snapshotEventCountList at %p\n", this);
    }
    virtual ~snapshotEventCountList(){
        //printf("Destroying snapshotEventCountList at %p, with size %ld\n", this, this->size());
        Tau_destructor_trigger();
    }
};
static snapshotEventCountList & Tau_snapshot_getEventCounts() {
  static snapshotEventCountList eventCounts;
  return eventCounts;
}

static inline void checkEventCountsVector(int tid){
	while(Tau_snapshot_getEventCounts().size()<=tid){
        RtsLayer::LockDB();
		Tau_snapshot_getEventCounts().push_back(0);
        RtsLayer::UnLockDB();
	}
}

static inline int Tau_snapshot_getEventCount(int tid){
    checkEventCountsVector(tid);
	return Tau_snapshot_getEventCounts()[tid];
}
static inline void Tau_snapshot_setEventCount(int tid, int value){
    checkEventCountsVector(tid);
	Tau_snapshot_getEventCounts()[tid]=value;
}

// Static holder for snapshot user event counts
struct snapshotUserEventCountList: vector<int>{
    snapshotUserEventCountList(){
        //printf("Creating snapshotUserEventCountList at %p\n", this);
    }
    virtual ~snapshotUserEventCountList(){
        //printf("Destroying snapshotUserEventCountList at %p, with size %ld\n", this, this->size());
        Tau_destructor_trigger();
    }
};
static snapshotUserEventCountList & Tau_snapshot_getUserEventCounts() {
  static snapshotUserEventCountList userEventCounts;
  return userEventCounts;
}
static inline void checkUserEventCountsVector(int tid){
	while(Tau_snapshot_getUserEventCounts().size()<=tid){
        RtsLayer::LockDB();
		Tau_snapshot_getUserEventCounts().push_back(0);
        RtsLayer::UnLockDB();
	}
}
static inline int Tau_snapshot_getUserEventCount(int tid){
    checkUserEventCountsVector(tid);
	return Tau_snapshot_getUserEventCounts()[tid];
}
static inline void Tau_snapshot_setUserEventCount(int tid, int value){
    checkUserEventCountsVector(tid);
	Tau_snapshot_getUserEventCounts()[tid]=value;
}

extern "C" int Tau_snapshot_writeToBuffer(const char *name) {
  Tau_snapshot_writeSnapshot(name, 1);
  return 0;
}


extern "C" int Tau_snapshot_writeIntermediate(const char *name) {
  TAU_PROFILE_TIMER(timer, "TAU_PROFILE_SNAPSHOT()", " ", TAU_IO);
  TAU_PROFILE_START(timer);
  
  Tau_snapshot_writeSnapshot(name, 0);
  
  TAU_PROFILE_STOP(timer);
  return 0;
}


extern "C" int Tau_snapshot_writeMetaDataBlock() {
  int tid = RtsLayer::myThread();
  int totalThreads = RtsLayer::getTotalThreads();
  //Tau_util_outputDevice *out = Tau_snapshot_getFiles()[tid];
  Tau_util_outputDevice *out = Tau_snapshot_GetFile(0);
  char threadid[4096];
  snprintf(threadid, sizeof(threadid),  "%d.%d.%d.%d", RtsLayer::myNode(), RtsLayer::myContext(), tid, RtsLayer::getPid());

  TAU_VERBOSE("tid=%d, totalThreads=%d\n", tid, totalThreads);

  // start of a profile block
  Tau_util_output (out, "<profile_xml>\n");
  
  // thread identifier
  Tau_util_output (out, "\n<thread id=\"%s\" node=\"%d\" context=\"%d\" thread=\"%d\">\n", 
	   threadid, RtsLayer::myNode(), RtsLayer::myContext(), tid);
  Tau_metadata_writeMetaData(out, tid);
  Tau_util_output (out, "</thread>\n");

  // end of profile block
  Tau_util_output (out, "</profile_xml>\n");
  return 0;
}



static int Tau_snapshot_writeSnapshot(const char *name, int to_buffer) {
  int tid = RtsLayer::myThread();
  int i, c;
  Tau_util_outputDevice *out = Tau_snapshot_GetFile(tid);
  
  char threadid[4096];
  snprintf(threadid, sizeof(threadid),  "%d.%d.%d.%d", RtsLayer::myNode(), RtsLayer::myContext(), tid, RtsLayer::getPid());
  
  RtsLayer::LockDB();
  int numFunc = TheFunctionDB().size();
  int numEvents = TheEventDB().size();

   if (!out) {
     startNewSnapshotFile(threadid, tid, to_buffer);
     out = Tau_snapshot_GetFile(tid);
   } else {
     Tau_util_output (out, "<profile_xml>\n");
   }
	 
   if (TauEnv_get_summary_only()) { /* skip writing event definitions */
	 	 return 0;
	 }
   
   // write out new events since the last snapshot
   if (Tau_snapshot_getEventCount(tid) != numFunc) {
     Tau_util_output (out, "\n<definitions thread=\"%s\">\n", threadid);
     for (int i=Tau_snapshot_getEventCount(tid); i < numFunc; i++) {
       FunctionInfo *fi = TheFunctionDB()[i];
       writeEventXML(out, i, fi);
     }
     Tau_util_output (out, "</definitions>\n");
     Tau_snapshot_setEventCount(tid, numFunc);
   }

   // write out new user events since the last snapshot
   if ( Tau_snapshot_getUserEventCount(tid) != numEvents) {
     Tau_util_output (out, "\n<definitions thread=\"%s\">\n", threadid);
     for (int i= Tau_snapshot_getUserEventCount(tid); i < numEvents; i++) {
       TauUserEvent *ue = TheEventDB()[i];
       writeUserEventXML(out, i, ue);
     }
     Tau_util_output (out, "</definitions>\n");
     Tau_snapshot_setUserEventCount(tid, numEvents);
   }

   // now write the actual profile data for this snapshot
   Tau_util_output (out, "\n<profile thread=\"%s\">\n", threadid);
   Tau_util_output (out, "<name>");
   Tau_XML_writeString(out, name);
   Tau_util_output (out, "</name>\n");

#ifdef TAU_WINDOWS
   Tau_util_output (out, "<timestamp>%I64d</timestamp>\n", TauMetrics_getTimeOfDay());
#else
   Tau_util_output (out, "<timestamp>%lld</timestamp>\n", TauMetrics_getTimeOfDay());
#endif

   char metricList[4096];
   char *loc = metricList;
   for (c=0; c<Tau_Global_numCounters; c++) {
       loc += sprintf (loc,"%d ", c);
   }
   Tau_util_output (out, "<interval_data metrics=\"%s\">\n", metricList);

   TauProfiler_updateIntermediateStatistics(tid);

   for (i=0; i < numFunc; i++) {
     FunctionInfo *fi = TheFunctionDB()[i];

     if (fi->GetCalls(tid) > 0) {
       // get currently stored values
       double *incltime = fi->getDumpInclusiveValues(tid);
       double *excltime = fi->getDumpExclusiveValues(tid);
       Tau_util_output (out, "%d %ld %ld ", i, fi->GetCalls(tid), fi->GetSubrs(tid));
       for (c=0; c<Tau_Global_numCounters; c++) {
         Tau_util_output (out, "%.16G %.16G ", excltime[c], incltime[c]);
       }
       Tau_util_output (out, "\n");
	 } else {
     }
   }
   Tau_util_output (out, "</interval_data>\n");


   // now write the user events
   Tau_util_output (out, "<atomic_data>\n");
   for (i=0; i < numEvents; i++) {
     TauUserEvent *ue = TheEventDB()[i];
     if (ue->GetNumEvents(tid) > 0) {
           Tau_util_output (out, "%d %ld %.16G %.16G %.16G %.16G\n", 
	     i, ue->GetNumEvents(tid), ue->GetMax(tid),
	     ue->GetMin(tid), ue->GetMean(tid), ue->GetSumSqr(tid));
     }
   }
   Tau_util_output (out, "</atomic_data>\n");

   Tau_util_output (out, "</profile>\n");
   Tau_util_output (out, "\n</profile_xml>\n");


   RtsLayer::UnLockDB();
   
   return 0;
}

#ifdef TAU_UNIFY
int Tau_snapshot_writeUnifiedBuffer(int tid) {
  //int tid = RtsLayer::myThread();
  int c;
  Tau_util_outputDevice *out = Tau_snapshot_GetFile(tid);
  
  char threadid[4096];
  snprintf(threadid, sizeof(threadid),  "%d.%d.%d.%d", RtsLayer::myNode(), RtsLayer::myContext(), tid, RtsLayer::getPid());
  
  RtsLayer::LockDB();

   if (!out) {
     int to_buffer=1;
     startNewSnapshotFile(threadid, tid, to_buffer);
     out = Tau_snapshot_GetFile(tid);
   } else {
     Tau_util_output (out, "<profile_xml>\n");
   }
   
   Tau_unify_object_t *functionUnifier, *atomicUnifier;
   functionUnifier = Tau_unify_getFunctionUnifier();
   atomicUnifier = Tau_unify_getAtomicUnifier();

   // create a reverse mapping, not strictly necessary, but it makes things easier
   int *globalmap = (int*)TAU_UTIL_MALLOC(functionUnifier->globalNumItems * sizeof(int));
   for (int i=0; i<functionUnifier->globalNumItems; i++) { // initialize all to -1
     globalmap[i] = -1; // -1 indicates that the event did not occur for this rank
   }
   for (int i=0; i<functionUnifier->localNumItems; i++) {
     globalmap[functionUnifier->mapping[i]] = i; // set reverse mapping
   }

   TauProfiler_updateIntermediateStatistics(tid);

   if (TauEnv_get_summary_only()) { /* skip event unification. */
     return 0;
   }

   // now write the actual profile data for this snapshot
   Tau_util_output (out, "\n<profile thread=\"%s\">\n", threadid);

#ifdef TAU_WINDOWS
   Tau_util_output (out, "<timestamp>%I64d</timestamp>\n", TauMetrics_getTimeOfDay());
#else
   Tau_util_output (out, "<timestamp>%lld</timestamp>\n", TauMetrics_getTimeOfDay());
#endif

   char metricList[4096];
   char *loc = metricList;
   for (c=0; c<Tau_Global_numCounters; c++) {
       loc += sprintf (loc,"%d ", c);
   }
   Tau_util_output (out, "<interval_data metrics=\"%s\">\n", metricList);

  // the global number of events
  int numItems = functionUnifier->globalNumItems;

  for (int e=0; e<numItems; e++) { // for each event
    if (globalmap[e] != -1) { // if it occurred in our rank
      
      int local_index = functionUnifier->sortMap[globalmap[e]];
      FunctionInfo *fi = TheFunctionDB()[local_index];
      if (fi->GetCalls(tid) > 0) {
      
      // get currently stored values
			double *incltime, *excltime;
			if (tid == 0)
			{
				incltime = fi->getDumpInclusiveValues(tid);
				excltime = fi->getDumpExclusiveValues(tid);
     	}
			else
			{
				incltime = fi->GetInclTime(tid);
				excltime = fi->GetExclTime(tid);
			}
      Tau_util_output (out, "%d %ld %ld ", e, fi->GetCalls(tid), fi->GetSubrs(tid));
      for (c=0; c<Tau_Global_numCounters; c++) {
	Tau_util_output (out, "%.16G %.16G ", excltime[c], incltime[c]);
      }
      Tau_util_output (out, "\n");
	  }
    }
  }
  
  Tau_util_output (out, "</interval_data>\n");

  free (globalmap);
  // create a reverse mapping, not strictly necessary, but it makes things easier
  globalmap = (int*)TAU_UTIL_MALLOC(atomicUnifier->globalNumItems * sizeof(int));
  for (int i=0; i<atomicUnifier->globalNumItems; i++) { // initialize all to -1
    globalmap[i] = -1; // -1 indicates that the event did not occur for this rank
  }
  for (int i=0; i<atomicUnifier->localNumItems; i++) {
    globalmap[atomicUnifier->mapping[i]] = i; // set reverse mapping
  }
  numItems = atomicUnifier->globalNumItems;


   // now write the user events
   Tau_util_output (out, "<atomic_data>\n");
   for (int e=0; e<numItems; e++) { // for each event
     if (globalmap[e] != -1) { // if it occurred in our rank
       int local_index = atomicUnifier->sortMap[globalmap[e]];
       TauUserEvent *ue = TheEventDB()[local_index];
           Tau_util_output (out, "%d %ld %.16G %.16G %.16G %.16G\n", 
			    e, ue->GetNumEvents(tid), ue->GetMax(tid),
			    ue->GetMin(tid), ue->GetMean(tid), ue->GetSumSqr(tid));
     }
   }
   free(globalmap);
   Tau_util_output (out, "</atomic_data>\n");

   Tau_util_output (out, "</profile>\n");
   Tau_util_output (out, "\n</profile_xml>\n");


   RtsLayer::UnLockDB();
   
   return 0;
}
#endif /* TAU_UNIFY */


static int startNewSnapshotFile(char *threadid, int tid, int to_buffer) {
  const char *profiledir = TauEnv_get_profiledir();
  
  Tau_util_outputDevice *out = (Tau_util_outputDevice*) malloc (sizeof(Tau_util_outputDevice));

  if (to_buffer == 1) {
    out->type = TAU_UTIL_OUTPUT_BUFFER;
    out->bufidx = 0;
    out->buflen = TAU_UTIL_INITIAL_BUFFER;
    out->buffer = (char *) malloc (out->buflen);
  } else {

    char filename[4096];
    snprintf (filename, sizeof(filename), "%s/snapshot.%d.%d.%d", profiledir, 
	     RtsLayer::myNode(), RtsLayer::myContext(), tid);
    FILE *fp;

    char cwd[1024];
    char *tst = getcwd(cwd, 1024);
	if (tst == NULL) {
      char errormsg[4096];
      snprintf(errormsg, sizeof(errormsg), "Error: Could not get current working directory");
      perror(errormsg);
      RtsLayer::UnLockDB();
      return 0;
	}
    TAU_VERBOSE("TAU: Opening Snapshot File %s, cwd = %s\n", filename, cwd);

    if ((fp = fopen (filename, "w+")) == NULL) {
      char errormsg[4196];
      snprintf(errormsg, sizeof(errormsg), "Error: Could not create %s",filename);
      perror(errormsg);
      RtsLayer::UnLockDB();
      return 0;
    }
    out->type = TAU_UTIL_OUTPUT_FILE;
    out->fp = fp;
  }
    
  // assign it back to the global structure for this thread
  Tau_snapshot_SetFile(tid, out);
	
  if (TauEnv_get_summary_only()) { /* skip thread id for summary */
		return 0;
	}

  // start of a profile block
  Tau_util_output (out, "<profile_xml>\n");
  
  // thread identifier
  Tau_util_output (out, "\n<thread id=\"%s\" node=\"%d\" context=\"%d\" thread=\"%d\">\n", 
	   threadid, RtsLayer::myNode(), RtsLayer::myContext(), tid);
  Tau_metadata_writeMetaData(out, tid);
  Tau_util_output (out, "</thread>\n");
  
  // definition block
  Tau_util_output (out, "\n<definitions thread=\"%s\">\n", threadid);
  
  for (int i=0; i<Tau_Global_numCounters; i++) {
      const char *tmpChar = RtsLayer::getCounterName(i);
      Tau_util_output (out, "<metric id=\"%d\">", i);
      Tau_XML_writeTag(out, "name", tmpChar, true);
      Tau_XML_writeTag(out, "units", "unknown", true);
      Tau_util_output (out, "</metric>\n");
  }

  // set the counts to zero
  Tau_snapshot_setEventCount(tid, 0);
  Tau_snapshot_setUserEventCount(tid, 0);

  Tau_util_output (out, "</definitions>\n");
  return 0;
}



extern "C" int Tau_snapshot_writeFinal(const char *name) {
  int tid = RtsLayer::myThread();
  Tau_util_outputDevice *out = Tau_snapshot_GetFile(tid);
  int haveWrittenSnapshot = 0;
 
  if (out != NULL) { 
    // if the output device is not null, then we must have written at least one snapshot, 
    // so we should write a "final" snapshot as well.
    haveWrittenSnapshot = 1;
  }
  
  if (haveWrittenSnapshot || (TauEnv_get_profile_format() == TAU_FORMAT_SNAPSHOT)) { 
    Tau_snapshot_writeSnapshot(name, 0);
    out = Tau_snapshot_GetFile(tid);
    if (out->type == TAU_UTIL_OUTPUT_FILE) {
      fclose(out->fp);
    }
  }
  return 0;
}
