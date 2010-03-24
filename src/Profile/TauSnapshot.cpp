/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2007  						   	   **
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

static int Tau_snapshot_writeSnapshot(const char *name, int to_buffer);
static int startNewSnapshotFile(char *threadid, int tid, int to_buffer);


// Static holder for snapshot file handles
static Tau_util_outputDevice **Tau_snapshot_getFiles() {
  static Tau_util_outputDevice **snapshotFiles = NULL;
  int i;
  if (!snapshotFiles) {
    snapshotFiles = new Tau_util_outputDevice*[TAU_MAX_THREADS];
    for (i=0; i<TAU_MAX_THREADS; i++) {
      snapshotFiles[i] = NULL;
    }
  }
  return snapshotFiles;
}

static void writeEventXML(Tau_util_outputDevice *out, int id, FunctionInfo *fi) {
  Tau_util_output (out, "<event id=\"%d\"><name>", id);
  Tau_XML_writeString(out, fi->GetName());
  Tau_util_output (out, "</name><group>");
  Tau_XML_writeString(out, fi->GetAllGroups());
  Tau_util_output (out, "</group></event>\n");
  return;
}

static void writeUserEventXML(Tau_util_outputDevice *out, int id, TauUserEvent *ue) {
  Tau_util_output (out, "<userevent id=\"%d\"><name>", id);
  Tau_XML_writeString(out, ue->GetEventName());
  Tau_util_output (out, "</name></userevent>\n");
  return;
}





extern "C" int Tau_snapshot_initialization() {
  return 0;
}

extern "C" char *Tau_snapshot_getBuffer() {
  // only support thread 0 right now
  char *buf = Tau_snapshot_getFiles()[0]->buffer;
  return buf;
}

extern "C" int Tau_snapshot_getBufferLength() {
  return Tau_snapshot_getFiles()[0]->bufidx;
}

// Static holder for snapshot event counts
static int *Tau_snapshot_getEventCounts() {
  static int eventCounts[TAU_MAX_THREADS];
  return eventCounts;
}

// Static holder for snapshot user event counts
static int *Tau_snapshot_getUserEventCounts() {
  static int userEventCounts[TAU_MAX_THREADS];
  return userEventCounts;
}


extern "C" int Tau_snapshot_writeToBuffer(const char *name) {
  Tau_snapshot_writeSnapshot(name, 1);
}


extern "C" int Tau_snapshot_writeIntermediate(const char *name) {
  int tid = RtsLayer::myThread();

   TAU_PROFILE_TIMER(timer, "TAU_PROFILE_SNAPSHOT()", " ", TAU_IO);
   TAU_PROFILE_START(timer);
  
   Tau_snapshot_writeSnapshot(name, 0);

   TAU_PROFILE_STOP(timer);
   return 0;
}


extern "C" int Tau_snapshot_writeMetaDataBlock() {
  int tid = RtsLayer::myThread();
  Tau_util_outputDevice *out = Tau_snapshot_getFiles()[tid];
  char threadid[4096];
  sprintf(threadid, "%d.%d.%d.%d", RtsLayer::myNode(), RtsLayer::myContext(), tid, RtsLayer::getPid());

  // start of a profile block
  Tau_util_output (out, "<profile_xml>\n");
  
  // thread identifier
  Tau_util_output (out, "\n<thread id=\"%s\" node=\"%d\" context=\"%d\" thread=\"%d\">\n", 
	   threadid, RtsLayer::myNode(), RtsLayer::myContext(), tid);
  Tau_metadata_writeMetaData(out);
  Tau_util_output (out, "</thread>\n");

  // end of profile block
  Tau_util_output (out, "</profile_xml>\n");
}



static int Tau_snapshot_writeSnapshot(const char *name, int to_buffer) {
  int tid = RtsLayer::myThread();
  int i, c;
  Tau_util_outputDevice *out = Tau_snapshot_getFiles()[tid];
  
  char threadid[4096];
  sprintf(threadid, "%d.%d.%d.%d", RtsLayer::myNode(), RtsLayer::myContext(), tid, RtsLayer::getPid());
  
  RtsLayer::LockDB();
  int numFunc = TheFunctionDB().size();
  int numEvents = TheEventDB().size();

   if (!out) {
     startNewSnapshotFile(threadid, tid, to_buffer);
     out = Tau_snapshot_getFiles()[tid];
   } else {
     Tau_util_output (out, "<profile_xml>\n");
   }
   
   // write out new events since the last snapshot
   if (Tau_snapshot_getEventCounts()[tid] != numFunc) {
     Tau_util_output (out, "\n<definitions thread=\"%s\">\n", threadid);
     for (int i=Tau_snapshot_getEventCounts()[tid]; i < numFunc; i++) {
       FunctionInfo *fi = TheFunctionDB()[i];
       writeEventXML(out, i, fi);
     }
     Tau_util_output (out, "</definitions>\n");
     Tau_snapshot_getEventCounts()[tid] = numFunc;
   }

   // write out new user events since the last snapshot
   if (Tau_snapshot_getUserEventCounts()[tid] != numEvents) {
     Tau_util_output (out, "\n<definitions thread=\"%s\">\n", threadid);
     for (int i=Tau_snapshot_getUserEventCounts()[tid]; i < numEvents; i++) {
       TauUserEvent *ue = TheEventDB()[i];
       writeUserEventXML(out, i, ue);
     }
     Tau_util_output (out, "</definitions>\n");
     Tau_snapshot_getUserEventCounts()[tid] = numEvents;
   }

   // now write the actual profile data for this snapshot
   Tau_util_output (out, "\n<profile thread=\"%s\">\n", threadid);
   Tau_util_output (out, "<name>");
   Tau_XML_writeString(out, name);
   Tau_util_output (out, "</name>\n");

#ifdef TAU_WINDOWS
   Tau_util_output (out, "<timestamp>%I64d</timestamp>\n", TauMetrics_getInitialTimeStamp());
#else
   Tau_util_output (out, "<timestamp>%lld</timestamp>\n", TauMetrics_getInitialTimeStamp());
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

     // get currently stored values
     double *incltime = fi->getDumpInclusiveValues(tid);
     double *excltime = fi->getDumpExclusiveValues(tid);
  
     
     Tau_util_output (out, "%d %ld %ld ", i, fi->GetCalls(tid), fi->GetSubrs(tid));
     for (c=0; c<Tau_Global_numCounters; c++) {
       Tau_util_output (out, "%.16G %.16G ", excltime[c], incltime[c]);
     }
     Tau_util_output (out, "\n");
   }
   Tau_util_output (out, "</interval_data>\n");


   // now write the user events
   Tau_util_output (out, "<atomic_data>\n");
   for (i=0; i < numEvents; i++) {
     TauUserEvent *ue = TheEventDB()[i];
           Tau_util_output (out, "%d %ld %.16G %.16G %.16G %.16G\n", 
	     i, ue->GetNumEvents(tid), ue->GetMax(tid),
	     ue->GetMin(tid), ue->GetMean(tid), ue->GetSumSqr(tid));
   }
   Tau_util_output (out, "</atomic_data>\n");

   Tau_util_output (out, "</profile>\n");
   Tau_util_output (out, "\n</profile_xml>\n");


   RtsLayer::UnLockDB();
   
   return 0;
}

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
    sprintf (filename,"%s/snapshot.%d.%d.%d", profiledir, 
	     RtsLayer::myNode(), RtsLayer::myContext(), tid);
    FILE *fp;

    char cwd[1024];
    char *tst = getcwd(cwd, 1024);
    TAU_VERBOSE("TAU: Opening Snapshot File %s, cwd = %s\n", filename, cwd);

    if ((fp = fopen (filename, "w+")) == NULL) {
      char errormsg[4096];
      sprintf(errormsg,"Error: Could not create %s",filename);
      perror(errormsg);
      RtsLayer::UnLockDB();
      return 0;
    }
    out->type = TAU_UTIL_OUTPUT_FILE;
    out->fp = fp;
  }
    
  // assign it back to the global structure for this thread
  Tau_snapshot_getFiles()[tid] = out;

  // start of a profile block
  Tau_util_output (out, "<profile_xml>\n");
  
  // thread identifier
  Tau_util_output (out, "\n<thread id=\"%s\" node=\"%d\" context=\"%d\" thread=\"%d\">\n", 
	   threadid, RtsLayer::myNode(), RtsLayer::myContext(), tid);
  Tau_metadata_writeMetaData(out);
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
  Tau_snapshot_getEventCounts()[tid] = 0;
  Tau_snapshot_getUserEventCounts()[tid] = 0;

  Tau_util_output (out, "</definitions>\n");
  return 0;
}



extern "C" int Tau_snapshot_writeFinal(const char *name) {
  int tid = RtsLayer::myThread();
  Tau_util_outputDevice *out = Tau_snapshot_getFiles()[tid];
  int haveWrittenSnapshot = 0;
 
  if (out != NULL) { 
    // if the output device is not null, then we must have written at least one snapshot, 
    // so we should write a "final" snapshot as well.
    haveWrittenSnapshot = 1;
  }
  
  if (haveWrittenSnapshot || (TauEnv_get_profile_format() == TAU_FORMAT_SNAPSHOT)) { 
    Tau_snapshot_writeSnapshot(name, 0);
    out = Tau_snapshot_getFiles()[tid];
    if (out->type == TAU_UTIL_OUTPUT_FILE) {
      fclose(out->fp);
    }
  }
}
