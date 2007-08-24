/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2007  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: MetaData.cpp  				   **
**	Description 	: TAU Profiling Package				   **
**	Author		: Alan Morris					   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : This file contains all the Metadata, XML and     **
**                        Snapshot related routines                        **
**                                                                         **
****************************************************************************/

#ifndef TAU_DISABLE_METADATA
#include "tau_config.h"
#if defined(TAU_WINDOWS)
double TauWindowsUsecD(); // from RtsLayer.cpp
#else
#include <sys/utsname.h> // for host identification (uname)
#include <unistd.h>
#include <sys/time.h>
#endif /* TAU_WINDOWS */
#endif /* TAU_DISABLE_METADATA */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "Profile/Profiler.h"
#include "tauarch.h"
#include "Profile/tau_types.h"

#ifdef TAU_BGL
#include <rts.h>
#include <bglpersonality.h>
#endif

#include <signal.h>

char * TauGetCounterString(void);

void tauSignalHandler(int sig) {
  fprintf (stderr, "Caught SIGUSR1, dumping TAU profile data\n");
  TAU_DB_DUMP_PREFIX("profile");
}

void tauToggleInstrumentationHandler(int sig) {
  fprintf (stderr, "Caught SIGUSR2, toggling TAU instrumentation\n");
  if (RtsLayer::TheEnableInstrumentation())
  {
    RtsLayer::TheEnableInstrumentation() = false;
  }
  else 
  {
    RtsLayer::TheEnableInstrumentation() = true;
  }
}
static x_uint64 getTimeStamp() {
  x_uint64 timestamp;
#ifdef TAU_WINDOWS
  timestamp = TauWindowsUsecD();
#else
  struct timeval tp;
  gettimeofday (&tp, 0);
  timestamp = (x_uint64)tp.tv_sec * (x_uint64)1e6 + (x_uint64)tp.tv_usec;
#endif
  return timestamp;
}

// We keep track of the timestamp at the initialization of TheFunctionDB()
// see FunctionInfo.cpp
// I do this because otherwise it is impossible to determine the duration
// of the first snapshot.  The user may not be using a time based metric
// so I can't just look at the top-level timer.  Instead, I just grab this
// at the earliest point.
static x_uint64 firstTimeStamp;
bool Tau_snapshot_initialization() {

#ifndef TAU_DISABLE_SIGUSR
  /* register SIGUSR1 handler */
  if (signal(SIGUSR1, tauSignalHandler) == SIG_ERR) {
    perror("failed to register TAU profile dump signal handler");
  }

  if (signal(SIGUSR2, tauToggleInstrumentationHandler) == SIG_ERR) {
    perror("failed to register TAU instrumentation toggle signal handler");
  }
#endif

  firstTimeStamp = getTimeStamp();
  return true;
}


// Static holder for snapshot file handles
static FILE **TauGetSnapshotFiles() {
  static FILE **snapshotFiles = NULL;
  if (!snapshotFiles) {
    snapshotFiles = new FILE*[TAU_MAX_THREADS];
    for (int i=0; i<TAU_MAX_THREADS; i++) {
      snapshotFiles[i] = NULL;
    }
  }
  return snapshotFiles;
}

// Static holder for snapshot event counts
static int *TauGetSnapshotEventCounts() {
  static int eventCounts[TAU_MAX_THREADS];
  return eventCounts;
}

// Static holder for snapshot user event counts
static int *TauGetSnapshotUserEventCounts() {
  static int userEventCounts[TAU_MAX_THREADS];
  return userEventCounts;
}

// Static holder for metadata name/value pairs
// These come from TAU_METADATA calls
static map<string,string> &TheMetaData() {
  static map<string,string> metadata;
  return metadata;
}


// doesn't work on ia64 for some reason
// #ifdef __linux__
// #include <sys/types.h>
// #include <linux/unistd.h>
// _syscall0(pid_t,gettid) 
// pid_t gettid(void);
// #endif /* __linux__ */

static int ReadFullLine(char *line, FILE *fp) {
  int ch;
  int i = 0; 
  while ( (ch = fgetc(fp)) && ch != EOF && ch != (int) '\n') {
    line[i++] = (unsigned char) ch;
  }
  line[i] = '\0'; 
  return i; 
}

static void writeXMLString(FILE *f, const char *s) {
  if (!s) return;
  
  bool useCdata = false;
  
  if (strchr(s, '<') || strchr(s, '&')) {
    useCdata = true;
  }
  
  if (strstr(s, "]]>") || strchr(s, '\n')) {
    useCdata = false;
  }
  
  if (useCdata) {
    fprintf (f,"<![CDATA[%s]]>",s);
    return;
  }

  // could grow up to 5 times in length
  char *str = (char *) malloc (6*strlen(s)+10);
  char *d = str;
  while (*s) {
    if ((*s == '<') || (*s == '>') || (*s == '&') || (*s == '\n')) {
      // escape these characters
      if (*s == '<') {
	strcpy (d,"&lt;");
	d+=4;
      }
      
      if (*s == '>') {
	strcpy (d,"&gt;");
	d+=4;
      }

      if (*s == '\n') {
	strcpy (d,"&#xa;");
	d+=5;
      }
      
      if (*s == '&') {
	strcpy (d,"&amp;");
	d+=5;
      }
    } else {
      *d = *s;
      d++; 
    }
    
    s++;
  }
  *d = 0;
  
  fprintf (f,"%s",str);
  free (str);
}

static void writeTagXML(FILE *f, const char *tag, const char *s, bool newline) {
  fprintf (f, "<%s>", tag);
  writeXMLString(f, s);
  fprintf (f, "</%s>",tag);
  if (newline) {
    fprintf (f, "\n");
  }
}


static void writeXMLAttribute(FILE *f, const char *name, const char *value, bool newline) {
  const char *endl = "";
  if (newline) {
    endl = "\n";
  }

  fprintf (f, "<attribute>%s<name>", endl);
  writeXMLString(f, name);
  fprintf (f, "</name>%s<value>", endl);
  writeXMLString(f, value);
  fprintf (f, "</value>%s</attribute>%s", endl, endl);
}


static void writeXMLAttribute(FILE *f, const char *name, const int value, bool newline) {
  char str[4096];
  sprintf (str, "%d", value);
  writeXMLAttribute(f, name, str, newline);
}

static int writeXMLTime(FILE *fp, bool newline) {

   time_t theTime = time(NULL);
//    char *stringTime = ctime(&theTime);
//    fprintf (fp, "<time>%s</time>\n", stringTime);

//    char *day = strtok(stringTime," ");
//    char *month = strtok(NULL," ");
//    char *dayInt = strtok(NULL," ");
//    char *time = strtok(NULL," ");
//    char *year = strtok(NULL," ");
//    //Get rid of the mewline.
//    year[4] = '\0';
//    char *newStringTime = new char[1024];
//    sprintf(newStringTime,"%s-%s-%s-%s-%s",day,month,dayInt,time,year);
//    fprintf (fp, "<date>%s</date>\n", newStringTime);

   const char *endl = "";
   if (newline) {
     endl = "\n";
   }

   char buf[4096];
   struct tm *thisTime = gmtime(&theTime);
   strftime (buf,4096,"%Y-%m-%dT%H:%M:%SZ", thisTime);
   fprintf (fp, "<attribute><name>UTC Time</name><value>%s</value></attribute>%s", buf, endl);

   thisTime = localtime(&theTime);
   strftime (buf,4096,"%Y-%m-%dT%H:%M:%S", thisTime);


   char tzone[7];
   strftime (tzone, 7, "%z", thisTime);
   if (strlen(tzone) == 5) {
     tzone[6] = 0;
     tzone[5] = tzone[4];
     tzone[4] = tzone[3];
     tzone[3] = ':';
   }
   fprintf (fp, "<attribute><name>Local Time</name><value>%s%s</value></attribute>%s", buf, tzone, endl);

   // write out the timestamp (number of microseconds since epoch (unsigned long long)
#ifdef TAU_WINDOWS
   fprintf (fp, "<attribute><name>Timestamp</name><value>%I64d</value></attribute>%s", getTimeStamp(), endl);
#else
   fprintf (fp, "<attribute><name>Timestamp</name><value>%lld</value></attribute>%s", getTimeStamp(), endl);
#endif

   return 0;
}


static char *removeRuns(char *str) {
  // replaces runs of spaces with a single space

  // also removes leading whitespace
  while (*str && *str == ' ') str++;

  int len = strlen(str);
  for (int i=0; i<len; i++) {
    if (str[i] == ' ') {
      int idx = i+1;
      while (idx < len && str[idx] == ' ') {
	idx++;
      }
      int skip = idx - i - 1;
      for (int j=i+1; j<=len-skip; j++) {
	str[j] = str[j+skip];
      }
    }
  }
  return str;
}





static bool helperIsFunction(FunctionInfo *fi, Profiler *profiler) {
  
#ifdef TAU_CALLPATH
  if (fi == profiler->ThisFunction || fi == profiler->CallPathFunction) {
#else
  if (fi == profiler->ThisFunction) { 
#endif
    return true;
  }
  return false;
}




static void writeEventXML(FILE *f, int id, FunctionInfo *fi) {
  fprintf (f, "<event id=\"%d\"><name>", id);
  writeXMLString(f, fi->GetName());
  fprintf (f, "</name><group>");
  writeXMLString(f, fi->GetAllGroups());
  fprintf (f, "</group></event>\n");
  return;
}

static void writeUserEventXML(FILE *f, int id, TauUserEvent *ue) {
  fprintf (f, "<userevent id=\"%d\"><name>", id);
  writeXMLString(f, ue->GetEventName());
  fprintf (f, "</name></userevent>\n");
  return;
}

static int writeMetaData(FILE *fp, bool newline, int counter) {
  const char *endl = "";
  if (newline) {
    endl = "\n";
  }

  fprintf (fp, "<metadata>%s", endl);


  if (counter != -1) {
#ifndef TAU_MULTIPLE_COUNTERS
    writeXMLAttribute(fp, "Metric Name", RtsLayer::getSingleCounterName(), newline);
#else
    writeXMLAttribute(fp, "Metric Name", MultipleCounterLayer::getCounterNameAt(counter), newline);
#endif
  }


  char tmpstr[1024];
#ifdef TAU_WINDOWS
  sprintf (tmpstr, "%I64d", firstTimeStamp);
#else
  sprintf (tmpstr, "%lld", firstTimeStamp);
#endif
  writeXMLAttribute(fp, "Starting Timestamp", tmpstr, newline);

  writeXMLTime(fp, newline);

#ifndef TAU_WINDOWS

  // try to grab meta-data
  char hostname[4096];
  gethostname(hostname,4096);
  writeXMLAttribute(fp, "Hostname", hostname, newline);

  struct utsname archinfo;

  uname (&archinfo);
  writeXMLAttribute(fp, "OS Name", archinfo.sysname, newline);
  writeXMLAttribute(fp, "OS Version", archinfo.version, newline);
  writeXMLAttribute(fp, "OS Release", archinfo.release, newline);
  writeXMLAttribute(fp, "OS Machine", archinfo.machine, newline);
  writeXMLAttribute(fp, "Node Name", archinfo.nodename, newline);

  writeXMLAttribute(fp, "TAU Architecture", TAU_ARCH, newline);
  writeXMLAttribute(fp, "TAU Config", TAU_CONFIG, newline);

  writeXMLAttribute(fp, "pid", getpid(), newline);
#endif


#ifdef TAU_BGL
  char buffer[4096];
  char location[BGLPERSONALITY_MAX_LOCATION];
  BGLPersonality personality;

  rts_get_personality(&personality, sizeof(personality));
  BGLPersonality_getLocationString(&personality, location);

  sprintf (buffer, "(%d,%d,%d)", BGLPersonality_xCoord(&personality),
	   BGLPersonality_yCoord(&personality),
	   BGLPersonality_zCoord(&personality));
  writeXMLAttribute(fp, "BGL Coords", buffer, newline);

  writeXMLAttribute(fp, "BGL Processor ID", rts_get_processor_id(), newline);

  sprintf (buffer, "(%d,%d,%d)", BGLPersonality_xSize(&personality),
	   BGLPersonality_ySize(&personality),
	   BGLPersonality_zSize(&personality));
  writeXMLAttribute(fp, "BGL Size", buffer, newline);


  if (BGLPersonality_virtualNodeMode(&personality)) {
    writeXMLAttribute(fp, "BGL Node Mode", "Virtual", newline);
  } else {
    writeXMLAttribute(fp, "BGL Node Mode", "Coprocessor", newline);
  }

  sprintf (buffer, "(%d,%d,%d)", BGLPersonality_isTorusX(&personality),
	   BGLPersonality_isTorusY(&personality),
	   BGLPersonality_isTorusZ(&personality));
  writeXMLAttribute(fp, "BGL isTorus", buffer, newline);

  writeXMLAttribute(fp, "BGL DDRSize", BGLPersonality_DDRSize(&personality), newline);
  writeXMLAttribute(fp, "BGL DDRModuleType", personality.DDRModuleType, newline);
  writeXMLAttribute(fp, "BGL Location", location, newline);

  writeXMLAttribute(fp, "BGL rankInPset", BGLPersonality_rankInPset(&personality), newline);
  writeXMLAttribute(fp, "BGL numNodesInPset", BGLPersonality_numNodesInPset(&personality), newline);
  writeXMLAttribute(fp, "BGL psetNum", BGLPersonality_psetNum(&personality), newline);
  writeXMLAttribute(fp, "BGL numPsets", BGLPersonality_numPsets(&personality), newline);

  sprintf (buffer, "(%d,%d,%d)", BGLPersonality_xPsetSize(&personality),
	   BGLPersonality_yPsetSize(&personality),
	   BGLPersonality_zPsetSize(&personality));
  writeXMLAttribute(fp, "BGL PsetSize", buffer, newline);

  sprintf (buffer, "(%d,%d,%d)", BGLPersonality_xPsetOrigin(&personality),
	   BGLPersonality_yPsetOrigin(&personality),
	   BGLPersonality_zPsetOrigin(&personality));
  writeXMLAttribute(fp, "BGL PsetOrigin", buffer, newline);

  sprintf (buffer, "(%d,%d,%d)", BGLPersonality_xPsetCoord(&personality),
	   BGLPersonality_yPsetCoord(&personality),
	   BGLPersonality_zPsetCoord(&personality));
  writeXMLAttribute(fp, "BGL PsetCoord", buffer, newline);
#endif /* TAU_BGL */


#ifdef __linux__
  // doesn't work on ia64 for some reason
  //fprintf (fp, "\t<linux_tid>%d</linux_tid>\n", gettid());

  // try to grab CPU info
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (f) {
    char line[4096];
    while (ReadFullLine(line, f)) {
      char buf[4096];
      char *value = strstr(line,":")+2;
      value = removeRuns(value);

      if (strncmp(line, "vendor_id", 9) == 0) {
	writeXMLAttribute(fp, "CPU Vendor", value, newline);
      }
      if (strncmp(line, "cpu MHz", 7) == 0) {
	writeXMLAttribute(fp, "CPU MHz", value, newline);
      }
      if (strncmp(line, "clock", 5) == 0) {
	writeXMLAttribute(fp, "CPU MHz", value, newline);
      }
      if (strncmp(line, "model name", 10) == 0) {
	writeXMLAttribute(fp, "CPU Type", value, newline);
      }
      if (strncmp(line, "family", 6) == 0) {
	writeXMLAttribute(fp, "CPU Type", value, newline);
      }
      if (strncmp(line, "cpu\t", 4) == 0) {
	writeXMLAttribute(fp, "CPU Type", value, newline);
      }
      if (strncmp(line, "cache size", 10) == 0) {
	writeXMLAttribute(fp, "Cache Size", value, newline);
      }
      if (strncmp(line, "cpu cores", 9) == 0) {
	writeXMLAttribute(fp, "CPU Cores", value, newline);
      }
    }
    fclose(f);
  }

  f = fopen("/proc/meminfo", "r");
  if (f) {
    char line[4096];
    while (ReadFullLine(line, f)) {
      char buf[4096];
      char *value = strstr(line,":")+2;
      value = removeRuns(value);

      if (strncmp(line, "MemTotal", 8) == 0) {
	writeXMLAttribute(fp, "Memory Size", value, newline);
      }
    }
    fclose(f);
  }


  char buffer[4096];
  bzero(buffer, 4096);
  int rc = readlink("/proc/self/exe", buffer, 4096);
  if (rc != -1) {
    writeXMLAttribute(fp, "Executable", buffer, newline);
  }
  bzero(buffer, 4096);
  rc = readlink("/proc/self/cwd", buffer, 4096);
  if (rc != -1) {
    writeXMLAttribute(fp, "CWD", buffer, newline);
  }
#endif /* __linux__ */

  char *user = getenv("USER");
  if (user != NULL) {
    writeXMLAttribute(fp, "username", user, newline);
  }


  // write out the user-specified (some from TAU) attributes
  for (map<string,string>::iterator it = TheMetaData().begin(); it != TheMetaData().end(); ++it) {
    const char *name = it->first.c_str();
    const char *value = it->second.c_str();
    writeXMLAttribute(fp, name, value, newline);
  }


  fprintf (fp, "</metadata>%s", endl);

  return 0;
}




int Profiler::Snapshot(char *name, bool finalize, int tid) {
   FILE *fp = TauGetSnapshotFiles()[tid];
   if (finalize && !fp) { 
     // finalize is true at the end of execution (regular profile output), if we haven't written a snapshot, don't bother
     return 0;
   }

   TAU_PROFILE_TIMER(timer, "TAU_PROFILE_SNAPSHOT()", " ", TAU_IO);

   if (!finalize) {
     // don't start the timer here, otherwise we'll go into an infinite loop
     // since our timer will always be on the stack
     TAU_PROFILE_START(timer);
   }


   //  printf ("Writing Snapshot [node %d:%d]\n",  RtsLayer::myNode(), tid);


#ifndef TAU_MULTIPLE_COUNTERS
   double currentTime = RtsLayer::getUSecD(tid); 
#else
   double currentTime[MAX_TAU_COUNTERS];
   for (int c=0; c<MAX_TAU_COUNTERS; c++) {
     currentTime[c] = 0;
   }
   RtsLayer::getUSecD(tid, currentTime);
#endif	

   char threadid[4096];
#ifdef TAU_WINDOWS
   sprintf(threadid, "%d.%d.%d", RtsLayer::myNode(), RtsLayer::myContext(), tid);
#else
   sprintf(threadid, "%d.%d.%d.%d", RtsLayer::myNode(), RtsLayer::myContext(), tid, getpid());
#endif

   RtsLayer::LockDB();
   int numFunc = TheFunctionDB().size();
   int numEvents = TheEventDB().size();

   if (!fp) {
     // create file 
     const char *dirname;
     const char *currentDirectory = ".";
     if ((dirname = getenv("PROFILEDIR")) == NULL) {
       dirname = currentDirectory;
     }

     char filename[4096];
     sprintf (filename,"%s/snapshot.%d.%d.%d", dirname, 
	      RtsLayer::myNode(), RtsLayer::myContext(), tid);

     if ((fp = fopen (filename, "w+")) == NULL) {
       char errormsg[4096];
       sprintf(errormsg,"Error: Could not create %s",filename);
       perror(errormsg);
       RtsLayer::UnLockDB();
       return 0;
     }

     // assign it back to the global structure for this thread
     TauGetSnapshotFiles()[tid] = fp;

     fprintf (fp, "<profile_xml>\n");

     fprintf (fp, "\n<thread id=\"%s\" node=\"%d\" context=\"%d\" thread=\"%d\">\n", threadid,
	      RtsLayer::myNode(), RtsLayer::myContext(), tid);
     writeMetaData(fp, true, -1);
     fprintf (fp, "</thread>\n");

     fprintf (fp, "\n<definitions thread=\"%s\">\n", threadid);

#ifndef TAU_MULTIPLE_COUNTERS
     fprintf (fp, "<metric id=\"0\">\n");
     writeTagXML(fp, "name", RtsLayer::getSingleCounterName(), true);
     writeTagXML(fp, "units", "unknown", true);
     fprintf (fp, "</metric>\n");
#else
      for(int i=0;i<MAX_TAU_COUNTERS;i++){
	if(MultipleCounterLayer::getCounterUsed(i)){
	  char *tmpChar = MultipleCounterLayer::getCounterNameAt(i);
	  fprintf (fp, "<metric id=\"%d\">", i);
	  writeTagXML(fp, "name", tmpChar, true);
	  writeTagXML(fp, "units", "unknown", true);
	  fprintf (fp, "</metric>\n");
	}
     }
#endif


     // write out events seen (so far)
     for (int i=0; i < numFunc; i++) {
       FunctionInfo *fi = TheFunctionDB()[i];
       writeEventXML(fp, i, fi);
     }

     // remember the number of events we've written to the snapshot file
     TauGetSnapshotEventCounts()[tid] = numFunc;

     // write out user events seen (so far)
     for (int i=0; i < numEvents; i++) {
       TauUserEvent *ue = TheEventDB()[i];
       writeUserEventXML(fp, i, ue);
     }

     // remember the number of userevents we've written to the snapshot file
     TauGetSnapshotUserEventCounts()[tid] = numEvents;


     fprintf (fp, "</definitions>\n");
   } else {
     fprintf (fp, "<profile_xml>\n");
   }


   
   // write out new events since the last snapshot
   if (TauGetSnapshotEventCounts()[tid] != numFunc) {
     fprintf (fp, "\n<definitions thread=\"%s\">\n", threadid);
     for (int i=TauGetSnapshotEventCounts()[tid]; i < numFunc; i++) {
       FunctionInfo *fi = TheFunctionDB()[i];
       writeEventXML(fp, i, fi);
     }
     fprintf (fp, "</definitions>\n");
     TauGetSnapshotEventCounts()[tid] = numFunc;
   }

   // write out new user events since the last snapshot
   if (TauGetSnapshotUserEventCounts()[tid] != numEvents) {
     fprintf (fp, "\n<definitions thread=\"%s\">\n", threadid);
     for (int i=TauGetSnapshotUserEventCounts()[tid]; i < numEvents; i++) {
       TauUserEvent *ue = TheEventDB()[i];
       writeUserEventXML(fp, i, ue);
     }
     fprintf (fp, "</definitions>\n");
     TauGetSnapshotUserEventCounts()[tid] = numEvents;
   }


   // now write the actual profile data for this snapshot
   fprintf (fp, "\n<profile thread=\"%s\">\n", threadid);
   fprintf (fp, "<name>");
   writeXMLString(fp, name);
   fprintf (fp, "</name>\n");

#ifdef TAU_WINDOWS
   fprintf (fp, "<timestamp>%I64d</timestamp>\n", getTimeStamp());
#else
   fprintf (fp, "<timestamp>%lld</timestamp>\n", getTimeStamp());
#endif


#ifndef TAU_MULTIPLE_COUNTERS
   fprintf (fp, "<interval_data metrics=\"0\">\n");
#else
   char metricList[4096];
   char *loc = metricList;
   for (int c=0; c<MAX_TAU_COUNTERS; c++) {
     if (MultipleCounterLayer::getCounterUsed(c)) {
       loc += sprintf (loc,"%d ", c);
     }
   }
   fprintf (fp, "<interval_data metrics=\"%s\">\n", metricList);
#endif   


 
   for (int i=0; i < numFunc; i++) {
     FunctionInfo *fi = TheFunctionDB()[i];

#ifndef TAU_MULTIPLE_COUNTERS
     double excltime, incltime;
#else
     double *excltime = NULL, *incltime = NULL;
#endif
     
     if (!fi->GetAlreadyOnStack(tid)) {
       // not on the callstack, the data is complete

       excltime = fi->GetExclTime(tid);
       incltime = fi->GetInclTime(tid); 

     } else {
       // this routine is currently on the callstack
       // we will have to compute the exclusive and inclusive time it has accumulated

       // Start with the data already accumulated
       // Then walk the entire stack, when this function the function in question is found we do two things
       // 1) Compute the current amount that should be added to the inclusive time.
       //    This is simply the current time minus the start time of our function.
       //    If a routine is in the callstack twice, only the highest (top-most) value
       //    will be retained, this is correct.
       // 2) Add to the exclusive value by subtracting the start time of the current
       //    child (if there is one) from the duration of this function so far.

       incltime = fi->GetInclTime(tid); 
       excltime = fi->GetExclTime(tid); 
#ifndef TAU_MULTIPLE_COUNTERS
       double inclusiveToAdd = 0;
       double prevStartTime = 0;

       for (Profiler *current = CurrentProfiler[tid]; current != 0; current = current->ParentProfiler) {
	 if (helperIsFunction(fi, current)) {
	   inclusiveToAdd = currentTime - current->StartTime; 
	   excltime += inclusiveToAdd - prevStartTime;
	 }
	 prevStartTime = currentTime - current->StartTime;  
       }
       incltime += inclusiveToAdd;
#else
       double inclusiveToAdd[MAX_TAU_COUNTERS];
       double prevStartTime[MAX_TAU_COUNTERS];
       for (int c=0; c<MAX_TAU_COUNTERS; c++) {
	 inclusiveToAdd[c] = 0;
	 prevStartTime[c] = 0;
       }

       for (Profiler *current = CurrentProfiler[tid]; current != 0; current = current->ParentProfiler) {
	 if (helperIsFunction(fi, current)) {
	   for (int c=0; c<MAX_TAU_COUNTERS; c++) {
	     inclusiveToAdd[c] = currentTime[c] - current->StartTime[c]; 
	     excltime[c] += inclusiveToAdd[c] - prevStartTime[c];
	   }
	 }
	 for (int c=0; c<MAX_TAU_COUNTERS; c++) {
	   prevStartTime[c] = currentTime[c] - current->StartTime[c];  
	 }
       }
       for (int c=0; c<MAX_TAU_COUNTERS; c++) {
	 incltime[c] += inclusiveToAdd[c];
       }
#endif
     }
     
#ifndef TAU_MULTIPLE_COUNTERS
     fprintf (fp, "%d %ld %ld %.16G %.16G \n", i, fi->GetCalls(tid), fi->GetSubrs(tid), 
	      excltime, incltime);
#else
     fprintf (fp, "%d %ld %ld ", i, fi->GetCalls(tid), fi->GetSubrs(tid));
     for (int c=0; c<MAX_TAU_COUNTERS; c++) {
       if (MultipleCounterLayer::getCounterUsed(c)) {
	 fprintf (fp, "%.16G %.16G ", excltime[c], incltime[c]);
       }
     }
     fprintf (fp, "\n");

#endif
     
   }
   fprintf (fp, "</interval_data>\n");


   // now write the user events
   fprintf (fp, "<atomic_data>\n");
   for (int i=0; i < numEvents; i++) {
     TauUserEvent *ue = TheEventDB()[i];
           fprintf(fp, "%d %ld %.16G %.16G %.16G %.16G\n", 
	     i, ue->GetNumEvents(tid), ue->GetMax(tid),
	     ue->GetMin(tid), ue->GetMean(tid), ue->GetSumSqr(tid));
   }
   fprintf (fp, "</atomic_data>\n");



   fprintf (fp, "</profile>\n");


   fprintf (fp, "\n</profile_xml>\n");

//    if (finalize) {
//      fprintf (fp, "\n</profile_xml>\n");
//      //     fclose(fp);
//    }

   RtsLayer::UnLockDB();
   
   if (!finalize) {
     TAU_PROFILE_STOP(timer);
   }

   return 0;
}


extern "C" void Tau_metadata(char *name, char *value) {
  // make copies
  char *myName = strdup(name);
  char *myValue = strdup(value);
  TheMetaData()[myName] = myValue;
}


int Tau_writeProfileMetaData(FILE *fp, int counter) {
#ifdef TAU_DISABLE_METADATA
  return 0;
#endif
  return writeMetaData(fp, false, counter);
}


