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

#ifdef TAU_IBM_XLC_BGP
#undef TAU_BGP
#endif
/* NOTE: IBM BG/P XLC does not work with metadata when it is compiled with -qpic=large */


#ifdef TAU_BGP
/* header files for BlueGene/P */
#include <bgp_personality.h>
#include <bgp_personality_inlines.h>
#include <kernel_interface.h>
#endif // TAU_BGP

#if (defined (TAU_CATAMOUNT) && defined (PTHREADS))
#define _BITS_PTHREADTYPES_H 1
#endif

#include <signal.h>
#include <stdarg.h>

char *TauGetCounterString(void);

void tauSignalHandler(int sig) {
  fprintf (stderr, "Caught SIGUSR1, dumping TAU profile data\n");
  TAU_DB_DUMP_PREFIX("profile");
}

void tauToggleInstrumentationHandler(int sig) {
  fprintf (stderr, "Caught SIGUSR2, toggling TAU instrumentation\n");
  if (RtsLayer::TheEnableInstrumentation()) {
    RtsLayer::TheEnableInstrumentation() = false;
  } else {
    RtsLayer::TheEnableInstrumentation() = true;
  }
}

extern "C" x_uint64 Tau_getTimeStamp() {
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

  firstTimeStamp = Tau_getTimeStamp();
  return true;
}

x_uint64 Tau_get_firstTimeStamp() {
  return firstTimeStamp;
}

typedef struct outputDevice_ {
  FILE *fp;
  int type; // 0 = file, 1 = buffer
  char *buffer;
  int bufidx;
  int buflen;
} outputDevice;

#define OUTPUT_FILE 0
#define OUTPUT_BUFFER 1
#define INITIAL_BUFFER 5000000
#define THRESHOLD 100000


// Static holder for snapshot file handles
static outputDevice **TauGetSnapshotFiles() {
  static outputDevice **snapshotFiles = NULL;
  int i;
  if (!snapshotFiles) {
    snapshotFiles = new outputDevice*[TAU_MAX_THREADS];
    for (i=0; i<TAU_MAX_THREADS; i++) {
      snapshotFiles[i] = NULL;
    }
  }
  return snapshotFiles;
}

extern "C" char *getSnapshotBuffer() {
  // only support thread 0 right now

  char *buf = TauGetSnapshotFiles()[0]->buffer;
  return buf;
}

extern "C" int getSnapshotBufferLength() {
  return TauGetSnapshotFiles()[0]->bufidx;
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

class MetaDataRepo : public map<string,string> {
public :
  ~MetaDataRepo() {
    Tau_destructor_trigger();
  }
};

// Static holder for metadata name/value pairs
// These come from TAU_METADATA calls
static map<string,string> &TheMetaData() {
  static MetaDataRepo metadata;
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

static int output(outputDevice *out, const char *format, ...) {
  int rs;
  va_list args;
  if (out->type == OUTPUT_BUFFER) {
    va_start(args, format);
    rs = vsprintf(out->buffer+out->bufidx, format, args);
    va_end(args);
    out->bufidx+=rs;
    if (out->bufidx+THRESHOLD > out->buflen) {
      out->buflen = out->buflen * 2;
      out->buffer = (char*) realloc (out->buffer, out->buflen);
    }

  } else {
    va_start(args, format);
    rs = vfprintf(out->fp, format, args);
    va_end(args);
  }
  return rs;
}

static void writeXMLString(outputDevice *out, const char *s) {
  if (!s) return;
  
  bool useCdata = false;
  
  if (strchr(s, '<') || strchr(s, '&')) {
    useCdata = true;
  }
  
  if (strstr(s, "]]>") || strchr(s, '\n')) {
    useCdata = false;
  }
  
  if (useCdata) {
    output (out,"<![CDATA[%s]]>",s);
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
  
  output (out,"%s",str);
  free (str);
}

static void writeTagXML(outputDevice *out, const char *tag, const char *s, bool newline) {
  output (out, "<%s>", tag);
  writeXMLString(out, s);
  output (out, "</%s>",tag);
  if (newline) {
    output (out, "\n");
  }
}


static void writeXMLAttribute(outputDevice *out, const char *name, const char *value, bool newline) {
  const char *endl = "";
  if (newline) {
    endl = "\n";
  }

  output (out, "<attribute>%s<name>", endl);
  writeXMLString(out, name);
  output (out, "</name>%s<value>", endl);
  writeXMLString(out, value);
  output (out, "</value>%s</attribute>%s", endl, endl);
}


static void writeXMLAttribute(outputDevice *out, const char *name, const int value, bool newline) {
  char str[4096];
  sprintf (str, "%d", value);
  writeXMLAttribute(out, name, str, newline);
}

static int writeXMLTime(outputDevice *out, bool newline) {

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
   output (out, "<attribute><name>UTC Time</name><value>%s</value></attribute>%s", buf, endl);

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
   output (out, "<attribute><name>Local Time</name><value>%s%s</value></attribute>%s", buf, tzone, endl);

   // write out the timestamp (number of microseconds since epoch (unsigned long long)
#ifdef TAU_WINDOWS
   output (out, "<attribute><name>Timestamp</name><value>%I64d</value></attribute>%s", Tau_getTimeStamp(), endl);
#else
   output (out, "<attribute><name>Timestamp</name><value>%lld</value></attribute>%s", Tau_getTimeStamp(), endl);
#endif

   return 0;
}


static char *removeRuns(char *str) {
  int i, idx;
  int len; 
  // replaces runs of spaces with a single space

  if (!str) {
    return str; /* do nothing with a null string */
  }

  // also removes leading whitespace
  while (*str && *str == ' ') str++;

  len = strlen(str);
  for (i=0; i<len; i++) {
    if (str[i] == ' ') {
      idx = i+1;
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


static void writeEventXML(outputDevice *out, int id, FunctionInfo *fi) {
  output (out, "<event id=\"%d\"><name>", id);
  writeXMLString(out, fi->GetName());
  output (out, "</name><group>");
  writeXMLString(out, fi->GetAllGroups());
  output (out, "</group></event>\n");
  return;
}

static void writeUserEventXML(outputDevice *out, int id, TauUserEvent *ue) {
  output (out, "<userevent id=\"%d\"><name>", id);
  writeXMLString(out, ue->GetEventName());
  output (out, "</name></userevent>\n");
  return;
}

static int writeMetaData(outputDevice *out, bool newline, int counter) {
  const char *endl = "";
  if (newline) {
    endl = "\n";
  }

  output (out, "<metadata>%s", endl);

  if (counter != -1) {
    writeXMLAttribute(out, "Metric Name", RtsLayer::getCounterName(counter), newline);
  }

  char tmpstr[1024];
#ifdef TAU_WINDOWS
  sprintf (tmpstr, "%I64d", firstTimeStamp);
#else
  sprintf (tmpstr, "%lld", firstTimeStamp);
#endif
  writeXMLAttribute(out, "Starting Timestamp", tmpstr, newline);

  writeXMLTime(out, newline);

#ifndef TAU_WINDOWS
  // try to grab meta-data
  char hostname[4096];
  gethostname(hostname,4096);
  writeXMLAttribute(out, "Hostname", hostname, newline);

  struct utsname archinfo;

  uname (&archinfo);
  writeXMLAttribute(out, "OS Name", archinfo.sysname, newline);
  writeXMLAttribute(out, "OS Version", archinfo.version, newline);
  writeXMLAttribute(out, "OS Release", archinfo.release, newline);
  writeXMLAttribute(out, "OS Machine", archinfo.machine, newline);
  writeXMLAttribute(out, "Node Name", archinfo.nodename, newline);

  writeXMLAttribute(out, "TAU Architecture", TAU_ARCH, newline);
  writeXMLAttribute(out, "TAU Config", TAU_CONFIG, newline);
  writeXMLAttribute(out, "TAU Makefile", TAU_MAKEFILE, newline);
  writeXMLAttribute(out, "TAU Version", TAU_VERSION, newline);

  writeXMLAttribute(out, "pid", getpid(), newline);
#endif

#ifdef TAU_BGL
  char bglbuffer[4096];
  char location[BGLPERSONALITY_MAX_LOCATION];
  BGLPersonality personality;

  rts_get_personality(&personality, sizeof(personality));
  BGLPersonality_getLocationString(&personality, location);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xCoord(&personality),
	   BGLPersonality_yCoord(&personality),
	   BGLPersonality_zCoord(&personality));
  writeXMLAttribute(out, "BGL Coords", bglbuffer, newline);

  writeXMLAttribute(out, "BGL Processor ID", rts_get_processor_id(), newline);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xSize(&personality),
	   BGLPersonality_ySize(&personality),
	   BGLPersonality_zSize(&personality));
  writeXMLAttribute(out, "BGL Size", bglbuffer, newline);


  if (BGLPersonality_virtualNodeMode(&personality)) {
    writeXMLAttribute(out, "BGL Node Mode", "Virtual", newline);
  } else {
    writeXMLAttribute(out, "BGL Node Mode", "Coprocessor", newline);
  }

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_isTorusX(&personality),
	   BGLPersonality_isTorusY(&personality),
	   BGLPersonality_isTorusZ(&personality));
  writeXMLAttribute(out, "BGL isTorus", bglbuffer, newline);

  writeXMLAttribute(out, "BGL DDRSize", BGLPersonality_DDRSize(&personality), newline);
  writeXMLAttribute(out, "BGL DDRModuleType", personality.DDRModuleType, newline);
  writeXMLAttribute(out, "BGL Location", location, newline);

  writeXMLAttribute(out, "BGL rankInPset", BGLPersonality_rankInPset(&personality), newline);
  writeXMLAttribute(out, "BGL numNodesInPset", BGLPersonality_numNodesInPset(&personality), newline);
  writeXMLAttribute(out, "BGL psetNum", BGLPersonality_psetNum(&personality), newline);
  writeXMLAttribute(out, "BGL numPsets", BGLPersonality_numPsets(&personality), newline);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetSize(&personality),
	   BGLPersonality_yPsetSize(&personality),
	   BGLPersonality_zPsetSize(&personality));
  writeXMLAttribute(out, "BGL PsetSize", bglbuffer, newline);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetOrigin(&personality),
	   BGLPersonality_yPsetOrigin(&personality),
	   BGLPersonality_zPsetOrigin(&personality));
  writeXMLAttribute(out, "BGL PsetOrigin", bglbuffer, newline);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetCoord(&personality),
	   BGLPersonality_yPsetCoord(&personality),
	   BGLPersonality_zPsetCoord(&personality));
  writeXMLAttribute(out, "BGL PsetCoord", bglbuffer, newline);
#endif /* TAU_BGL */

#ifdef TAU_BGP
  char bgpbuffer[4096];
  char location[BGPPERSONALITY_MAX_LOCATION];
  _BGP_Personality_t personality;

  Kernel_GetPersonality(&personality, sizeof(_BGP_Personality_t));
  BGP_Personality_getLocationString(&personality, location);

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xCoord(&personality),
	   BGP_Personality_yCoord(&personality),
	   BGP_Personality_zCoord(&personality));
  writeXMLAttribute(out, "BGP Coords", bgpbuffer, newline);

  writeXMLAttribute(out, "BGP Processor ID", Kernel_PhysicalProcessorID(), newline);

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xSize(&personality),
	   BGP_Personality_ySize(&personality),
	   BGP_Personality_zSize(&personality));
  writeXMLAttribute(out, "BGP Size", bgpbuffer, newline);


  if (Kernel_ProcessCount() > 1) {
    writeXMLAttribute(out, "BGP Node Mode", "Virtual", newline);
  } else {
    sprintf(bgpbuffer, "Coprocessor (%d)", Kernel_ProcessCount);
    writeXMLAttribute(out, "BGP Node Mode", bgpbuffer, newline);
  }

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_isTorusX(&personality),
	   BGP_Personality_isTorusY(&personality),
	   BGP_Personality_isTorusZ(&personality));
  writeXMLAttribute(out, "BGP isTorus", bgpbuffer, newline);

  writeXMLAttribute(out, "BGP DDRSize (MB)", BGP_Personality_DDRSizeMB(&personality), newline);
/* CHECK: 
  writeXMLAttribute(out, "BGP DDRModuleType", personality.DDRModuleType, newline);
*/
  writeXMLAttribute(out, "BGP Location", location, newline);

  writeXMLAttribute(out, "BGP rankInPset", BGP_Personality_rankInPset(&personality), newline);
/*
  writeXMLAttribute(out, "BGP numNodesInPset", Kernel_ProcessCount(), newline);
*/
  writeXMLAttribute(out, "BGP psetSize", BGP_Personality_psetSize(&personality), newline);
  writeXMLAttribute(out, "BGP psetNum", BGP_Personality_psetNum(&personality), newline);
  writeXMLAttribute(out, "BGP numPsets", BGP_Personality_numComputeNodes(&personality), newline);

/* CHECK: 
  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetSize(&personality),
	   BGP_Personality_yPsetSize(&personality),
	   BGP_Personality_zPsetSize(&personality));
  writeXMLAttribute(out, "BGP PsetSize", bgpbuffer, newline);

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetOrigin(&personality),
	   BGP_Personality_yPsetOrigin(&personality),
	   BGP_Personality_zPsetOrigin(&personality));
  writeXMLAttribute(out, "BGP PsetOrigin", bgpbuffer, newline);

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetCoord(&personality),
	   BGP_Personality_yPsetCoord(&personality),
	   BGP_Personality_zPsetCoord(&personality));
  writeXMLAttribute(out, "BGP PsetCoord", bgpbuffer, newline);
*/
#endif /* TAU_BGP */

#ifdef __linux__
  // doesn't work on ia64 for some reason
  //output (out, "\t<linux_tid>%d</linux_tid>\n", gettid());

  // try to grab CPU info
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (f) {
    char line[4096];
    while (ReadFullLine(line, f)) {
      char *value = strstr(line,":");
      if (!value) break;
      else value += 2;

      value = removeRuns(value);

      if (strncmp(line, "vendor_id", 9) == 0) {
	writeXMLAttribute(out, "CPU Vendor", value, newline);
      }
      if (strncmp(line, "vendor", 6) == 0) {
	writeXMLAttribute(out, "CPU Vendor", value, newline);
      }
      if (strncmp(line, "cpu MHz", 7) == 0) {
	writeXMLAttribute(out, "CPU MHz", value, newline);
      }
      if (strncmp(line, "clock", 5) == 0) {
	writeXMLAttribute(out, "CPU MHz", value, newline);
      }
      if (strncmp(line, "model name", 10) == 0) {
	writeXMLAttribute(out, "CPU Type", value, newline);
      }
      if (strncmp(line, "family", 6) == 0) {
	writeXMLAttribute(out, "CPU Type", value, newline);
      }
      if (strncmp(line, "cpu\t", 4) == 0) {
	writeXMLAttribute(out, "CPU Type", value, newline);
      }
      if (strncmp(line, "cache size", 10) == 0) {
	writeXMLAttribute(out, "Cache Size", value, newline);
      }
      if (strncmp(line, "cpu cores", 9) == 0) {
	writeXMLAttribute(out, "CPU Cores", value, newline);
      }
    }
    fclose(f);
  }

  f = fopen("/proc/meminfo", "r");
  if (f) {
    char line[4096];
    while (ReadFullLine(line, f)) {
      char *value = strstr(line,":");

      if (!value) break;
      else value += 2;

      value = removeRuns(value);

      if (strncmp(line, "MemTotal", 8) == 0) {
	writeXMLAttribute(out, "Memory Size", value, newline);
      }
    }
    fclose(f);
  }

  char buffer[4096];
  bzero(buffer, 4096);
  int rc = readlink("/proc/self/exe", buffer, 4096);
  if (rc != -1) {
    writeXMLAttribute(out, "Executable", buffer, newline);
  }
  bzero(buffer, 4096);
  rc = readlink("/proc/self/cwd", buffer, 4096);
  if (rc != -1) {
    writeXMLAttribute(out, "CWD", buffer, newline);
  }
  bzero(buffer, 4096);
  rc = readlink("/proc/self/cmdline", buffer, 4096);
  if (rc != -1) {
    writeXMLAttribute(out, "Command Line", buffer, newline);
  }
#endif /* __linux__ */

  char *user = getenv("USER");
  if (user != NULL) {
    writeXMLAttribute(out, "username", user, newline);
  }


  // Write data from the TAU_METADATA environment variable
  char *tauMetaDataEnvVar = getenv("TAU_METADATA");
  if (tauMetaDataEnvVar != NULL) {
    if (strncmp(tauMetaDataEnvVar, "<attribute>", strlen("<attribute>")) != 0) {
      fprintf (stderr, "Error in formating TAU_METADATA environment variable\n");
    } else {
      output (out, tauMetaDataEnvVar);
    }
  }

  // write out the user-specified (some from TAU) attributes
  for (map<string,string>::iterator it = TheMetaData().begin(); it != TheMetaData().end(); ++it) {
    const char *name = it->first.c_str();
    const char *value = it->second.c_str();
    writeXMLAttribute(out, name, value, newline);
  }

  output (out, "</metadata>%s", endl);
  return 0;
}



static int startNewSnapshotFile(char *threadid, int tid) {
  const char *profiledir = TauEnv_get_profiledir();
  
  outputDevice *out = (outputDevice*) malloc (sizeof(outputDevice));

  if (TauEnv_get_profile_format() == TAU_FORMAT_MERGED) {
    out->type = OUTPUT_BUFFER;
    out->bufidx = 0;
    out->buflen = INITIAL_BUFFER;
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
    out->type = OUTPUT_FILE;
    out->fp = fp;
  }
    
  // assign it back to the global structure for this thread
  TauGetSnapshotFiles()[tid] = out;

  // start of a profile block
  output (out, "<profile_xml>\n");
  
  // thread identifier
  output (out, "\n<thread id=\"%s\" node=\"%d\" context=\"%d\" thread=\"%d\">\n", 
	   threadid, RtsLayer::myNode(), RtsLayer::myContext(), tid);
  writeMetaData(out, true, -1);
  output (out, "</thread>\n");
  
  // definition block
  output (out, "\n<definitions thread=\"%s\">\n", threadid);
  
  for (int i=0; i<Tau_Global_numCounters; i++) {
      const char *tmpChar = RtsLayer::getCounterName(i);
      output (out, "<metric id=\"%d\">", i);
      writeTagXML(out, "name", tmpChar, true);
      writeTagXML(out, "units", "unknown", true);
      output (out, "</metric>\n");
  }

  TauGetSnapshotEventCounts()[tid] = 0;
  TauGetSnapshotUserEventCounts()[tid] = 0;

  output (out, "</definitions>\n");
  return 0;
}


extern "C" int Tau_write_snapshot(const char *name, int finalize) {
  return TauProfiler_Snapshot(name, finalize, RtsLayer::myThread());
}

int TauProfiler_Snapshot(const char *name, bool finalize, int tid) {
   int i, c;
   outputDevice *out = TauGetSnapshotFiles()[tid];

   if (finalize && !out && !(TauEnv_get_profile_format() == TAU_FORMAT_SNAPSHOT)) { 
     // finalize is true at the end of execution (regular profile output), 
     // if we haven't written a snapshot, don't bother, unless snapshot is the
     // requested output format

     if (!(TauEnv_get_profile_format() == TAU_FORMAT_MERGED)) {
       return 0;
     }
   }

   TAU_PROFILE_TIMER(timer, "TAU_PROFILE_SNAPSHOT()", " ", TAU_IO);

   if (!finalize) {
     // don't start the timer here, otherwise we'll go into an infinite loop
     // since our stop call will initiate another final snapshot
     TAU_PROFILE_START(timer);
   }

   char threadid[4096];
   sprintf(threadid, "%d.%d.%d.%d", RtsLayer::myNode(), RtsLayer::myContext(), tid, RtsLayer::getPid());

   RtsLayer::LockDB();
   int numFunc = TheFunctionDB().size();
   int numEvents = TheEventDB().size();

   if (!out) {
     startNewSnapshotFile(threadid, tid);
     out = TauGetSnapshotFiles()[tid];
   } else {
     output (out, "<profile_xml>\n");
   }
   
   // write out new events since the last snapshot
   if (TauGetSnapshotEventCounts()[tid] != numFunc) {
     output (out, "\n<definitions thread=\"%s\">\n", threadid);
     for (int i=TauGetSnapshotEventCounts()[tid]; i < numFunc; i++) {
       FunctionInfo *fi = TheFunctionDB()[i];
       writeEventXML(out, i, fi);
     }
     output (out, "</definitions>\n");
     TauGetSnapshotEventCounts()[tid] = numFunc;
   }

   // write out new user events since the last snapshot
   if (TauGetSnapshotUserEventCounts()[tid] != numEvents) {
     output (out, "\n<definitions thread=\"%s\">\n", threadid);
     for (int i=TauGetSnapshotUserEventCounts()[tid]; i < numEvents; i++) {
       TauUserEvent *ue = TheEventDB()[i];
       writeUserEventXML(out, i, ue);
     }
     output (out, "</definitions>\n");
     TauGetSnapshotUserEventCounts()[tid] = numEvents;
   }

   // now write the actual profile data for this snapshot
   output (out, "\n<profile thread=\"%s\">\n", threadid);
   output (out, "<name>");
   writeXMLString(out, name);
   output (out, "</name>\n");

#ifdef TAU_WINDOWS
   output (out, "<timestamp>%I64d</timestamp>\n", Tau_getTimeStamp());
#else
   output (out, "<timestamp>%lld</timestamp>\n", Tau_getTimeStamp());
#endif

   char metricList[4096];
   char *loc = metricList;
   for (c=0; c<Tau_Global_numCounters; c++) {
       loc += sprintf (loc,"%d ", c);
   }
   output (out, "<interval_data metrics=\"%s\">\n", metricList);

   TauProfiler_updateIntermediateStatistics(tid);

   for (i=0; i < numFunc; i++) {
     FunctionInfo *fi = TheFunctionDB()[i];

     // get currently stored values
     double *incltime = fi->getDumpInclusiveValues(tid);
     double *excltime = fi->getDumpExclusiveValues(tid);
  
     
     output (out, "%d %ld %ld ", i, fi->GetCalls(tid), fi->GetSubrs(tid));
     for (c=0; c<Tau_Global_numCounters; c++) {
       output (out, "%.16G %.16G ", excltime[c], incltime[c]);
     }
     output (out, "\n");
   }
   output (out, "</interval_data>\n");


   // now write the user events
   output (out, "<atomic_data>\n");
   for (i=0; i < numEvents; i++) {
     TauUserEvent *ue = TheEventDB()[i];
           output (out, "%d %ld %.16G %.16G %.16G %.16G\n", 
	     i, ue->GetNumEvents(tid), ue->GetMax(tid),
	     ue->GetMin(tid), ue->GetMean(tid), ue->GetSumSqr(tid));
   }
   output (out, "</atomic_data>\n");

   output (out, "</profile>\n");
   output (out, "\n</profile_xml>\n");

//    if (finalize) {
//      output (out, "\n</profile_xml>\n");
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
  RtsLayer::LockDB();
  TheMetaData()[myName] = myValue;
  RtsLayer::UnLockDB();
}

extern "C" void Tau_context_metadata(char *name, char *value) {
  // get the current calling context
  Profiler *current = TauInternal_CurrentProfiler(RtsLayer::getTid());
  FunctionInfo *fi = current->ThisFunction;
  const char *fname = fi->GetName();

  char *myName = (char*) malloc (strlen(name) + strlen(fname) + 10);
  sprintf (myName, "%s => %s", fname, name);
  char *myValue = strdup(value);
  RtsLayer::LockDB();
  TheMetaData()[myName] = myValue;
  RtsLayer::UnLockDB();
}

extern "C" void Tau_phase_metadata(char *name, char *value) {
  #ifdef TAU_PROFILEPHASE
  // get the current calling context
  Profiler *current = TauInternal_CurrentProfiler(RtsLayer::getTid());
  std::string myString = "";
  while (current != NULL) {
    if (current->GetPhase()) {
      FunctionInfo *fi = current->ThisFunction;
      const char *fname = fi->GetName();
      myString = std::string(fname) + " => " + myString;
    }    
    current = current->ParentProfiler;
  }

  myString = myString + name;
  char *myName = strdup(myString.c_str());
  char *myValue = strdup(value);
 
  RtsLayer::LockDB();
  TheMetaData()[myName] = myValue;
  RtsLayer::UnLockDB();
  #else
  Tau_context_metadata(name, value);
  #endif
}


int Tau_writeProfileMetaData(outputDevice *out, int counter) {
#ifdef TAU_DISABLE_METADATA
  return 0;
#endif
  int retval;
  retval = writeMetaData(out, false, counter);
  return retval;
}

int Tau_writeProfileMetaData(FILE *fp, int counter) {
  outputDevice out;
  out.fp = fp;
  out.type = OUTPUT_FILE;
  return Tau_writeProfileMetaData(&out, counter);
}

