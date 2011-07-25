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

#include <sstream>

#include "tauarch.h"
#include <Profile/Profiler.h>
#include <Profile/tau_types.h>
#include <Profile/TauMetrics.h>
#include <TauUtil.h>
#include <TauXML.h>
#include <TauMetaData.h>

#ifdef TAU_BGL
#include <rts.h>
#include <bglpersonality.h>
#endif


/* Re-enabled since we believe this is now working (2009-11-02) */
/* 
   #ifdef TAU_IBM_XLC_BGP
   #undef TAU_BGP
   #endif
*/
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



class MetaDataRepo : public map<string,string> {
public :
  ~MetaDataRepo() {
    Tau_destructor_trigger();
  }
};

// Static holder for metadata name/value pairs
// These come from Tau_metadata_register calls
map<string,string> &Tau_metadata_getMetaData() {
  static MetaDataRepo metadata;
  return metadata;
}



extern "C" void Tau_metadata(char *name, const char *value) {
#ifdef TAU_DISABLE_METADATA
  return;
#endif

  // make copies
  char *myName = strdup(name);
  char *myValue = strdup(value);
  RtsLayer::LockDB();
  Tau_metadata_getMetaData()[myName] = myValue;
  RtsLayer::UnLockDB();
}


void Tau_metadata_register(char *name, int value) {
  char buf[256];
  sprintf (buf, "%d", value);
  Tau_metadata(name, buf);
}

void Tau_metadata_register(char *name, const char *value) {
  Tau_global_incr_insideTAU();
  Tau_metadata(name, value);
  Tau_global_decr_insideTAU();
}


int Tau_metadata_fillMetaData() {


#ifdef TAU_DISABLE_METADATA
  return 0;
#else


  static int filled = 0;

  if (filled) {
    return 0;
  }
  filled = 1;


#ifdef TAU_WINDOWS
  const char *timeFormat = "%I64d";
#else
  const char *timeFormat = "%lld";
#endif
  

  char tmpstr[4096];
  sprintf (tmpstr, timeFormat, TauMetrics_getInitialTimeStamp());
  Tau_metadata_register("Starting Timestamp", tmpstr);



  time_t theTime = time(NULL);
  struct tm *thisTime = gmtime(&theTime);
  strftime (tmpstr,4096,"%Y-%m-%dT%H:%M:%SZ", thisTime);
  Tau_metadata_register("UTC Time", tmpstr);


  thisTime = localtime(&theTime);
  char buf[4096];
  strftime (buf,4096,"%Y-%m-%dT%H:%M:%S", thisTime);
  
  char tzone[7];
  strftime (tzone, 7, "%z", thisTime);
  if (strlen(tzone) == 5) {
    tzone[6] = 0;
    tzone[5] = tzone[4];
    tzone[4] = tzone[3];
    tzone[3] = ':';
  }
  sprintf (tmpstr, "%s%s", buf, tzone);

  Tau_metadata_register("Local Time", tmpstr);

   // write out the timestamp (number of microseconds since epoch (unsigned long long)
  sprintf (tmpstr, timeFormat, TauMetrics_getTimeOfDay());
  Tau_metadata_register("Timestamp", tmpstr);


#ifndef TAU_WINDOWS
  // try to grab meta-data
  char hostname[4096];
  gethostname(hostname,4096);
  Tau_metadata_register("Hostname", hostname);

  struct utsname archinfo;

  uname (&archinfo);
  Tau_metadata_register("OS Name", archinfo.sysname);
  Tau_metadata_register("OS Version", archinfo.version);
  Tau_metadata_register("OS Release", archinfo.release);
  Tau_metadata_register("OS Machine", archinfo.machine);
  Tau_metadata_register("Node Name", archinfo.nodename);

  Tau_metadata_register("TAU Architecture", TAU_ARCH);
  Tau_metadata_register("TAU Config", TAU_CONFIG);
  Tau_metadata_register("TAU Makefile", TAU_MAKEFILE);
  Tau_metadata_register("TAU Version", TAU_VERSION);

  Tau_metadata_register("pid", (int)getpid());
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
  Tau_metadata_register("BGL Coords", bglbuffer);

  Tau_metadata_register("BGL Processor ID", rts_get_processor_id());

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xSize(&personality),
	   BGLPersonality_ySize(&personality),
	   BGLPersonality_zSize(&personality));
  Tau_metadata_register("BGL Size", bglbuffer);


  if (BGLPersonality_virtualNodeMode(&personality)) {
    Tau_metadata_register("BGL Node Mode", "Virtual");
  } else {
    Tau_metadata_register("BGL Node Mode", "Coprocessor");
  }

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_isTorusX(&personality),
	   BGLPersonality_isTorusY(&personality),
	   BGLPersonality_isTorusZ(&personality));
  Tau_metadata_register("BGL isTorus", bglbuffer);

  Tau_metadata_register("BGL DDRSize", BGLPersonality_DDRSize(&personality));
  Tau_metadata_register("BGL DDRModuleType", personality.DDRModuleType);
  Tau_metadata_register("BGL Location", location);

  Tau_metadata_register("BGL rankInPset", BGLPersonality_rankInPset(&personality));
  Tau_metadata_register("BGL numNodesInPset", BGLPersonality_numNodesInPset(&personality));
  Tau_metadata_register("BGL psetNum", BGLPersonality_psetNum(&personality));
  Tau_metadata_register("BGL numPsets", BGLPersonality_numPsets(&personality));

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetSize(&personality),
	   BGLPersonality_yPsetSize(&personality),
	   BGLPersonality_zPsetSize(&personality));
  Tau_metadata_register("BGL PsetSize", bglbuffer);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetOrigin(&personality),
	   BGLPersonality_yPsetOrigin(&personality),
	   BGLPersonality_zPsetOrigin(&personality));
  Tau_metadata_register("BGL PsetOrigin", bglbuffer);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetCoord(&personality),
	   BGLPersonality_yPsetCoord(&personality),
	   BGLPersonality_zPsetCoord(&personality));
  Tau_metadata_register("BGL PsetCoord", bglbuffer);
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
  Tau_metadata_register("BGP Coords", bgpbuffer);

  Tau_metadata_register("BGP Processor ID", Kernel_PhysicalProcessorID());

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xSize(&personality),
	   BGP_Personality_ySize(&personality),
	   BGP_Personality_zSize(&personality));
  Tau_metadata_register("BGP Size", bgpbuffer);


  if (Kernel_ProcessCount() > 1) {
    Tau_metadata_register("BGP Node Mode", "Virtual");
  } else {
    sprintf(bgpbuffer, "Coprocessor (%d)", Kernel_ProcessCount());
    Tau_metadata_register("BGP Node Mode", bgpbuffer);
  }

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_isTorusX(&personality),
	   BGP_Personality_isTorusY(&personality),
	   BGP_Personality_isTorusZ(&personality));
  Tau_metadata_register("BGP isTorus", bgpbuffer);

  Tau_metadata_register("BGP DDRSize (MB)", BGP_Personality_DDRSizeMB(&personality));
/* CHECK: 
  Tau_metadata_register("BGP DDRModuleType", personality.DDRModuleType);
*/
  Tau_metadata_register("BGP Location", location);

  Tau_metadata_register("BGP rankInPset", BGP_Personality_rankInPset(&personality));
/*
  Tau_metadata_register("BGP numNodesInPset", Kernel_ProcessCount());
*/
  Tau_metadata_register("BGP psetSize", BGP_Personality_psetSize(&personality));
  Tau_metadata_register("BGP psetNum", BGP_Personality_psetNum(&personality));
  Tau_metadata_register("BGP numPsets", BGP_Personality_numComputeNodes(&personality));

/* CHECK: 
  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetSize(&personality),
	   BGP_Personality_yPsetSize(&personality),
	   BGP_Personality_zPsetSize(&personality));
  Tau_metadata_register("BGP PsetSize", bgpbuffer);

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetOrigin(&personality),
	   BGP_Personality_yPsetOrigin(&personality),
	   BGP_Personality_zPsetOrigin(&personality));
  Tau_metadata_register("BGP PsetOrigin", bgpbuffer);

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetCoord(&personality),
	   BGP_Personality_yPsetCoord(&personality),
	   BGP_Personality_zPsetCoord(&personality));
  Tau_metadata_register("BGP PsetCoord", bgpbuffer);
*/
#endif /* TAU_BGP */

#ifdef __linux__
  // doesn't work on ia64 for some reason
  //Tau_util_output (out, "\t<linux_tid>%d</linux_tid>\n", gettid());

  // try to grab CPU info
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (f) {
    char line[4096];
    while (Tau_util_readFullLine(line, f)) {
      char *value = strstr(line,":");
      if (!value) {
	break;
      } else {
	/* skip over colon */
	value += 2;
      }

      value = Tau_util_removeRuns(value);

      if (strncmp(line, "vendor_id", 9) == 0) {
	Tau_metadata_register("CPU Vendor", value);
      }
      if (strncmp(line, "vendor", 6) == 0) {
	Tau_metadata_register("CPU Vendor", value);
      }
      if (strncmp(line, "cpu MHz", 7) == 0) {
	Tau_metadata_register("CPU MHz", value);
      }
      if (strncmp(line, "clock", 5) == 0) {
	Tau_metadata_register("CPU MHz", value);
      }
      if (strncmp(line, "model name", 10) == 0) {
	Tau_metadata_register("CPU Type", value);
      }
      if (strncmp(line, "family", 6) == 0) {
	Tau_metadata_register("CPU Type", value);
      }
      if (strncmp(line, "cpu\t", 4) == 0) {
	Tau_metadata_register("CPU Type", value);
      }
      if (strncmp(line, "cache size", 10) == 0) {
	Tau_metadata_register("Cache Size", value);
      }
      if (strncmp(line, "cpu cores", 9) == 0) {
	Tau_metadata_register("CPU Cores", value);
      }
    }
    fclose(f);
  }

  f = fopen("/proc/meminfo", "r");
  if (f) {
    char line[4096];
    while (Tau_util_readFullLine(line, f)) {
      char *value = strstr(line,":");

      if (!value) {
	break;
      } else {
	value += 2;
      }

      value = Tau_util_removeRuns(value);

      if (strncmp(line, "MemTotal", 8) == 0) {
	Tau_metadata_register("Memory Size", value);
      }
    }
    fclose(f);
  }

  char buffer[4096];
  bzero(buffer, 4096);
  int rc = readlink("/proc/self/exe", buffer, 4096);
  if (rc != -1) {
    Tau_metadata_register("Executable", buffer);
  }
  bzero(buffer, 4096);
  rc = readlink("/proc/self/cwd", buffer, 4096);
  if (rc != -1) {
    Tau_metadata_register("CWD", buffer);
  }


  f = fopen("/proc/self/cmdline", "r");
  if (f) {
    char line[4096];

    std::ostringstream os;

    while (Tau_util_readFullLine(line, f)) {
      if (os.str().length() != 0) {
	os << " ";
      }
      os << line;
    }
    Tau_metadata_register("Command Line", os.str().c_str());
    fclose(f);
  }
#endif /* __linux__ */

  char *user = getenv("USER");
  if (user != NULL) {
    Tau_metadata_register("username", user);
  }

  return 0;
#endif

}


static int writeMetaData(Tau_util_outputDevice *out, bool newline, int counter) {
  const char *endl = "";
  if (newline) {
    endl = "\n";
  }

  Tau_util_output (out, "<metadata>%s", endl);

  if (counter != -1) {
    Tau_XML_writeAttribute(out, "Metric Name", RtsLayer::getCounterName(counter), newline);
  }


  // Write data from the Tau_metadata_register environment variable
  // char *tauMetaDataEnvVar = getenv("Tau_metadata_register");
  // if (tauMetaDataEnvVar != NULL) {
  //   if (strncmp(tauMetaDataEnvVar, "<attribute>", strlen("<attribute>")) != 0) {
  //     fprintf (stderr, "Error in formating TAU_METADATA environment variable\n");
  //   } else {
  //     Tau_util_output (out, tauMetaDataEnvVar);
  //   }
  // }


  // write out the user-specified (some from TAU) attributes
  for (map<string,string>::iterator it = Tau_metadata_getMetaData().begin(); it != Tau_metadata_getMetaData().end(); ++it) {
    const char *name = it->first.c_str();
    const char *value = it->second.c_str();
    Tau_XML_writeAttribute(out, name, value, newline);
  }

  Tau_util_output (out, "</metadata>%s", endl);
  return 0;

}





extern "C" void Tau_context_metadata(char *name, char *value) {

#ifdef TAU_DISABLE_METADATA
  return;
#endif

  // get the current calling context
  Profiler *current = TauInternal_CurrentProfiler(RtsLayer::getTid());
  FunctionInfo *fi = current->ThisFunction;
  const char *fname = fi->GetName();

  char *myName = (char*) malloc (strlen(name) + strlen(fname) + 10);
  sprintf (myName, "%s => %s", fname, name);
  char *myValue = strdup(value);
  RtsLayer::LockDB();
  Tau_metadata_getMetaData()[myName] = myValue;
  RtsLayer::UnLockDB();
}

extern "C" void Tau_phase_metadata(char *name, char *value) {

#ifdef TAU_DISABLE_METADATA
  return;
#endif

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
  Tau_metadata_getMetaData()[myName] = myValue;
  RtsLayer::UnLockDB();
  #else
  Tau_context_metadata(name, value);
  #endif
}


int Tau_metadata_writeMetaData(Tau_util_outputDevice *out) {

#ifdef TAU_DISABLE_METADATA
  return 0;
#endif

  Tau_metadata_fillMetaData();
  return writeMetaData(out, true, -1);
}

int Tau_metadata_writeMetaData(Tau_util_outputDevice *out, int counter) {
#ifdef TAU_DISABLE_METADATA
  return 0;
#endif

  Tau_metadata_fillMetaData();
  int retval;
  retval = writeMetaData(out, false, counter);
  return retval;
}

/* helper function to write to already established file pointer */
int Tau_metadata_writeMetaData(FILE *fp, int counter) {
  Tau_util_outputDevice out;
  out.fp = fp;
  out.type = TAU_UTIL_OUTPUT_FILE;
  return Tau_metadata_writeMetaData(&out, counter);
}




Tau_util_outputDevice *Tau_metadata_generateMergeBuffer() {
  Tau_util_outputDevice *out = Tau_util_createBufferOutputDevice();

  Tau_util_output(out,"%d%c", Tau_metadata_getMetaData().size(), '\0');

  for (map<string,string>::iterator it = Tau_metadata_getMetaData().begin(); it != Tau_metadata_getMetaData().end(); ++it) {
    const char *name = it->first.c_str();
    const char *value = it->second.c_str();
    Tau_util_output(out,"%s%c", name, '\0');
    Tau_util_output(out,"%s%c", value, '\0');
  }
  return out;
}


void Tau_metadata_removeDuplicates(char *buffer, int buflen) {
  // read the number of items and allocate arrays
  int numItems;
  sscanf(buffer,"%d", &numItems);
  buffer = strchr(buffer, '\0')+1;

  char **attributes = (char **) malloc(sizeof(char*) * numItems);
  char **values = (char **) malloc(sizeof(char*) * numItems);

  // assign char pointers to the values inside the buffer
  for (int i=0; i<numItems; i++) {
    const char *attribute = buffer;
    buffer = strchr(buffer, '\0')+1;
    const char *value = buffer;
    buffer = strchr(buffer, '\0')+1;

    map<string,string>::iterator iter = Tau_metadata_getMetaData().find(attribute);
    if (iter != Tau_metadata_getMetaData().end()) {
      const char *my_value = iter->second.c_str();
      if (0 == strcmp(value, my_value)) {
	Tau_metadata_getMetaData().erase(attribute);
      }
    }
  }
}


