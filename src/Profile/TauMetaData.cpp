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
// These come from TAU_METADATA calls
map<string,string> &Tau_metadata_getMetaData() {
  static MetaDataRepo metadata;
  return metadata;
}


int Tau_metadata_fillMetaData() {
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
  TAU_METADATA("Starting Timestamp", tmpstr);



  time_t theTime = time(NULL);
  struct tm *thisTime = gmtime(&theTime);
  strftime (tmpstr,4096,"%Y-%m-%dT%H:%M:%SZ", thisTime);
  TAU_METADATA("UTC Time", tmpstr);


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

  TAU_METADATA("Local Time", tmpstr);

   // write out the timestamp (number of microseconds since epoch (unsigned long long)
  sprintf (tmpstr, timeFormat, TauMetrics_getTimeOfDay());
  TAU_METADATA("Timestamp", tmpstr);


#ifndef TAU_WINDOWS
  // try to grab meta-data
  char hostname[4096];
  gethostname(hostname,4096);
  TAU_METADATA("Hostname", hostname);

  struct utsname archinfo;

  uname (&archinfo);
  TAU_METADATA("OS Name", archinfo.sysname);
  TAU_METADATA("OS Version", archinfo.version);
  TAU_METADATA("OS Release", archinfo.release);
  TAU_METADATA("OS Machine", archinfo.machine);
  TAU_METADATA("Node Name", archinfo.nodename);

  TAU_METADATA("TAU Architecture", TAU_ARCH);
  TAU_METADATA("TAU Config", TAU_CONFIG);
  TAU_METADATA("TAU Makefile", TAU_MAKEFILE);
  TAU_METADATA("TAU Version", TAU_VERSION);

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
  TAU_METADATA("BGL Coords", bglbuffer);

  TAU_METADATA("BGL Processor ID", rts_get_processor_id());

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xSize(&personality),
	   BGLPersonality_ySize(&personality),
	   BGLPersonality_zSize(&personality));
  TAU_METADATA("BGL Size", bglbuffer);


  if (BGLPersonality_virtualNodeMode(&personality)) {
    TAU_METADATA("BGL Node Mode", "Virtual");
  } else {
    TAU_METADATA("BGL Node Mode", "Coprocessor");
  }

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_isTorusX(&personality),
	   BGLPersonality_isTorusY(&personality),
	   BGLPersonality_isTorusZ(&personality));
  TAU_METADATA("BGL isTorus", bglbuffer);

  TAU_METADATA("BGL DDRSize", BGLPersonality_DDRSize(&personality));
  TAU_METADATA("BGL DDRModuleType", personality.DDRModuleType);
  TAU_METADATA("BGL Location", location);

  TAU_METADATA("BGL rankInPset", BGLPersonality_rankInPset(&personality));
  TAU_METADATA("BGL numNodesInPset", BGLPersonality_numNodesInPset(&personality));
  TAU_METADATA("BGL psetNum", BGLPersonality_psetNum(&personality));
  TAU_METADATA("BGL numPsets", BGLPersonality_numPsets(&personality));

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetSize(&personality),
	   BGLPersonality_yPsetSize(&personality),
	   BGLPersonality_zPsetSize(&personality));
  TAU_METADATA("BGL PsetSize", bglbuffer);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetOrigin(&personality),
	   BGLPersonality_yPsetOrigin(&personality),
	   BGLPersonality_zPsetOrigin(&personality));
  TAU_METADATA("BGL PsetOrigin", bglbuffer);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetCoord(&personality),
	   BGLPersonality_yPsetCoord(&personality),
	   BGLPersonality_zPsetCoord(&personality));
  TAU_METADATA("BGL PsetCoord", bglbuffer);
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
  TAU_METADATA("BGP Coords", bgpbuffer);

  TAU_METADATA("BGP Processor ID", Kernel_PhysicalProcessorID());

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xSize(&personality),
	   BGP_Personality_ySize(&personality),
	   BGP_Personality_zSize(&personality));
  TAU_METADATA("BGP Size", bgpbuffer);


  if (Kernel_ProcessCount() > 1) {
    TAU_METADATA("BGP Node Mode", "Virtual");
  } else {
    sprintf(bgpbuffer, "Coprocessor (%d)", Kernel_ProcessCount);
    TAU_METADATA("BGP Node Mode", bgpbuffer);
  }

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_isTorusX(&personality),
	   BGP_Personality_isTorusY(&personality),
	   BGP_Personality_isTorusZ(&personality));
  TAU_METADATA("BGP isTorus", bgpbuffer);

  TAU_METADATA("BGP DDRSize (MB)", BGP_Personality_DDRSizeMB(&personality));
/* CHECK: 
  TAU_METADATA("BGP DDRModuleType", personality.DDRModuleType);
*/
  TAU_METADATA("BGP Location", location);

  TAU_METADATA("BGP rankInPset", BGP_Personality_rankInPset(&personality));
/*
  TAU_METADATA("BGP numNodesInPset", Kernel_ProcessCount());
*/
  TAU_METADATA("BGP psetSize", BGP_Personality_psetSize(&personality));
  TAU_METADATA("BGP psetNum", BGP_Personality_psetNum(&personality));
  TAU_METADATA("BGP numPsets", BGP_Personality_numComputeNodes(&personality));

/* CHECK: 
  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetSize(&personality),
	   BGP_Personality_yPsetSize(&personality),
	   BGP_Personality_zPsetSize(&personality));
  TAU_METADATA("BGP PsetSize", bgpbuffer);

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetOrigin(&personality),
	   BGP_Personality_yPsetOrigin(&personality),
	   BGP_Personality_zPsetOrigin(&personality));
  TAU_METADATA("BGP PsetOrigin", bgpbuffer);

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetCoord(&personality),
	   BGP_Personality_yPsetCoord(&personality),
	   BGP_Personality_zPsetCoord(&personality));
  TAU_METADATA("BGP PsetCoord", bgpbuffer);
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
	TAU_METADATA("CPU Vendor", value);
      }
      if (strncmp(line, "vendor", 6) == 0) {
	TAU_METADATA("CPU Vendor", value);
      }
      if (strncmp(line, "cpu MHz", 7) == 0) {
	TAU_METADATA("CPU MHz", value);
      }
      if (strncmp(line, "clock", 5) == 0) {
	TAU_METADATA("CPU MHz", value);
      }
      if (strncmp(line, "model name", 10) == 0) {
	TAU_METADATA("CPU Type", value);
      }
      if (strncmp(line, "family", 6) == 0) {
	TAU_METADATA("CPU Type", value);
      }
      if (strncmp(line, "cpu\t", 4) == 0) {
	TAU_METADATA("CPU Type", value);
      }
      if (strncmp(line, "cache size", 10) == 0) {
	TAU_METADATA("Cache Size", value);
      }
      if (strncmp(line, "cpu cores", 9) == 0) {
	TAU_METADATA("CPU Cores", value);
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
	TAU_METADATA("Memory Size", value);
      }
    }
    fclose(f);
  }

  char buffer[4096];
  bzero(buffer, 4096);
  int rc = readlink("/proc/self/exe", buffer, 4096);
  if (rc != -1) {
    TAU_METADATA("Executable", buffer);
  }
  bzero(buffer, 4096);
  rc = readlink("/proc/self/cwd", buffer, 4096);
  if (rc != -1) {
    TAU_METADATA("CWD", buffer);
  }
  bzero(buffer, 4096);
  rc = readlink("/proc/self/cmdline", buffer, 4096);
  if (rc != -1) {
    TAU_METADATA("Command Line", buffer);
  }
#endif /* __linux__ */

  char *user = getenv("USER");
  if (user != NULL) {
    TAU_METADATA("username", user);
  }

  return 0;

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


  // Write data from the TAU_METADATA environment variable
  // char *tauMetaDataEnvVar = getenv("TAU_METADATA");
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



extern "C" void Tau_metadata(char *name, char *value) {
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


extern "C" void Tau_context_metadata(char *name, char *value) {
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


