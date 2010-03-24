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
static map<string,string> &TheMetaData() {
  static MetaDataRepo metadata;
  return metadata;
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

  char tmpstr[1024];
#ifdef TAU_WINDOWS
  sprintf (tmpstr, "%I64d", TauMetrics_getInitialTimeStamp());
#else
  sprintf (tmpstr, "%lld", TauMetrics_getInitialTimeStamp());
#endif
  Tau_XML_writeAttribute(out, "Starting Timestamp", tmpstr, newline);

  Tau_XML_writeTime(out, newline);

#ifndef TAU_WINDOWS
  // try to grab meta-data
  char hostname[4096];
  gethostname(hostname,4096);
  Tau_XML_writeAttribute(out, "Hostname", hostname, newline);

  struct utsname archinfo;

  uname (&archinfo);
  Tau_XML_writeAttribute(out, "OS Name", archinfo.sysname, newline);
  Tau_XML_writeAttribute(out, "OS Version", archinfo.version, newline);
  Tau_XML_writeAttribute(out, "OS Release", archinfo.release, newline);
  Tau_XML_writeAttribute(out, "OS Machine", archinfo.machine, newline);
  Tau_XML_writeAttribute(out, "Node Name", archinfo.nodename, newline);

  Tau_XML_writeAttribute(out, "TAU Architecture", TAU_ARCH, newline);
  Tau_XML_writeAttribute(out, "TAU Config", TAU_CONFIG, newline);
  Tau_XML_writeAttribute(out, "TAU Makefile", TAU_MAKEFILE, newline);
  Tau_XML_writeAttribute(out, "TAU Version", TAU_VERSION, newline);

  Tau_XML_writeAttribute(out, "pid", getpid(), newline);
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
  Tau_XML_writeAttribute(out, "BGL Coords", bglbuffer, newline);

  Tau_XML_writeAttribute(out, "BGL Processor ID", rts_get_processor_id(), newline);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xSize(&personality),
	   BGLPersonality_ySize(&personality),
	   BGLPersonality_zSize(&personality));
  Tau_XML_writeAttribute(out, "BGL Size", bglbuffer, newline);


  if (BGLPersonality_virtualNodeMode(&personality)) {
    Tau_XML_writeAttribute(out, "BGL Node Mode", "Virtual", newline);
  } else {
    Tau_XML_writeAttribute(out, "BGL Node Mode", "Coprocessor", newline);
  }

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_isTorusX(&personality),
	   BGLPersonality_isTorusY(&personality),
	   BGLPersonality_isTorusZ(&personality));
  Tau_XML_writeAttribute(out, "BGL isTorus", bglbuffer, newline);

  Tau_XML_writeAttribute(out, "BGL DDRSize", BGLPersonality_DDRSize(&personality), newline);
  Tau_XML_writeAttribute(out, "BGL DDRModuleType", personality.DDRModuleType, newline);
  Tau_XML_writeAttribute(out, "BGL Location", location, newline);

  Tau_XML_writeAttribute(out, "BGL rankInPset", BGLPersonality_rankInPset(&personality), newline);
  Tau_XML_writeAttribute(out, "BGL numNodesInPset", BGLPersonality_numNodesInPset(&personality), newline);
  Tau_XML_writeAttribute(out, "BGL psetNum", BGLPersonality_psetNum(&personality), newline);
  Tau_XML_writeAttribute(out, "BGL numPsets", BGLPersonality_numPsets(&personality), newline);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetSize(&personality),
	   BGLPersonality_yPsetSize(&personality),
	   BGLPersonality_zPsetSize(&personality));
  Tau_XML_writeAttribute(out, "BGL PsetSize", bglbuffer, newline);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetOrigin(&personality),
	   BGLPersonality_yPsetOrigin(&personality),
	   BGLPersonality_zPsetOrigin(&personality));
  Tau_XML_writeAttribute(out, "BGL PsetOrigin", bglbuffer, newline);

  sprintf (bglbuffer, "(%d,%d,%d)", BGLPersonality_xPsetCoord(&personality),
	   BGLPersonality_yPsetCoord(&personality),
	   BGLPersonality_zPsetCoord(&personality));
  Tau_XML_writeAttribute(out, "BGL PsetCoord", bglbuffer, newline);
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
  Tau_XML_writeAttribute(out, "BGP Coords", bgpbuffer, newline);

  Tau_XML_writeAttribute(out, "BGP Processor ID", Kernel_PhysicalProcessorID(), newline);

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xSize(&personality),
	   BGP_Personality_ySize(&personality),
	   BGP_Personality_zSize(&personality));
  Tau_XML_writeAttribute(out, "BGP Size", bgpbuffer, newline);


  if (Kernel_ProcessCount() > 1) {
    Tau_XML_writeAttribute(out, "BGP Node Mode", "Virtual", newline);
  } else {
    sprintf(bgpbuffer, "Coprocessor (%d)", Kernel_ProcessCount);
    Tau_XML_writeAttribute(out, "BGP Node Mode", bgpbuffer, newline);
  }

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_isTorusX(&personality),
	   BGP_Personality_isTorusY(&personality),
	   BGP_Personality_isTorusZ(&personality));
  Tau_XML_writeAttribute(out, "BGP isTorus", bgpbuffer, newline);

  Tau_XML_writeAttribute(out, "BGP DDRSize (MB)", BGP_Personality_DDRSizeMB(&personality), newline);
/* CHECK: 
  Tau_XML_writeAttribute(out, "BGP DDRModuleType", personality.DDRModuleType, newline);
*/
  Tau_XML_writeAttribute(out, "BGP Location", location, newline);

  Tau_XML_writeAttribute(out, "BGP rankInPset", BGP_Personality_rankInPset(&personality), newline);
/*
  Tau_XML_writeAttribute(out, "BGP numNodesInPset", Kernel_ProcessCount(), newline);
*/
  Tau_XML_writeAttribute(out, "BGP psetSize", BGP_Personality_psetSize(&personality), newline);
  Tau_XML_writeAttribute(out, "BGP psetNum", BGP_Personality_psetNum(&personality), newline);
  Tau_XML_writeAttribute(out, "BGP numPsets", BGP_Personality_numComputeNodes(&personality), newline);

/* CHECK: 
  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetSize(&personality),
	   BGP_Personality_yPsetSize(&personality),
	   BGP_Personality_zPsetSize(&personality));
  Tau_XML_writeAttribute(out, "BGP PsetSize", bgpbuffer, newline);

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetOrigin(&personality),
	   BGP_Personality_yPsetOrigin(&personality),
	   BGP_Personality_zPsetOrigin(&personality));
  Tau_XML_writeAttribute(out, "BGP PsetOrigin", bgpbuffer, newline);

  sprintf (bgpbuffer, "(%d,%d,%d)", BGP_Personality_xPsetCoord(&personality),
	   BGP_Personality_yPsetCoord(&personality),
	   BGP_Personality_zPsetCoord(&personality));
  Tau_XML_writeAttribute(out, "BGP PsetCoord", bgpbuffer, newline);
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
	Tau_XML_writeAttribute(out, "CPU Vendor", value, newline);
      }
      if (strncmp(line, "vendor", 6) == 0) {
	Tau_XML_writeAttribute(out, "CPU Vendor", value, newline);
      }
      if (strncmp(line, "cpu MHz", 7) == 0) {
	Tau_XML_writeAttribute(out, "CPU MHz", value, newline);
      }
      if (strncmp(line, "clock", 5) == 0) {
	Tau_XML_writeAttribute(out, "CPU MHz", value, newline);
      }
      if (strncmp(line, "model name", 10) == 0) {
	Tau_XML_writeAttribute(out, "CPU Type", value, newline);
      }
      if (strncmp(line, "family", 6) == 0) {
	Tau_XML_writeAttribute(out, "CPU Type", value, newline);
      }
      if (strncmp(line, "cpu\t", 4) == 0) {
	Tau_XML_writeAttribute(out, "CPU Type", value, newline);
      }
      if (strncmp(line, "cache size", 10) == 0) {
	Tau_XML_writeAttribute(out, "Cache Size", value, newline);
      }
      if (strncmp(line, "cpu cores", 9) == 0) {
	Tau_XML_writeAttribute(out, "CPU Cores", value, newline);
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
	Tau_XML_writeAttribute(out, "Memory Size", value, newline);
      }
    }
    fclose(f);
  }

  char buffer[4096];
  bzero(buffer, 4096);
  int rc = readlink("/proc/self/exe", buffer, 4096);
  if (rc != -1) {
    Tau_XML_writeAttribute(out, "Executable", buffer, newline);
  }
  bzero(buffer, 4096);
  rc = readlink("/proc/self/cwd", buffer, 4096);
  if (rc != -1) {
    Tau_XML_writeAttribute(out, "CWD", buffer, newline);
  }
  bzero(buffer, 4096);
  rc = readlink("/proc/self/cmdline", buffer, 4096);
  if (rc != -1) {
    Tau_XML_writeAttribute(out, "Command Line", buffer, newline);
  }
#endif /* __linux__ */

  char *user = getenv("USER");
  if (user != NULL) {
    Tau_XML_writeAttribute(out, "username", user, newline);
  }


  // Write data from the TAU_METADATA environment variable
  char *tauMetaDataEnvVar = getenv("TAU_METADATA");
  if (tauMetaDataEnvVar != NULL) {
    if (strncmp(tauMetaDataEnvVar, "<attribute>", strlen("<attribute>")) != 0) {
      fprintf (stderr, "Error in formating TAU_METADATA environment variable\n");
    } else {
      Tau_util_output (out, tauMetaDataEnvVar);
    }
  }

  // write out the user-specified (some from TAU) attributes
  for (map<string,string>::iterator it = TheMetaData().begin(); it != TheMetaData().end(); ++it) {
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


int Tau_metadata_writeMetaData(Tau_util_outputDevice *out) {
  return writeMetaData(out, true, -1);
}

int Tau_metadata_writeMetaData(Tau_util_outputDevice *out, int counter) {
#ifdef TAU_DISABLE_METADATA
  return 0;
#endif
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


