/*****************************************************************************
 **			TAU Portable Profiling Package			    **
 **			http://www.cs.uoregon.edu/research/paracomp/tau     **
 *****************************************************************************
 **    Copyright 2005  						   	    **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 **    Research Center Juelich, Germany                                     **
 ****************************************************************************/
/*****************************************************************************
 **	File 		: TAU_tf_writer.cpp				    **
 **	Description 	: TAU trace format writer library C, C++ API	    **
 **	Author		: Alan Morris            			    **
 **	Contact		: amorris@cs.uoregon.edu 	                    **
 ****************************************************************************/


#include "TAU_tf.h"
#include "TAU_tf_headers.h"
#include <Profile/tau_types.h>


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */



  static void checkFlush(Ttf_fileT *tFile) {
    // Write the current trace buffer to file if necessary.

    if (tFile->tracePosition >= TAU_MAX_RECORDS) {
      Ttf_FlushTrace(tFile);
    }
  }

  static int flushEdf(Ttf_fileT *tFile) {
    FILE *fp;
    if ((fp = fopen (tFile->EdfFile, "wb")) == NULL) {
      perror("Error while flushing EDF file");
      return -1;
    }

    int numEvents = tFile->EventIdMap->size();

    fprintf(fp, "%d dynamic_trace_events\n", numEvents); 

    fprintf(fp,"# FunctionId Group Tag \"Name Type\" Parameters\n");


    for (EventIdMapT::iterator it = tFile->EventIdMap->begin(); it != tFile->EventIdMap->end(); ++it) {
      int id = (*it).first;
      Ttf_EventDescrT eventDesc = (*it).second;

      fprintf(fp, "%ld %s %ld \"%s\" %s\n", id, eventDesc.Group, eventDesc.Tag, eventDesc.EventName, eventDesc.Param);

    }

    fclose(fp);
    tFile->needsEdfFlush = false;
    return 0;
  }


  Ttf_FileHandleT Ttf_OpenFileForOutput( const char *name, const char *edf) {
    Ttf_fileT *tFile;
    FILE *fp;
    
    /* first, allocate space for the trace file id struct */
    tFile = new Ttf_fileT;
    if (tFile == NULL) {
      perror("ERROR: memory allocation failed for tFile");
      return NULL;
    }

    // Allocation the trace buffer
    tFile->traceBuffer = new EVENT[TAU_MAX_RECORDS];
    if (tFile->traceBuffer == NULL) {
      perror("ERROR: memory allocation failed for trace buffer");
      return NULL;
    }
    tFile->tracePosition = 1; // 0 will be the EV_INIT record
    tFile->initialized = false;
    tFile->forWriting = true;

    /* Open the trace file */
    if ((tFile->Fid = open (name, O_WRONLY|O_CREAT|O_TRUNC|O_APPEND|O_BINARY|LARGEFILE_OPTION, 0600)) < 0) {
      perror (name);
      return NULL;
    }

    
    if ((fp = fopen (edf, "wb")) == NULL) {
      printf("EDF file = %s\n", edf);
      perror("ERROR: opening edf file");
      return NULL;
    }
    fclose(fp);

    /* make a copy of the EDF file name */
    tFile->EdfFile = strdup(edf);

    
    tFile->NidTidMap = new NidTidMapT;
    if (tFile->NidTidMap == (NidTidMapT *) NULL) {
      perror("ERROR: memory allocation failed for NidTidMap");
      return NULL;
    }
    
    /* Allocate space for event id map */
    tFile->EventIdMap = new EventIdMapT;
    if (tFile->EventIdMap == (EventIdMapT *) NULL) {
      perror("ERROR: memory allocation failed for EventIdMap");
      return NULL;
    }
    
    /* Allocate space for group id map */
    tFile->GroupIdMap = new GroupIdMapT;
    if (tFile->GroupIdMap == (GroupIdMapT *) NULL) {
      perror("ERROR: memory allocation failed for GroupIdMap");
      return NULL;
    }


    tFile->groupNameMap = new GroupNameMapT;
    if (tFile->groupNameMap == NULL) {
      perror("ERROR: memory allocation failed for GroupNameMap");
      return NULL;
    }


    tFile->needsEdfFlush = true;

    /* initialize clock */
    tFile->ClkInitialized = FALSE;
    
    /* initialize the first timestamp for the trace */
    tFile->FirstTimestamp = 0.0;


    /* define some events */


    Ttf_EventDescrT newEventDesc;

    newEventDesc.Eid = TAU_EV_INIT;
    newEventDesc.Group = "TRACER";
    newEventDesc.EventName = "EV_INIT";
    newEventDesc.Tag = 0;
    newEventDesc.Param = "none";
    (*tFile->EventIdMap)[TAU_EV_INIT] = newEventDesc;

    newEventDesc.Eid = TAU_EV_CLOSE;
    newEventDesc.Group = "TRACER";
    newEventDesc.EventName = "FLUSH_CLOSE";
    newEventDesc.Tag = 0;
    newEventDesc.Param = "none";
    (*tFile->EventIdMap)[TAU_EV_CLOSE] = newEventDesc;


    newEventDesc.Eid = TAU_EV_WALL_CLOCK;
    newEventDesc.Group = "TRACER";
    newEventDesc.EventName = "WALL_CLOCK";
    newEventDesc.Tag = 0;
    newEventDesc.Param = "none";
    (*tFile->EventIdMap)[TAU_EV_WALL_CLOCK] = newEventDesc;


    newEventDesc.Eid = TAU_MESSAGE_SEND;
    newEventDesc.Group = "TAU_MESSAGE";
    newEventDesc.EventName = "MESSAGE_SEND";
    newEventDesc.Tag = -7;
    newEventDesc.Param = "par";
    (*tFile->EventIdMap)[TAU_MESSAGE_SEND] = newEventDesc;


    newEventDesc.Eid = TAU_MESSAGE_RECV;
    newEventDesc.Group = "TAU_MESSAGE";
    newEventDesc.EventName = "MESSAGE_RECV";
    newEventDesc.Tag = -8;
    newEventDesc.Param = "par";
    (*tFile->EventIdMap)[TAU_MESSAGE_RECV] = newEventDesc;

    
    /* return file handle */
    return (Ttf_FileHandleT) tFile;
  }








  static int checkInitialized(Ttf_FileHandleT file, unsigned int nodeToken, unsigned int threadToken, double time) {
    // Adds the initialization record
    Ttf_fileT *tFile = (Ttf_fileT*)file;
    if (!tFile->initialized) {
      int pos = 0;
      tFile->traceBuffer[pos].ev = TAU_EV_INIT;
      tFile->traceBuffer[pos].nid = nodeToken;
      tFile->traceBuffer[pos].tid = threadToken;
      tFile->traceBuffer[pos].ti = (x_uint64) time;
      tFile->traceBuffer[pos].par = 3;
      tFile->initialized = true;
    }
    return 0;
  }
  

  int Ttf_DefThread(Ttf_FileHandleT file, unsigned int nodeToken, unsigned int threadToken, 
		    const char *threadName) {

    Ttf_fileT *tFile = (Ttf_fileT*)file;

    int nid = nodeToken;
    int tid = threadToken;


    NidTidMapT::iterator nit = tFile->NidTidMap->find(pair<int, int>(nid,tid));
    if (nit == tFile->NidTidMap->end()) {
      /* this pair of node and thread has not been encountered before */
      /* add it to the map! */
      (*(tFile->NidTidMap))[pair<int,int>(nid,tid)] = 1;
    }

    return 0;
  }



  // returns stateGroupToken
  int Ttf_DefStateGroup(Ttf_FileHandleT file, const char *stateGroupName, unsigned int stateGroupToken) {
    Ttf_fileT *tFile = (Ttf_fileT*)file;

    (*tFile->groupNameMap)[stateGroupToken] = strdup(stateGroupName);
    return 0;
  }




  // returns stateToken
  int Ttf_DefState(Ttf_FileHandleT file, unsigned int stateToken, const char *stateName, unsigned int stateGroupToken) {
    Ttf_fileT *tFile = (Ttf_fileT*)file;


    Ttf_EventDescrT newEventDesc;

    GroupNameMapT::iterator git = tFile->groupNameMap->find(stateGroupToken);
    if (git == tFile->groupNameMap->end()) { 
      fprintf (stderr, "Ttf_DefState: Have not seen %d stateGroupToken before, please define it first\n", stateGroupToken);
    }

    newEventDesc.Eid = stateToken;
    newEventDesc.Group = const_cast<char*>((*tFile->groupNameMap)[stateGroupToken]);
    newEventDesc.EventName = strdup(stateName);
    newEventDesc.Tag = 0;
    newEventDesc.Param = "EntryExit";

    (*tFile->EventIdMap)[stateToken] = newEventDesc;

    tFile->needsEdfFlush = true;

    return 0;
  }

  

  int Ttf_FlushTrace(Ttf_FileHandleT file) {
    Ttf_fileT *tFile = (Ttf_fileT*)file;
    
    checkInitialized(file, tFile->traceBuffer[1].nid, 
		     tFile->traceBuffer[1].tid, tFile->traceBuffer[1].ti);

    // compute size of write
    int size = tFile->tracePosition * sizeof(EVENT);
    
    // reset trace position
    tFile->tracePosition = 0;

    // must write out edf file first
    if (tFile->needsEdfFlush) {
      if (flushEdf(tFile) != 0) {
	return -1;
      }
    }
    
    //printf ("flushing %d bytes\n", size);
    int ret = write (tFile->Fid, tFile->traceBuffer, size);
    if (ret != size) {
      perror("Error flushing trace buffer");
      return -1;
    }
    return 0;
  }


  int Ttf_CloseOutputFile(Ttf_FileHandleT file) {
    Ttf_fileT *tFile = (Ttf_fileT*)file;

    if (tFile->forWriting == false) {
      return (int)((long)Ttf_CloseFile(tFile));
    }

    for (NidTidMapT::iterator it = tFile->NidTidMap->begin(); it != tFile->NidTidMap->end(); ++it) {
      pair<int, int> nidtid = (*it).first;
      

      checkFlush(tFile);
      int pos = tFile->tracePosition;
      tFile->traceBuffer[pos].ev = TAU_EV_CLOSE;
      tFile->traceBuffer[pos].nid = nidtid.first;
      tFile->traceBuffer[pos].tid = nidtid.second;
      tFile->traceBuffer[pos].ti = (x_uint64) tFile->lastTimestamp;
      tFile->traceBuffer[pos].par = 0;
      tFile->tracePosition++;
      
      pos = tFile->tracePosition;
      tFile->traceBuffer[pos].ev = TAU_EV_WALL_CLOCK;
      tFile->traceBuffer[pos].nid = nidtid.first;
      tFile->traceBuffer[pos].tid = nidtid.second;
      tFile->traceBuffer[pos].ti = (x_uint64) tFile->lastTimestamp;
      tFile->traceBuffer[pos].par = 0;
      tFile->tracePosition++;
    

    }
    Ttf_FlushTrace(file);
    close(tFile->Fid);
    return 0;
  }




  static int enterExit(Ttf_FileHandleT file, x_uint64 time, 
		     unsigned int nodeToken, unsigned int threadToken, 
		       unsigned int stateToken, int parameter) {
    Ttf_fileT *tFile = (Ttf_fileT*)file;

    checkFlush(tFile);
    int pos = tFile->tracePosition;

    tFile->traceBuffer[pos].ev = stateToken;
    tFile->traceBuffer[pos].nid = nodeToken;
    tFile->traceBuffer[pos].tid = threadToken;
    tFile->traceBuffer[pos].ti = time;
    tFile->traceBuffer[pos].par = parameter;
    tFile->tracePosition++;
    tFile->lastTimestamp = time;
    return 0;
  }

  int Ttf_EnterState(Ttf_FileHandleT file, x_uint64 time, 
		     unsigned int nodeToken, unsigned int threadToken, 
		     unsigned int stateToken) {
	return enterExit(file, time, nodeToken, threadToken, stateToken, 1); // entry
  }

  int Ttf_LeaveState(Ttf_FileHandleT file, x_uint64 time, 
		     unsigned int nodeToken, unsigned int threadToken, unsigned int stateToken) {
    return enterExit(file, time, nodeToken, threadToken, stateToken, -1); // exit
  }

  int Ttf_DefClkPeriod(Ttf_FileHandleT file, double clkPeriod) {
    return 0;
  }



  static int sendRecv(Ttf_FileHandleT file, double time, unsigned int sourceNodeToken,
		      unsigned int sourceThreadToken,
		      unsigned int destinationNodeToken,
		      unsigned int destinationThreadToken,
		      unsigned int messageSize,
		      unsigned int messageTag,
		      unsigned int messageComm, int eventId) {
    Ttf_fileT *tFile = (Ttf_fileT*)file;


    x_int64 parameter;
    x_uint64 xother, xtype, xlength, xcomm;
    
    xother = destinationNodeToken;
    xtype = messageTag;
    xlength = messageSize;
    xcomm = messageComm;

    parameter = (xlength >> 16 << 54 >> 22) |
      ((xtype >> 8 & 0xFF) << 48) |
      ((xother >> 8 & 0xFF) << 56) |
      (xlength & 0xFFFF) | 
      ((xtype & 0xFF)  << 16) | 
      ((xother & 0xFF) << 24) |
      (xcomm << 58 >> 16);


    checkFlush(tFile);
    int pos = tFile->tracePosition;
    tFile->traceBuffer[pos].ev = eventId;
    tFile->traceBuffer[pos].nid = sourceNodeToken;
    tFile->traceBuffer[pos].tid = sourceThreadToken;
    tFile->traceBuffer[pos].ti = (x_uint64)time;
    tFile->traceBuffer[pos].par = parameter;
    tFile->tracePosition++;
    tFile->lastTimestamp = time;

    return 0;
  }

  int Ttf_SendMessage(Ttf_FileHandleT file, double time, unsigned int sourceNodeToken,
		      unsigned int sourceThreadToken,
		      unsigned int destinationNodeToken,
		      unsigned int destinationThreadToken,
		      unsigned int messageSize,
		      unsigned int messageTag,
		      unsigned int messageComm) {
    return sendRecv(file, time, sourceNodeToken, sourceThreadToken, destinationNodeToken, 
		    destinationThreadToken, messageSize, messageTag, messageComm, TAU_MESSAGE_SEND);

  }

  int Ttf_RecvMessage(Ttf_FileHandleT file, double time, unsigned int sourceNodeToken,
		      unsigned int sourceThreadToken,
		      unsigned int destinationNodeToken,
		      unsigned int destinationThreadToken,
		      unsigned int messageSize,
		      unsigned int messageTag,
		      unsigned int messageComm) {
    return sendRecv(file, time, destinationNodeToken, 
		    destinationThreadToken, sourceNodeToken, sourceThreadToken, 
		    messageSize, messageTag, messageComm, TAU_MESSAGE_RECV);

  }



  int Ttf_DefUserEvent(Ttf_FileHandleT file, unsigned int userEventToken, 
			  const char *userEventName, int monotonicallyIncreasing) {
    Ttf_fileT *tFile = (Ttf_fileT*)file;


    Ttf_EventDescrT newEventDesc;
    newEventDesc.Eid = userEventToken;
    newEventDesc.Group = "TAUEVENT";
    newEventDesc.EventName = strdup(userEventName);
    newEventDesc.Tag = monotonicallyIncreasing;
    newEventDesc.Param = "TriggerValue";

    (*tFile->EventIdMap)[userEventToken] = newEventDesc;

    tFile->needsEdfFlush = true;
    return 0;
  }

  int Ttf_EventTrigger(Ttf_FileHandleT file, double time, 
			   unsigned int nodeToken,
			   unsigned int threadToken,
			   unsigned int userEventToken,
			   double userEventValue
			   ) {
    Ttf_fileT *tFile = (Ttf_fileT*)file;
    
    
    checkFlush(tFile);
    int pos = tFile->tracePosition;
    
    tFile->traceBuffer[pos].ev = userEventToken;
    tFile->traceBuffer[pos].nid = nodeToken;
    tFile->traceBuffer[pos].tid = threadToken;
    tFile->traceBuffer[pos].ti = (x_uint64) time;
    // currently casting to x_uint64
    tFile->traceBuffer[pos].par = (x_uint64) userEventValue;
    tFile->tracePosition++;
    tFile->lastTimestamp = time;
    return 0;
  }

  /*
	This is a helper function to write out user defined events to the trace file. 
	Trace writer APIs can not be directly used here as we need to use the entire 
	bytes of the parameter. 
*/

int  Ttf_LongEventTrigger(Ttf_FileHandleT file,  unsigned long long time, 
				unsigned int nodeToken, 
				unsigned int threadToken,  
				unsigned int userEventToken, 
				 unsigned long long userEventValue)
{
	Ttf_fileT *tFile = (Ttf_fileT*)file;
	checkFlush(tFile);
    	int pos = tFile->tracePosition;		
    	tFile->traceBuffer[pos].ev = userEventToken;
    	tFile->traceBuffer[pos].nid = nodeToken;
    	tFile->traceBuffer[pos].tid = threadToken;
    	tFile->traceBuffer[pos].ti = (x_uint64)time;
    	tFile->traceBuffer[pos].par = userEventValue;
    	tFile->tracePosition++;
    	tFile->lastTimestamp = time;
    	return 0;
}
  
  

#ifdef __cplusplus
}
#endif /* __cplusplus */
