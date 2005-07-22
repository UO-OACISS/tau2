/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2003  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Research Center Juelich, Germany                                     **
****************************************************************************/
/***************************************************************************
**	File 		: TAU_tf.cpp					  **
**	Description 	: TAU trace format reader library		  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu 	                  **
***************************************************************************/
#include <TAU_tf.h>
#include <TAU_tf_headers.h>

/* Defines */
#define TAU_MESSAGE_SEND_EVENT -7
#define TAU_MESSAGE_RECV_EVENT -8

/* Open the trace file and return a pointer to the Ttf_fileT struct that
 * contains the file id and the maps */

Ttf_FileHandleT Ttf_OpenFileForInput( const char *filename, const char *EDF)
{
  Ttf_fileT *tFile;
  FILE *fp;

  /* first, allocate space for the trace file id struct */
  tFile = new Ttf_fileT;
  if (tFile == (Ttf_file *) NULL)
  {
    perror("ERROR: memory allocation failed for tFile");
    return NULL;
  }

  /* Open the trace file */
  if ( (tFile->Fid = open (filename, O_RDONLY | O_BINARY | LARGEFILE_OPTION)) < 0 )
  {
    perror (filename);
    return NULL;
  }

  /* Open the EDF (event description file) to read the <name, id> event 
   * tables*/
  if ((fp = fopen (EDF, "rb")) == NULL)
  {
    printf("EDF file = %s\n", EDF);
    perror("ERROR: opening edf file");
    return NULL;
  }
  /* check that the file is OK. close it for now. It will be re-read when 
   * the event map is filled */
  fclose(fp); 
 
  /* make a copy of the EDF file name */
  tFile->EdfFile = strdup(EDF);

  /* Allocate space for nodeid, thread id map */
  tFile->NidTidMap = new NidTidMapT;
  if (tFile->NidTidMap == (NidTidMapT *) NULL)
  {
    perror("ERROR: memory allocation failed for NidTidMap");
    return NULL;
  }

  /* Allocate space for event id map */
  tFile->EventIdMap = new EventIdMapT;
  if (tFile->EventIdMap == (EventIdMapT *) NULL)
  {
    perror("ERROR: memory allocation failed for EventIdMap");
    return NULL;
  }

  /* Allocate space for group id map */
  tFile->GroupIdMap = new GroupIdMapT;
  if (tFile->GroupIdMap == (GroupIdMapT *) NULL)
  {
    perror("ERROR: memory allocation failed for GroupIdMap");
    return NULL;
  }
  /* initialize clock */
  tFile->ClkInitialized = FALSE;

  /* initialize the first timestamp for the trace */
  tFile->FirstTimestamp = 0.0;

  /* return file handle */
  return (Ttf_FileHandleT) tFile;
}

/* close a trace file */
Ttf_FileHandleT Ttf_CloseFile( Ttf_FileHandleT fileHandle )
{
  Ttf_fileT *tFile; 

  tFile = (Ttf_fileT *) fileHandle;


  if (tFile == (Ttf_file *) NULL)
    return NULL;

  /* Close the trace file using the handle */
  if (close(tFile->Fid) < 0)
  {
    perror("ERROR: closing trace file");
    return NULL;
  }

  /* delete the maps and free the memory */
  delete tFile->NidTidMap;
  delete tFile->EventIdMap;
  delete tFile->GroupIdMap;

  /* return old file handle */
  return fileHandle; 
}


/* Seek to an absolute event position. 
 *    A negative position indicates to start from the tail of the event stream. 
 *       Returns the position */
int  Ttf_AbsSeek( Ttf_FileHandleT handle, int eventPosition )
{
  Ttf_fileT *tFile = (Ttf_fileT *) handle;
  off_t position;
  if (eventPosition > 0)
  { /* start from the top, to the absolute position */
    position = lseek(tFile->Fid, eventPosition*sizeof(PCXX_EV), SEEK_SET);
    if (position)
    { /* success */
      return position/sizeof(PCXX_EV);
    } 
    else
    { /* failure */
      return 0;
    }
  } 
  else 
  { /* start from the tail of the event stream */
    position = lseek(tFile->Fid, eventPosition*sizeof(PCXX_EV), SEEK_END);
    if (position)
    {
      /* success, return the position */
      return position/sizeof(PCXX_EV);
    }
    else
    {
      /* failure */
      return 0;
    }  
  }
}

/* seek to a event position relative to the current position 
 * (just for completeness!) */
int Ttf_RelSeek( Ttf_FileHandleT handle, int plusMinusNumEvents )
{
  Ttf_fileT *tFile = (Ttf_fileT *) handle;
  off_t position;

  /* seek relative to the current position */
  position = lseek(tFile->Fid, plusMinusNumEvents*sizeof(PCXX_EV), SEEK_CUR);
  if (position)
  {
    /* success */
    return position/sizeof(PCXX_EV);
  }
  else
  {
    /* failure */
    return 0;
  }
}

/* read n events and call appropriate handlers 
 * Returns the number of records read (can be 0).
 * Returns a -1 value when an error takes place. Check errno */
int Ttf_ReadNumEvents( Ttf_FileHandleT fileHandle, Ttf_CallbacksT callbacks, 
		int numberOfEvents )
{
  Ttf_fileT *tFile = (Ttf_fileT *) fileHandle;
  PCXX_EV traceBuffer[TAU_BUFSIZE];
  long bytesRead, recordsRead, recordsToRead, i;
  int otherTid, otherNid, msgLen, msgTag;

  if (tFile == (Ttf_fileT *) NULL)
    return 0; /* ERROR */

  /* How many bytes are to be read? */
  recordsToRead = numberOfEvents > TAU_BUFSIZE ? TAU_BUFSIZE : numberOfEvents;

  /* if clock needs to be initialized, initialize it */
  if (!tFile->ClkInitialized)
  {
    if (*callbacks.DefClkPeriod)
      (*callbacks.DefClkPeriod)(callbacks.UserData, 1E-6);
    /* set flag to initialized */
    tFile->ClkInitialized = TRUE; 

    /* Read the first record and check its timestamp 
     * For this we need to lseek to the beginning of the file, read one 
     * record and then lseek it back to where it was */
    int originalPosition, currentPosition;
    originalPosition = lseek(tFile->Fid, 0, SEEK_CUR);
#ifdef DEBUG
    printf("Original position = %d\n", originalPosition);
#endif /* DEBUG */

    currentPosition = lseek(tFile->Fid, 0, SEEK_SET);
    if (currentPosition == -1)
    {
      perror("lseek failed in Ttf_ReadNumEvents");
    }
#ifdef DEBUG
    printf("Current position = %d\n", currentPosition);
#endif /* DEBUG */

    /* read just one record to get the first timestamp */
    while ((bytesRead = read(tFile->Fid, traceBuffer, sizeof(PCXX_EV))) !=
		    sizeof(PCXX_EV))
    {
      /* retry! The file may not have any data in it. Wait till it has some */
      currentPosition = lseek(tFile->Fid, 0, SEEK_SET);
#ifdef DEBUG
      printf("retrying current position = %d\n", currentPosition);
#endif /* DEBUG */
    }
    /* it now has exactly one record */
    /* FOR MONITORING, we disable this first timestamp check. Re-introduce it
     * later! 
     * tFile->FirstTimestamp = traceBuffer[0].ti;
     * */

    tFile->FirstTimestamp = traceBuffer[0].ti;

    /* now return the trace file to its original position */
    currentPosition = lseek(tFile->Fid, originalPosition, SEEK_SET);
#ifdef DEBUG
    printf("Returning trace to %d position\n", currentPosition);
#endif /* DEBUG */

  }
  /* Read n records and go through each event record */
  if ((bytesRead = read(tFile->Fid, traceBuffer, recordsToRead*sizeof(PCXX_EV)))
		  != (long)(numberOfEvents * sizeof(PCXX_EV)) )
  {
    /* Check if data read is inconsistent with the size of trace record */
    if ((bytesRead % sizeof(PCXX_EV)) != 0)
    {
#ifdef DEBUG
      printf("ERROR reading trace data, bytes read = %d, rec size=%d, recs to read = %d\n", 
		      bytesRead, sizeof(PCXX_EV), recordsToRead);
      printf("READ Error: inconsistent trace file. \n");
      printf("Bytes read are not integer multiples of the trace record size.\n");
      printf("Rewinding %d bytes... \n", bytesRead);
#endif /* DEBUG */
      int rewind_bytes = -1 * bytesRead;
      lseek(tFile->Fid, rewind_bytes, SEEK_CUR);
      return 0;
    }
  }
  /* the number of records read */
  recordsRead = bytesRead/sizeof(PCXX_EV); 

  /* See if the events are all present */
  for (i = 0; i < recordsRead; i++)
  {
    if (!isEventIDRegistered(tFile, traceBuffer[i].ev))
    {
      /* if event id is not found in the event id map, read the EDF file */
      if (!refreshTables(tFile, callbacks))
      { /* error */
	return -1;
      }
      if (!isEventIDRegistered(tFile, traceBuffer[i].ev))
      { /* even after reading the edf file, if we don't find the event id, 
	   then there's an error */
	return -1;
      }
      /* we did find the event id, process the trace file */
		  
    }
    /* event is OK. Examine each event and invoke callbacks for Entry/Exit/Node*/
    /* first check nodeid, threadid */
    int nid = traceBuffer[i].nid;
    int tid = traceBuffer[i].tid;
    NidTidMapT::iterator nit = tFile->NidTidMap->find(
      pair<int, int>(nid,tid));
    if (nit == tFile->NidTidMap->end())
    {
      /* this pair of node and thread has not been encountered before */
      char nodename[32];
      /* 
      sprintf(nodename, "Node %d Thread %d", nid, tid);
      */
      sprintf(nodename, "process %d:%d", nid, tid);

      /* invoke callback routine */
      if (*callbacks.DefThread)
        (*callbacks.DefThread)(callbacks.UserData, nid, tid, nodename);
      /* add it to the map! */
      (*(tFile->NidTidMap))[pair<int,int>(nid,tid)] = 1;
    }
    /* check the event to see if it is entry or exit */
    double ts = (double) (traceBuffer[i].ti - tFile->FirstTimestamp);
    long long parameter = traceBuffer[i].par;
    /* Get param entry from EventIdMap */
    Ttf_EventDescrT eventDescr = (*tFile->EventIdMap)[traceBuffer[i].ev];
    if ((eventDescr.Param != NULL) && (strcmp(eventDescr.Param,"EntryExit\n")==0))
    { /* entry/exit event */
#ifdef DEBUG
      printf("entry/exit event %s \n",eventDescr.EventName);
#endif /* DEBUG */
      if (parameter == 1)
      { /* entry event, invoke the callback routine */
	if (*callbacks.EnterState)
	  (*callbacks.EnterState)(callbacks.UserData, ts, nid, tid,
				   traceBuffer[i].ev);
      }
      else
      { if (parameter == -1)
	{ /* exit event */
	  if (*callbacks.LeaveState)
            (*callbacks.LeaveState)(callbacks.UserData,ts, nid, tid);
	}
      } 
	  
    } /* entry exit events *//* add message passing events here */
    else 
    {
      if ((eventDescr.Param != NULL) && (strcmp(eventDescr.Param,"TriggerValue\n")==0))
      { /* User defined event */
	 if (*callbacks.EventTrigger)
	   (*callbacks.EventTrigger)(callbacks.UserData, ts, nid, tid, 
				     traceBuffer[i].ev, traceBuffer[i].par);
      }
      if (eventDescr.Tag == TAU_MESSAGE_SEND_EVENT) 
      {
        /* send message */
        /* See RtsLayer::TraceSendMsg for documentation on the bit patterns of "parameter" */

	/* extract the information from the parameter */
	msgTag   = ((parameter>>16) & 0x000000FF) | (((parameter >> 48) & 0xFF) << 8);
	otherNid = ((parameter>>24) & 0x000000FF) | (((parameter >> 56) & 0xFF) << 8);
	msgLen   = parameter & 0x0000FFFF | (((parameter >> 32) & 0xFFFF) << 16);


	/* If the application is multithreaded, insert call for matching sends/recvs here */
	otherTid = 0;
	if (*callbacks.SendMessage) 
	  (*callbacks.SendMessage)(callbacks.UserData, ts, nid, tid, otherNid, otherTid, msgLen, msgTag);
	/* the args are user, time, source nid (my), source tid (my), dest nid (other), dest
	 * tid (other), size, tag */

      }
      else
      { /* Check if it is a message receive operation */
        if (eventDescr.Tag == TAU_MESSAGE_RECV_EVENT)
	{ 

        /* See RtsLayer::TraceSendMsg for documentation on the bit patterns of "parameter" */

	/* extract the information from the parameter */
	  msgTag   = ((parameter>>16) & 0x000000FF) | (((parameter >> 48) & 0xFF) << 8);
	  otherNid = ((parameter>>24) & 0x000000FF) | (((parameter >> 56) & 0xFF) << 8);
	  msgLen   = parameter & 0x0000FFFF | (((parameter >> 32) & 0xFFFF) << 16);
	  msgLen   = (((parameter >> 32) & 0xFFFF) );

	  /* If the application is multithreaded, insert call for matching sends/recvs here */
	  otherTid = 0;
	  if (*callbacks.RecvMessage) 
	    (*callbacks.RecvMessage)(callbacks.UserData, ts, otherNid, otherTid, nid, tid, msgLen, msgTag);
	  /* the args are user, time, source nid (my), source tid (my), dest nid (other), dest
	   * tid (other), size, tag */

	}
      }
    }
    if ((parameter == 0) && (eventDescr.EventName != NULL) &&
		    (strcmp(eventDescr.EventName, "\"FLUSH_CLOSE\"") == 0))
    {
      /* reset the flag in NidTidMap to 0 (from 1) */
      (*tFile->NidTidMap)[pair<int,int>(nid,tid)] = 0; 
      /* setting this flag to 0 tells us that a flush close has taken place 
       * on this node, thread */
    }
    else
    { /* see if it is a WALL_CLOCK record */
      if ((parameter != 1) && (parameter != -1) && (eventDescr.EventName != NULL) 
        && (strcmp(eventDescr.EventName, "\"WALL_CLOCK\"") == 0)) 
      { /* ok, it is a wallclock event alright. But is it the *last* wallclock event?
	 * We can confirm that it is if the NidTidMap flag has been set to 0 by a 
	 * previous FLUSH_CLOSE call */

	 if ((*tFile->NidTidMap)[pair<int,int>(nid,tid)] == 0 )
	 {
#ifdef DEBUG
           printf("LAST WALL_CLOCK! End of trace file detected \n");
#endif /* DEBUG */
	   /* see if an end of the trace callback is registered and 
	    * if it is, invoke it.*/
	   if (*callbacks.EndTrace) 
	     (*callbacks.EndTrace)(callbacks.UserData, nid, tid);
	 }
      } 
    } /* is it a WALL_CLOCK record? */

        
  } /* cycle through all records */
  
  /* return the number of event records read */
  return recordsRead;
}


/* Look for an event in the event map */
int isEventIDRegistered(Ttf_fileT *tFile, long int event)
{
  EventIdMapT::iterator it;
  if ((it = tFile->EventIdMap->find(event)) == tFile->EventIdMap->end())
  {
    /* couldn't locate the event id */
    return FALSE;
  }
  else
  {
    /* located the event id */
    return TRUE;
  }
}

/* Event ID is not found in the event map. Re-read the event 
 * description file */
int refreshTables(Ttf_fileT *tFile, Ttf_CallbacksT cb)
{
#ifdef DEBUG
  printf("Inside refreshTables! \n");
#endif /* DEBUG */

  int i,j,k; 
  char linebuf[LINEMAX], eventname[LINEMAX], traceflag[32]; 
  char group[512], param[512];
  int numevents, dynamictrace, tag, groupid; 
  long localEventId;
  EventIdMapT::iterator it;


  FILE *edf;

  dynamictrace = FALSE;
 
  /* first, open the edf file */
  if ((edf = fopen (tFile->EdfFile, "rb")) == NULL )
  {
    printf("ERROR: opening edf file %s\n", tFile->EdfFile);
    perror (tFile->EdfFile);
    return 0;
  }

  fgets (linebuf, LINEMAX, edf);
  sscanf (linebuf, "%d %s", &numevents, traceflag);
  if ((traceflag != NULL) && (strcmp(traceflag, "dynamic_trace_events") == 0)) 
  { 
    dynamictrace = TRUE;
  }

  for (i=0; i<numevents; i++)
  {
    fgets (linebuf, LINEMAX, edf);
    if ( (linebuf[0] == '\n') || (linebuf[0] == '#') )
    {
      /* -- skip empty, header and comment lines -- */
      i--;
      continue;
    }

    localEventId = -1;
    eventname[0]  = '\0';
    param[0] = '\0';
    if (dynamictrace) /* get eventname in quotes */
    { 
      memset(group,0,sizeof(group));
      sscanf (linebuf, "%ld %s %d", &localEventId, group, &tag);
#ifdef DEBUG
      printf("Got localEventId %ld group %s tag %d\n", localEventId, group, tag);
#endif /* DEBUG */
      for(j=0; linebuf[j] !='"'; j++)
	;
      eventname[0] = linebuf[j];
      j++;
      /* skip over till eventname begins */
      for (k=j; linebuf[k] != '"'; k++)
      {
	eventname[k-j+1] = linebuf[k];
      } 
      eventname[k-j+1] = '"';
      eventname[k-j+2] = '\0'; /* terminate eventname */

      strcpy(param, &linebuf[k+2]);

      // Fix 13/10 to 10 for event files generated with windows
      if (param[strlen(param)-2] == 13) {
 	param[strlen(param)-2] = 10;
 	param[strlen(param)-1] = 0;
      }

#ifdef DEBUG 
      printf(" Got eventname=%s param=%s\n", eventname, param);
#endif /* DEBUG */
      /* see if the event id exists in the map */
      if ((it = tFile->EventIdMap->find(localEventId)) == tFile->EventIdMap->end())
      {
        /* couldn't locate the event id */
	/* fill an event description object */
        Ttf_EventDescrT *eventDescr = new Ttf_EventDescrT;
	eventDescr->Eid = localEventId;
	eventDescr->EventName = strdup(eventname);
	eventDescr->Group = strdup(group);
	eventDescr->Tag = tag;
	eventDescr->Param = strdup(param);
	(*(tFile->EventIdMap))[localEventId] = (*eventDescr); /* add it to the map */
#ifdef DEBUG
	printf("Added event %ld %s %s to the map\n", localEventId, eventname, group);
#endif  /* DEBUG */
        /* invoke callbacks? Check group? */
	GroupIdMapT::iterator git = tFile->GroupIdMap->find(eventDescr->Group);
	if (git == tFile->GroupIdMap->end())
	{ /* group id not found. Generate group id on the fly */
	  groupid = tFile->GroupIdMap->size()+1;
	  (*(tFile->GroupIdMap))[eventDescr->Group] = groupid;
	  /* invoke group callback */
	  /* check Param to see if its a user defined event */
	  if (strcmp(eventDescr->Param, "TriggerValue") != 0)
	  { /* it is not a user defined event */
	    if (*cb.DefStateGroup)
	      (*cb.DefStateGroup)(cb.UserData, groupid, eventDescr->Group); 
	  }
	}
	else
	{ /* retrieve the stored group id token */
	  groupid = (*git).second;
	}
        /* invoke callback for registering a new state */
	if (strcmp(eventDescr->Param, "TriggerValue\n") == 0)
        { /* it is a user defined event */
          if (*cb.DefUserEvent)
	    (*cb.DefUserEvent)(cb.UserData, localEventId, eventDescr->EventName, eventDescr->Tag);
	}
	else
        { /* it is not a user defined event */
	  if (*cb.DefState)
	    (*cb.DefState)(cb.UserData, localEventId, eventDescr->EventName, 
		      groupid);
	}
	
      }
      /* else, do nothing, examine the next record */

    } /* not dynamic trace- what is to be done? */ 
    else 
    {  
      sscanf (linebuf, "%ld %s %d %s %s", &localEventId, 
        group, &tag, eventname, param);
    }

    if ( (localEventId < 0) || !*eventname )
    {
      fprintf (stderr, "%s: blurb in line %d\n", tFile->EdfFile, i+2);
    }
  } /* for loop */

  

  return TRUE;
}
/***************************************************************************
 * $RCSfile: TAU_tf.cpp,v $   $Author: amorris $
 * $Revision: 1.9 $   $Date: 2005/07/22 16:58:56 $
 * TAU_VERSION_ID: $Id: TAU_tf.cpp,v 1.9 2005/07/22 16:58:56 amorris Exp $ 
 ***************************************************************************/
