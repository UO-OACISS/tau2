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
**	File 		: TAU_tf_decl.h					  **
**	Description 	: TAU trace format reader library internal data   **
**                        structures                                      **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu 	                  **
***************************************************************************/
/* TAU Trace format */
#ifndef _TAU_TF_DECL_H_
#define _TAU_TF_DECL_H_

/* general declarations */
# ifndef TRUE
#   define FALSE  0
#   define TRUE   1
# endif

#ifdef TAU_HP_GNU
# define LINEMAX        2*1024
#else
# define LINEMAX        64*1024
#endif /* TAU_HP_GNU */

#define TAU_BUFSIZE 	1024

/* TAU trace library related declarations */
typedef struct Ttf_EventDescr {
  long int  Eid; /* event id */
  char *Group; /* state as in TAU_VIZ */
  char *EventName; /* name as in "foo" */
  int  Tag; /* -7 for send etc. */
  char *Param; /* param as in EntryExit */
} Ttf_EventDescrT;

struct Ttf_ltstr
{
  bool operator()(const char* s1, const char* s2) const
  {
    return strcmp(s1, s2) < 0;
  }
};


typedef map< pair<int, int>, int, less < pair<int, int> > > NidTidMapT;
typedef map< long int , Ttf_EventDescrT, less <long int> > EventIdMapT;
typedef map< const char *, int, Ttf_ltstr > GroupIdMapT;
typedef struct Ttf_file 
{
  int 		Fid;
  char * 	EdfFile;
  NidTidMapT 	*NidTidMap;
  EventIdMapT 	*EventIdMap;
  GroupIdMapT	*GroupIdMap;
  int 		ClkInitialized;
  double        FirstTimestamp;
} Ttf_fileT;

int refreshTables(Ttf_fileT *tFile, Ttf_CallbacksT cb);
int isEventIDRegistered(Ttf_fileT *tFile, long int eid);

#endif /* _TAU_TF_DECL_H_ */

/***************************************************************************
 * $RCSfile: TAU_tf_decl.h,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 2003/11/13 00:09:30 $
 * TAU_VERSION_ID: $Id: TAU_tf_decl.h,v 1.1 2003/11/13 00:09:30 sameer Exp $ 
 ***************************************************************************/
