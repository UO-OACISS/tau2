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

#include "Profile/tau_types.h"

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
#define TAU_MAX_RECORDS 64*1024

/* TAU trace library related declarations */
typedef struct Ttf_EventDescr {
  int  Eid; /* event id */
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


#define FORMAT_NATIVE  0   // as a fallback
#define FORMAT_32      1
#define FORMAT_64      2
#define FORMAT_32_SWAP 3
#define FORMAT_64_SWAP 4


/* for 32 bit platforms */
typedef struct {
  x_int32            ev;    /* -- event id        -- */
  x_uint16           nid;   /* -- node id         -- */
  x_uint16           tid;   /* -- thread id       -- */
  x_int64            par;   /* -- event parameter -- */
  x_uint64           ti;    /* -- time [us]?      -- */
} TAU_EV32;

/* for 64 bit platforms */
typedef struct {
  x_int64            ev;    /* -- event id        -- */
  x_uint16           nid;   /* -- node id         -- */
  x_uint16           tid;   /* -- thread id       -- */
  x_uint32           padding; /*  space wasted for 8-byte aligning the next item */ 
  x_int64            par;   /* -- event parameter -- */
  x_uint64           ti;    /* -- time [us]?      -- */
} TAU_EV64;


typedef TAU_EV32 EVENT;
typedef TAU_EV TAU_EV_NATIVE;




#define swap16(A)  ((((x_uint16)(A) & 0xff00) >> 8) | \
                   (((x_uint16)(A) & 0x00ff) << 8))
#define swap32(A)  ((((x_uint32)(A) & 0xff000000) >> 24) | \
                   (((x_uint32)(A) & 0x00ff0000) >> 8)  | \
                   (((x_uint32)(A) & 0x0000ff00) << 8)  | \
                   (((x_uint32)(A) & 0x000000ff) << 24))
#define swap64(A)  ((((x_uint64)(A) & 0xff00000000000000ull) >> 56) | \
                    (((x_uint64)(A) & 0x00ff000000000000ull) >> 40) | \
                    (((x_uint64)(A) & 0x0000ff0000000000ull) >> 24) | \
                    (((x_uint64)(A) & 0x000000ff00000000ull) >> 8) | \
                    (((x_uint64)(A) & 0x00000000ff000000ull) << 8) | \
                    (((x_uint64)(A) & 0x0000000000ff0000ull) << 24)  | \
                    (((x_uint64)(A) & 0x000000000000ff00ull) << 40)  | \
                    (((x_uint64)(A) & 0x00000000000000ffull) << 56))
  



typedef map< pair<int, int>, int, less < pair<int, int> > > NidTidMapT;
typedef map< long int , Ttf_EventDescrT, less <long int> > EventIdMapT;
typedef map< const char *, int, Ttf_ltstr > GroupIdMapT;

/* for trace writing */
typedef map< int, const char * > GroupNameMapT;

typedef struct Ttf_file 
{
  int 		Fid;
  char * 	EdfFile;
  NidTidMapT 	*NidTidMap;
  EventIdMapT 	*EventIdMap;
  GroupIdMapT	*GroupIdMap;
  int 		ClkInitialized;
  double        FirstTimestamp;
  bool		subtractFirstTimestamp;
  bool		nonBlocking;
  int           format;    // see above
  int           eventSize; // sizeof() the corresponding format struct

  /* For Trace Writing */
  EVENT         *traceBuffer; 
  int           tracePosition;
  bool          needsEdfFlush;
  GroupNameMapT *groupNameMap;
  bool		initialized;
  double        lastTimestamp;

  bool          forWriting;
} Ttf_fileT;





extern "C" int refreshTables(Ttf_fileT *tFile, Ttf_CallbacksT cb);
extern "C" int isEventIDRegistered(Ttf_fileT *tFile, long int eid);



#endif /* _TAU_TF_DECL_H_ */

/***************************************************************************
 * $RCSfile: TAU_tf_decl.h,v $   $Author: amorris $
 * $Revision: 1.7 $   $Date: 2009/02/19 22:30:03 $
 * TAU_VERSION_ID: $Id: TAU_tf_decl.h,v 1.7 2009/02/19 22:30:03 amorris Exp $ 
 ***************************************************************************/
