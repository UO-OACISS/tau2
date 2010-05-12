/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauIoWrap.h    				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : io wrapper                                       **
**                                                                         **
****************************************************************************/


/*********************************************************************
 * register different kinds of events here
 ********************************************************************/
#define NUM_EVENTS 4
typedef enum {
  WRITE_BW,
  WRITE_BYTES,
  READ_BW,
  READ_BYTES
} event_type;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


void Tau_iowrap_checkInit();
void *Tau_iowrap_getEvent(event_type type, int fid);
int Tau_iowrap_checkPassThrough();
void Tau_iowrap_registerEvents(int fid, const char *pathname);
void Tau_iowrap_unregisterEvents(int fid);
void Tau_iowrap_dupEvents(int oldfid, int newfid);

extern void *global_write_bandwidth, *global_read_bandwidth, 
  *global_bytes_written, *global_bytes_read;


#ifdef __cplusplus
} /* for extern "C" */
#endif /* __cplusplus */

#define TAU_GET_IOWRAP_EVENT(e, event, fid) void *e = Tau_iowrap_getEvent(event, fid);
