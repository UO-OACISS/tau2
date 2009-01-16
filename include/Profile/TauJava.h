/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 1997                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**      File            : TauJava.h                                       **
**      Description     : TAU interface for JVMPI                         **
**      Author          : Sameer Shende                                   **
**      Contact         : tau-bugs@cs.uoregon.edu                         **
**      Documentation   : See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Declarations 
//////////////////////////////////////////////////////////////////////

#ifndef _TAU_JAVA_H_
#define _TAU_JAVA_H_

struct TauJavaLayer {
  static void Init(char *options);
  static void NotifyEvent(JVMPI_Event *event);
  static void ClassLoad(JVMPI_Event *event);
  static void MethodEntry(JVMPI_Event *event);
  static void MethodExit(JVMPI_Event *event);
  static void ThreadStart(JVMPI_Event *event);
  static void ThreadEnd(JVMPI_Event *event);
  static void ShutDown(JVMPI_Event *event);
  static void DataDump(JVMPI_Event *event);
  static void DataPurge(JVMPI_Event *event);
  static int *RegisterThread(JVMPI_Event *event);
  static void CreateTopLevelRoutine(char *name, char *type, char *groupname, 
                        int tid);
  static int  GetTid(JVMPI_Event *event);
  static int  NumThreads;
};


extern "C" {
  JNIEXPORT jint JNICALL JVM_OnLoad(JavaVM *jvm, char *options, void *reserved);
}

#endif /* _TAU_JAVA_H_ */
