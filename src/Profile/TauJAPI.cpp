#include <jni.h>
#include "Profile/Profiler.h"
#include "Profile/TauJAPI.h"
#include "Profile/TauJava.h"

/*
 * Class:     Profile
 * Method:    NativeProfile
 * Signature: (Ljava/lang/String;Ljava/lang/String;J)V
 */

JNIEXPORT void JNICALL Java_TAU_Profile_NativeProfile
  (JNIEnv *env, jobject obj, jstring name, jstring type, jstring groupname, 
	jlong group)
{

  /* Get name and type strings from the JVM */
  const char *blockName = env->GetStringUTFChars(name, 0);
  const char *blockType = env->GetStringUTFChars(type, 0);
  const char *blockGroup = env->GetStringUTFChars(groupname, 0);

  /* create a new FunctionInfo object by passing these to it */
  FunctionInfo *f = new FunctionInfo(blockName, blockType, (TauGroup_t) group, 
	blockGroup, true);
  /* true indicates InitData will ensure that all data is clean */


  /* Now release the strings back to the JVM */
  env->ReleaseStringUTFChars(name, blockName);
  env->ReleaseStringUTFChars(type, blockType);
  env->ReleaseStringUTFChars(groupname, blockGroup);

  /* Find the field FuncInfoPtr in the Profile class where we need to store 
     the address of the FunctionInfo object just created */

  jclass cls = env->GetObjectClass(obj);
  jfieldID fid = env->GetFieldID(cls, "FuncInfoPtr", "J");


  /* Check if new was successful */

  if (f == (FunctionInfo *) NULL)
  {
    cout << "ERROR: FunctionInfo new returns NULL: Memory problem"<<endl;
  }

  /* Store the address of f in the Java class field where it can be accessed
     by successive JNI calls such as Start and Stop */

  env->SetLongField(obj, fid, (jlong) f); 
  DEBUGPROFMSG("Java_Profile_NativeProfile: FunctionInfoPtr set to "<<f<<endl);

}


/*
 * Class:     Profile
 * Method:    NativeStart
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_TAU_Profile_NativeStart
  (JNIEnv *env, jobject obj)
{

  /* Find the FunctionInfo Pointer associated with this method*/
  jclass cls = env->GetObjectClass(obj);
  jfieldID fid;
  FunctionInfo *f; 

  fid = env->GetFieldID(cls, "FuncInfoPtr", "J");

  f = (FunctionInfo *) env->GetLongField(obj, fid); 

  Profiler *p = new Profiler(f, f != (FunctionInfo *) 0 ? f->GetProfileGroup() : TAU_DEFAULT, true); 

  if (p == (Profiler *) NULL)
  {
    cout << "ERROR: Profiler new returns NULL: Memory problem"<<endl;
  }
  else 
  {
    /* Everything went well. Start the Profiler */
    p->Start(RtsLayer::myThread()); 
    DEBUGPROFMSG("TAU STMT START: Profiler = "<< p<< " Name = "<<
	p->ThisFunction->GetName()<<endl);
  }
}




/*
 * Class:     Profile
 * Method:    NativeStop
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_TAU_Profile_NativeStop
  (JNIEnv * env, jobject obj)
{

  /* Stop the Current profiler */
  int tid = RtsLayer::myThread();
  Profiler *p = Profiler::CurrentProfiler[tid];
  p->Stop(tid);
  DEBUGPROFMSG("TAU STMT STOP: Profiler = "<< p<< " Name = "<<
	p->ThisFunction->GetName()<<endl);
  delete p;

}

/* EOF Profile.cpp */

/***************************************************************************
 * $RCSfile: TauJAPI.cpp,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 2000/12/02 19:50:02 $
 * TAU_VERSION_ID: $Id: TauJAPI.cpp,v 1.1 2000/12/02 19:50:02 sameer Exp $
 ***************************************************************************/

