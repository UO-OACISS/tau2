/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauJAPI.java					  **
**	Description 	: TAU Profiling Package API wrapper for C	  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**			  -DSGI_TIMERS  for SGI fast nanosecs timer	  **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/

package TAU;

public class Profile
{
  public static final long TAU_DEFAULT = 0xffffffffL;

  public Profile(String name, String type, String groupname, long profileGroup)
  {
    //System.out.println("Inside Profile name = "+name+ " type = "+ type ); 
    NativeProfile(name, type, groupname, profileGroup);
  }
  public native void NativeProfile(String name, String type, String groupname,
	 long profileGroup);
 
  public void Start()
  {
    NativeStart();
  }

  public native void NativeStart();

  public void Stop()
  {
    NativeStop(); 
  }
  public native void NativeStop();

  public long FuncInfoPtr; /* Pointer to FunctionInfo Object */
 
  static
  {
    System.loadLibrary("TAU");
  }

}
   
