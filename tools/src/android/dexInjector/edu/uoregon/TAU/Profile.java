/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauJAPI.java					  **
**	Description 	: TAU Profiling Package API wrapper for C	  **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

package edu.uoregon.TAU;

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
    try {
      System.loadLibrary("TAU");
    } catch (SecurityException e) {
      System.out.println("Security Manager doesn't allow loading libTAU!");
    } catch (UnsatisfiedLinkError e) {
      System.out.println("Can't find libTAU!");
    }
  }

}
   
