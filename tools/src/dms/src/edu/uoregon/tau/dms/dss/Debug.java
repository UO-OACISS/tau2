
/* 
   Name:        ParaProfDataSession.java
   Author:      Robert Bell
   Description:  
*/

package edu.uoregon.tau.dms.dss;

import java.io.*;

public class Debug{

    public Debug(){
	super();
	try{
	    out = new PrintWriter(new FileWriter(new File("ParaProf.debug.out")));
	}
	catch(IOException exception){
	    System.out.println("An error occured trying whilst trying to create ParaProf.debug.out!");
	    System.out.println("Debugging information will be printed to the standard out instead.");
	    System.out.println("Details about this error are below:");
	    exception.printStackTrace();
	}
	
    }

    //####################################
    //Public Section.
    //####################################
    public void outputToFile(String s){
	if(out!=null)
	    out.println("\n######\n"+s+"\n######");
	else
	    System.out.println(s);
    }
    
    public void flushDebugFileBuffer(){
	if(out!=null)
	    out.flush();
    }
    
    public void closeDebugFile(){
	if(out!=null)
	    out.close();
    }
    //####################################
    //End - Public Section.
    //####################################

    //####################################
    //Protected Section.
    //####################################
    //####################################
    //End - Protected Section.
    //####################################

    //####################################
    //Private Section.
    //####################################
    //####################################
    //End - Private Section.
    //####################################

    //####################################
    //Instance data.
    //####################################

    //######
    //Private Section.
    //######
    public static boolean debug = false;
    //When in debugging mode, this class can print a lot of data.
    //Initialized in this.setDebug(...).
    private PrintWriter out = null;
    //######
    //End - Private Section.
    //######

    //####################################
    //End - Instance data.
    //####################################
}    
