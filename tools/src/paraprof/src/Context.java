/* 
   Context.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;

public class Context{

    public Context(){
	threads = new Vector();}

    public void setID(int id){
	this.id = id;}

    public int getID(){
	return id;}
    
    //Adds a thread to this context's list of threads.
    public void addThread(Thread thread){
	try{
	    threads.addElement(thread);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "C1");
	}
    }
    
    public Vector getThreads(){
	return threads;}

    public Thread getThread(int id){
	Thread thread = null;
	try{
	    thread = (Thread) threads.elementAt(id);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "C2");
	}
	return thread;
    }

    public int getNumberOfThreads(){
	return threads.size();}
    
    //Instance data.
    int id = -1;
    Vector threads;
}
