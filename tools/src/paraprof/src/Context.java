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

    public Context(int nodeID, int contextID){
	this.nodeID = nodeID;
	this.contextID = contextID;
	threads = new Vector();
    }

    public void setNodeId(int nodeID){
	this.nodeID = nodeID;}

    public int getNodeID(){
	return nodeID;}

    public void setContextID(int contextID){
	this.contextID = contextID;}

    public int getContextID(){
	return contextID;}
    
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
    int nodeID = -1;
    int contextID = -1;
    Vector threads;
}
