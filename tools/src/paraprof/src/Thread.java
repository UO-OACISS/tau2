/* 
   Thread.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description: 
*/

package paraprof;

import java.util.*;
import java.io.*;

public class Thread implements Comparable{

    //####################################
    //Contructor(s).
    //####################################
    public Thread(){
	doubleList = new double[7];
    }
    
    public Thread(int nodeID, int contextID, int threadID){
	this.nodeID = nodeID;
	this.contextID = contextID;
	this.threadID = threadID;
	doubleList = new double[7];
    }
    //####################################
    //End - Contructor(s).
    //####################################
    
    //####################################
    //Public section.
    //####################################
    public void setNodeId(int nodeID){
	this.nodeID = nodeID;}

    public int getNodeID(){
	return nodeID;}

    public void setContextID(int contextID){
	this.contextID = contextID;}

    public int getContextID(){
	return contextID;}

    public void setThreadID(int threadID){
	this.threadID = threadID;}

    public int getThreadID(){
	return threadID;}

    public void initializeFunctionList(int size){
	functions = new Vector(size);
	Object ref = null;
	for(int i=0;i<size;i++){
	    functions.add(ref);
	}
    }

    public void initializeUsereventList(int size){
	userevents = new Vector(size);
	Object ref = null;
	for(int i=0;i<size;i++){
	    userevents.add(ref);
	}
    }
    
    public void incrementStorage(){
	int currentLength = doubleList.length;
	//can use a little space here ... space for speed! :-)
	double[] newArray = new double[currentLength+7];
	
	for(int i=0;i<currentLength;i++){
	    newArray[i] = doubleList[i];
	}
	doubleList = newArray;
    }

    public void addFunction(GlobalThreadDataElement ref){
	functions.addElement(ref);}
  
    public void addFunction(GlobalThreadDataElement ref, int id){
	//There are two paths here.
	//1) This id has not been seen in the system before.
	//   In this case, add to the end of functions.
	//2) The id has been seen in the system before.
	//   In this case, check to see if its location is
	//   not set to null in functions, and if it is not
	//   set the location to point to ref.
	if(id >= (functions.size()))
	    functions.add(ref);
	else{
	    if((functions.elementAt(id))==null)
		functions.setElementAt(ref, id);
	}
	if(this.debug()){
	    System.out.println("######");
	    System.out.println("Thread.addFuntion(...)");
	    System.out.println("id:"+id);
	    System.out.println("functions.size():"+functions.size());
	    System.out.println("######");
	}
    }
    
    public void addUserevent(GlobalThreadDataElement ref){
	userevents.addElement(ref);}

    public void addUserevent(GlobalThreadDataElement ref, int id){
	//There are two paths here.
	//1) This id has not been seen in the system before.
	//   In this case, add to the end of functions.
	//2) The id has been seen in the system before.
	//   In this case, check to see if its location is
	//   not set to null in functions, and if it is not
	//   set the location to point to ref.
	if(id >= (userevents.size()))
	    userevents.add(ref);
	else{
	    if((userevents.elementAt(id))==null)
		userevents.setElementAt(ref, id);
	}
	if(this.debug()){
	    System.out.println("######");
	    System.out.println("Thread.addUserevent(...)");
	    System.out.println("id:"+id);
	    System.out.println("userevents.size():"+functions.size());
	    System.out.println("######");
	}
    }
  
    public GlobalThreadDataElement getFunction(int id){
	GlobalThreadDataElement globalThreadDataElement = null;
	try{
	    if((functions!=null)&&(id<functions.size()))
		globalThreadDataElement = (GlobalThreadDataElement) functions.elementAt(id);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "T2");
	}
	return globalThreadDataElement;
    }

    public Vector getFunctionList(){
	return functions;}

    public ListIterator getFunctionListIterator(){
	return new ParaProfIterator(functions);}
  
    public GlobalThreadDataElement getUserevent(int id){
	GlobalThreadDataElement globalThreadDataElement= null;
	try{
	    if((userevents!=null)&&(id<userevents.size()))
		globalThreadDataElement = (GlobalThreadDataElement) userevents.elementAt(id);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "T3");
	}
	return globalThreadDataElement;
    }

    public Vector getUsereventList(){
	return userevents;}

    public ListIterator getUsereventListIterator(){
	return new ParaProfIterator(userevents);}

    
    public void setThreadData(int metric){
	this.setThreadDataHelper(metric);
	this.setPercentData(metric);
	this.setThreadDataHelper(metric);
    }

    private void setThreadDataHelper(int metric){
	double maxInclusiveValue = 0.0;
	double maxExclusiveValue = 0.0;
	double maxInclusivePercentValue = 0.0;
	double maxExclusivePercentValue = 0.0;
	double maxUserSecPerCall = 0.0;
	int maxNumberOfCalls = 0;
	int maxNumberOfSubroutines = 0;

	double d = 0.0;
	int i = 0;
	ListIterator l = this.getFunctionListIterator();
	
	while(l.hasNext()){
	    GlobalThreadDataElement globalThreadDataElement = (GlobalThreadDataElement) l.next();
	    if(globalThreadDataElement!=null){
		d = globalThreadDataElement.getInclusiveValue(metric);
		if(d>maxInclusiveValue)
		    maxInclusiveValue = d;
		d = globalThreadDataElement.getExclusiveValue(metric);
		if(d>maxExclusiveValue)
		    maxExclusiveValue = d;
		d = globalThreadDataElement.getInclusivePercentValue(metric);
		if(d>maxInclusivePercentValue)
		    maxInclusivePercentValue = d;
		d = globalThreadDataElement.getExclusivePercentValue(metric);
		if(d>maxExclusivePercentValue)
		    maxExclusivePercentValue = d;
		d = globalThreadDataElement.getUserSecPerCall(metric);
		if(d>maxUserSecPerCall)
		    maxUserSecPerCall = d;
		i = globalThreadDataElement.getNumberOfCalls();
		if(i>maxNumberOfCalls)
		    maxNumberOfCalls = i;
		i = globalThreadDataElement.getNumberOfSubRoutines();
		if(i>maxNumberOfSubroutines)
		    maxNumberOfSubroutines = i;
	    }
	}

	if(this.debug()){
	    System.out.println("------");
	    System.out.println("T-D01");
	    System.out.println("maxInclusiveValue:"+maxInclusiveValue);
	    System.out.println("maxExclusiveValue:"+maxExclusiveValue);
	    System.out.println("maxInclusivePercentValue:"+maxInclusivePercentValue);
	    System.out.println("maxExclusivePercentValue:"+maxExclusivePercentValue);
	    System.out.println("maxUserSecPerCall:"+maxUserSecPerCall);
	    System.out.println("maxNumberOfCalls:"+maxNumberOfCalls);
	    System.out.println("maxNumberOfSubroutines:"+maxNumberOfSubroutines);
	    System.out.println("------");
	}
	this.setMaxInclusiveValue(metric, maxInclusiveValue);
	this.setMaxExclusiveValue(metric, maxExclusiveValue);
	this.setMaxInclusivePercentValue(metric, maxInclusivePercentValue);
	this.setMaxExclusivePercentValue(metric, maxExclusivePercentValue);
	this.setMaxUserSecPerCall(metric, maxUserSecPerCall);
	this.setMaxNumberOfCalls(maxNumberOfCalls);
	this.setMaxNumberOfSubRoutines(maxNumberOfSubroutines);
    }

    private void setPercentData(int metric){
	ListIterator l = this.getFunctionListIterator();
	while(l.hasNext()){
	    GlobalThreadDataElement globalThreadDataElement = (GlobalThreadDataElement) l.next();
	    
	    if(globalThreadDataElement!=null){
		GlobalMappingElement globalMappingElement = globalThreadDataElement.getGlobalMappingElement();

		double inclusiveMax = this.getMaxInclusiveValue(metric);
		double d1 = globalThreadDataElement.getExclusiveValue(metric);
		double d2 = globalThreadDataElement.getInclusiveValue(metric);
		
		if(inclusiveMax!=0){
		    double result = (d1/inclusiveMax)*100.00;
		    globalThreadDataElement.setExclusivePercentValue(metric, result);
		    //Now do the global mapping element exclusive stuff.
		    if((globalMappingElement.getMaxExclusivePercentValue(metric)) < result)
			globalMappingElement.setMaxExclusivePercentValue(metric, result);

		    result = (d2/inclusiveMax) * 100;
		    globalThreadDataElement.setInclusivePercentValue(metric, result);
		    //Now do the global mapping element exclusive stuff.
		    if((globalMappingElement.getMaxInclusivePercentValue(metric)) < result)
			globalMappingElement.setMaxInclusivePercentValue(metric, result);
		}
	    }
	}
    }
    
    public void setMaxInclusiveValue(int metric, double inDouble){
	this.insertDouble(metric,0,inDouble);}
  
    public double getMaxInclusiveValue(int metric){
	return this.getDouble(metric,0);}

    public void setMaxExclusiveValue(int metric, double inDouble){
	this.insertDouble(metric,1,inDouble);}
  
    public double getMaxExclusiveValue(int metric){
	return this.getDouble(metric,1);}

    public void setMaxInclusivePercentValue(int metric, double inDouble){
	this.insertDouble(metric,2,inDouble);}
  
    public double getMaxInclusivePercentValue(int metric){
	return this.getDouble(metric,2);}

    public void setMaxExclusivePercentValue(int metric, double inDouble){
	this.insertDouble(metric,3,inDouble);}
  
    public double getMaxExclusivePercentValue(int metric){
	return this.getDouble(metric,3);}

    public void setMaxUserSecPerCall(int metric, double inDouble){
	this.insertDouble(metric,4,inDouble);}
  
    public double getMaxUserSecPerCall(int metric){
	return this.getDouble(metric,4);}
  
    public void setMaxNumberOfCalls(int inInt){
	maxNumberOfCalls = inInt;}
    
    public int getMaxNumberOfCalls(){
	return maxNumberOfCalls;}
    
    public void setMaxNumberOfSubRoutines(int inInt){
	maxNumberOfSubRoutines = inInt;}
    
    public int getMaxNumberOfSubRoutines(){
	return maxNumberOfSubRoutines;}

    //#######
    //The following two functions assist in determining whether this
    //thread's callpath information has been set correctly.
    //######
    public void setTrimmed(boolean b){
	trimmed = b;
    }

    public boolean trimmed(){
	return trimmed;}

    public int compareTo(Object obj){
	if(obj instanceof Integer)
	    return threadID - ((Integer)obj).intValue();
	else
	    return threadID - ((Thread)obj).getThreadID();
    }

    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}
    //####################################
    //End - Public section.
    //####################################


    //####################################
    //Private section.
    //####################################
    private void insertDouble(int metric, int offset, double inDouble){
	int actualLocation = (metric*5)+offset;
	try{
	    doubleList[actualLocation] = inDouble;
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "GT01");
	}
    }
  
    private double getDouble(int metric, int offset){
	int actualLocation = (metric*5)+offset;
	try{
	    return doubleList[actualLocation];
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "GT01");
	}
	return -1;
    }
    //####################################
    //End - Private section.
    //####################################

    //####################################
    //Instance data.
    //####################################
    int nodeID = -1;
    int contextID = -1;
    int threadID = -1;
    Vector functions = null;
    Vector userevents = null;
    private double[] doubleList;
    double totalExclusiveValue = 0;
    double totalInclusiveValue = 0;
    private int maxNumberOfCalls = 0;
    private int maxNumberOfSubRoutines = 0;
    private boolean trimmed = false;

    private boolean debug = false;
    //####################################
    //End - Instance data.
    //####################################
}
