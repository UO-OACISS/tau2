/* 
   Thread.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description: 
*/

package paraprof;

import java.util.*;
import java.io.*;

public class Thread{

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
  
    public void addFunction(GlobalThreadDataElement ref, int pos){
	functions.setElementAt(ref, pos);}
    
    public void addUserevent(GlobalThreadDataElement ref){
	userevents.addElement(ref);}

    public void addUserevent(GlobalThreadDataElement ref, int pos){
	userevents.setElementAt(ref, pos);}
  
    public GlobalThreadDataElement getFunction(int id){
	GlobalThreadDataElement globalThreadDataElement= null;
	try{
	    globalThreadDataElement = (GlobalThreadDataElement) functions.elementAt(id);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "T2");
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
	    globalThreadDataElement = (GlobalThreadDataElement) userevents.elementAt(id);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "T2");
	}
	return globalThreadDataElement;
    }

    public Vector getUsereventList(){
	return userevents;}

    public ListIterator getUsereventListIterator(){
	return new ParaProfIterator(userevents);}

    public void setThreadSummaryData(int metric){
	double maxInclusiveValue = 0.0;
	double maxExclusiveValue = 0.0;
	double maxInclusivePercentValue = 0.0;
	double maxExclusivePercentValue = 0.0;
	double maxUserSecPerCall = 0.0;
	int maxNumberOfCalls = 0;
	int maxNumberOfSubroutines = 0;

	double totalInclusiveValue = 0.0;
	double totalExclusiveValue = 0.0;
	
	double d = 0.0;
	int i = 0;
	ListIterator l = this.getFunctionListIterator();
	
	while(l.hasNext()){
	    GlobalThreadDataElement globalThreadDataElement = (GlobalThreadDataElement) l.next();
	    
	    d = globalThreadDataElement.getInclusiveValue(metric);
	    if(d>maxInclusiveValue)
		maxInclusiveValue = d;
	    totalInclusiveValue+=d;
	    d = globalThreadDataElement.getExclusiveValue(metric);
	    if(d>maxExclusiveValue)
		maxExclusiveValue = d;
	    totalExclusiveValue+=d;
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

	if(ParaProf.debugIsOn){
	    System.out.println("------");
	    System.out.println("T-D01");
	    System.out.println("maxInclusiveValue:"+maxInclusiveValue);
	    System.out.println("maxExclusiveValue:"+maxExclusiveValue);
	    System.out.println("maxInclusivePercentValue:"+maxInclusivePercentValue);
	    System.out.println("maxExclusivePercentValue:"+maxExclusivePercentValue);
	    System.out.println("maxUserSecPerCall:"+maxUserSecPerCall);
	    System.out.println("maxNumberOfCalls:"+maxNumberOfCalls);
	    System.out.println("maxNumberOfSubroutines:"+maxNumberOfSubroutines);
	    System.out.println("totalInclusiveValue:"+totalInclusiveValue);
	    System.out.println("totalExclusiveValue:"+totalExclusiveValue);
	    System.out.println("------");
	}
	this.setMaxInclusiveValue(metric, maxInclusiveValue);
	this.setMaxExclusiveValue(metric, maxExclusiveValue);
	this.setMaxInclusivePercentValue(metric, maxInclusivePercentValue);
	this.setMaxExclusivePercentValue(metric, maxExclusivePercentValue);
	this.setMaxUserSecPerCall(metric, maxUserSecPerCall);
	this.setMaxNumberOfCalls(maxNumberOfCalls);
	this.setMaxNumberOfSubRoutines(maxNumberOfSubroutines);
	this.setTotalInclusiveValue(metric, totalInclusiveValue);
	this.setTotalExclusiveValue(metric, totalExclusiveValue);
    }

    public void setPercentData(int dataValueLocation){
	ListIterator l = this.getFunctionListIterator();
	while(l.hasNext()){
	    GlobalThreadDataElement globalThreadDataElement = (GlobalThreadDataElement) l.next();
	    double exclusiveTotal = this.getTotalExclusiveValue(dataValueLocation);
	    double inclusiveMax = this.getMaxInclusiveValue(dataValueLocation);
	    
	    double d1 = globalThreadDataElement.getExclusiveValue(dataValueLocation);
	    double d2 = globalThreadDataElement.getInclusiveValue(dataValueLocation);
		
	    if(exclusiveTotal!=0){
		double result = (d1/exclusiveTotal)*100.00;
		globalThreadDataElement.setExclusivePercentValue(dataValueLocation, result);
	    }

	    if(inclusiveMax!=0){
		double result = (d2/inclusiveMax) * 100;
		globalThreadDataElement.setInclusivePercentValue(dataValueLocation, result);
	    }
	}
    }
    
    public void setMaxInclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,0,inDouble);}
  
    public double getMaxInclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,0);}

    public void setMaxExclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,1,inDouble);}
  
    public double getMaxExclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,1);}

    public void setMaxInclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,2,inDouble);}
  
    public double getMaxInclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,2);}

    public void setMaxExclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,3,inDouble);}
  
    public double getMaxExclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,3);}

    public void setMaxUserSecPerCall(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,4,inDouble);}
  
    public double getMaxUserSecPerCall(int dataValueLocation){
	return this.getDouble(dataValueLocation,4);}
  
    public void setMaxNumberOfCalls(int inInt){
	maxNumberOfCalls = inInt;}
    
    public int getMaxNumberOfCalls(){
	return maxNumberOfCalls;}
    
    public void setMaxNumberOfSubRoutines(int inInt){
	maxNumberOfSubRoutines = inInt;}
    
    public int getMaxNumberOfSubRoutines(){
	return maxNumberOfSubRoutines;}

    public void setTotalInclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,5,inDouble);}

    public double getTotalInclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,5);}

    public void setTotalExclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,6,inDouble);}

    public double getTotalExclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,6);}
        
    //#######
    //The following two functions assist in determining whether this
    //thread's callpath information has been set correctly.
    //######
    public void setTrimmed(boolean b){
	trimmed = b;
    }

    public boolean trimmed(){
	return trimmed;}

    //####################################
    //End - Public section.
    //####################################


    //####################################
    //Private section.
    //####################################
    private void insertDouble(int dataValueLocation, int offset, double inDouble){
	int actualLocation = (dataValueLocation*5)+offset;
	try{
	    doubleList[actualLocation] = inDouble;
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "GT01");
	}
    }
  
    private double getDouble(int dataValueLocation, int offset){
	int actualLocation = (dataValueLocation*5)+offset;
	try{
	    return doubleList[actualLocation];
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "GT01");
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
    //####################################
    //End - Instance data.
    //####################################
}
