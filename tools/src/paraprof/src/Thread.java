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
	doubleList = new double[5];
    }
    //####################################
    //End - Contructor(s).
    //####################################
    
    //####################################
    //Public section.
    //####################################
    public void setID(int id){
	this.id = id;}

    public int getID(){
	return id;}

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
	double[] newArray = new double[currentLength+5];
	
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
    
    //Functions used to calculate the percentage values for derived values (such as flops)
    public void setTotalExclusiveValue(double inDouble){
	totalExclusiveValue = inDouble;}
    
    public void incrementTotalExclusiveValue(double inDouble){
	totalExclusiveValue = totalExclusiveValue + inDouble;}
    
    public double getTotalExclusiveValue(){
	return totalExclusiveValue;}
    
    public void setTotalInclusiveValue(double inDouble){
	totalInclusiveValue = inDouble;}
    
    public void incrementTotalInclusiveValue(double inDouble){
	totalInclusiveValue = totalInclusiveValue + inDouble;}
    
    public double getTotalInclusiveValue(){
	return totalInclusiveValue;}

    
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
    int id = -1;
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
