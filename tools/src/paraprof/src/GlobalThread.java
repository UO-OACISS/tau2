/* 
   GlobalThread.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import java.io.*;

public class GlobalThread 
{
    //Constructor.
    public GlobalThread(){
	doubleList = new double[5];
    }

    public void initializeThreadDataList(int size){
	threadDataList = new Vector(size);
	Object ref = null;
	for(int i=0;i<size;i++){
	    threadDataList.add(ref);
	}
    }

    public void initializeUserThreadDataList(int size){
	userThreadDataList = new Vector(size);
	Object ref = null;
	for(int i=0;i<size;i++){
	    userThreadDataList.add(ref);
	}
    }
    
    //Rest of the public functions.
  
    public void incrementStorage(){
	int currentLength = doubleList.length;
	
	//can use a little space here ... space for speed! :-)
	double[] newArray = new double[currentLength+5];
	
	for(int i=0;i<currentLength;i++){
	    newArray[i] = doubleList[i];
	}
	doubleList = newArray;
    }

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
    
    //The following function adds a thread data element to
    //the threadDataList
    void addThreadDataElement(GlobalThreadDataElement inGTDE){
	threadDataList.addElement(inGTDE);
    }
  
    void addThreadDataElement(GlobalThreadDataElement inGTDE, int inPosition){
	threadDataList.setElementAt(inGTDE, inPosition);
    }
    
  //The following function adds a thread data element to
  //the userThreadDataList
    void addUserThreadDataElement(GlobalThreadDataElement inGTDE){
	userThreadDataList.addElement(inGTDE);
    }

    void addUserThreadDataElement(GlobalThreadDataElement inGTDE, int inPosition){
	userThreadDataList.setElementAt(inGTDE, inPosition);
    }
  
    Vector getThreadDataList(){
	return threadDataList;
    }
  
    Vector getUserThreadDataList(){
	return userThreadDataList;
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
    
    //Instance data.
    Vector threadDataList;
    Vector userThreadDataList;
    private double[] doubleList;
    double totalExclusiveValue = 0;
    double totalInclusiveValue = 0;
    private int maxNumberOfCalls = 0;
    private int maxNumberOfSubRoutines = 0;
}
