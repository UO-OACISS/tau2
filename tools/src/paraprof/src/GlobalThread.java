/* 
   GlobalThread.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import java.io.*;

public class GlobalThread implements Serializable 
{
    //Constructor.
    public GlobalThread(){
	threadDataList = new Vector(10,10);
	userThreadDataList = new Vector(10,10);
    }
    
    //Rest of the public functions.
  
    public void incrementStorage(){
	int currentLength = maxInclusiveValueList.length;
	
	//can use a little space here ... space for speed! :-)
	double[] newArray1 = new double[currentLength+1];
	double[] newArray2 = new double[currentLength+1];
	double[] newArray3 = new double[currentLength+1];
	double[] newArray4 = new double[currentLength+1];
	double[] newArray5 = new double[currentLength+1];
	
	for(int i=0;i<currentLength;i++){
	    newArray1[i] = maxInclusiveValueList[i];
	    newArray2[i] = maxExclusiveValueList[i];
	    newArray3[i] = maxInclusivePercentValueList[i];
	    newArray4[i] = maxExclusivePercentValueList[i];
	    newArray5[i] = maxUserSecPerCallList[i];
	}
	
	maxInclusiveValueList = newArray1;
	maxExclusiveValueList = newArray2;
	maxInclusivePercentValueList = newArray3;
	maxExclusivePercentValueList = newArray4;
	maxUserSecPerCallList = newArray5;
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
  
    Vector getThreadDataList(){
	return threadDataList;
    }
  
    Vector getUserThreadDataList(){
	return userThreadDataList;
    }
  
    public void setMaxInclusiveValue(int dataValueLocation, double inDouble){
	maxInclusiveValueList[dataValueLocation] = inDouble;}
  
    public double getMaxInclusiveValue(int dataValueLocation){
	return maxInclusiveValueList[dataValueLocation];}

    public void setMaxExclusiveValue(int dataValueLocation, double inDouble){
	maxExclusiveValueList[dataValueLocation] = inDouble;}
  
    public double getMaxExclusiveValue(int dataValueLocation){
	return maxExclusiveValueList[dataValueLocation];}

    public void setMaxInclusivePercentValue(int dataValueLocation, double inDouble){
	maxInclusivePercentValueList[dataValueLocation] = inDouble;}
  
    public double getMaxInclusivePercentValue(int dataValueLocation){
	return maxInclusivePercentValueList[dataValueLocation];}

    public void setMaxExclusivePercentValue(int dataValueLocation, double inDouble){
	maxExclusivePercentValueList[dataValueLocation] = inDouble;}
  
    public double getMaxExclusivePercentValue(int dataValueLocation){
	return maxExclusivePercentValueList[dataValueLocation];}

    public void setMaxUserSecPerCall(int dataValueLocation, double inDouble){
	maxUserSecPerCallList[dataValueLocation] = inDouble;}
  
    public double getMaxUserSecPerCall(int dataValueLocation){
	return maxUserSecPerCallList[dataValueLocation];}
  
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

    private double[] maxInclusiveValueList = new double[1];
    private double[] maxExclusiveValueList = new double[1];
    private double[] maxInclusivePercentValueList = new double[1];
    private double[] maxExclusivePercentValueList = new double[1];
    private double[] maxUserSecPerCallList = new double[1];

    double totalExclusiveValue = 0;
    double totalInclusiveValue = 0;
    private int maxNumberOfCalls = 0;
    private int maxNumberOfSubRoutines = 0;
}
