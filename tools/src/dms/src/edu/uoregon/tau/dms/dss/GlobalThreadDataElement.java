/* 
   Name:        GlobalThreadDataElement.java
   Author:      Robert Bell
   Description: 
*/

package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.io.*;
import java.text.*;


public class GlobalThreadDataElement implements Mapping{
    
    //####################################
    //Contructor(s).
    //####################################
    public GlobalThreadDataElement(GlobalMappingElement globalMappingElement, boolean ue){
	if(ue){
	    userevent = true;
	    doubleList = new double[4];
	}
	else{
	    doubleList = new double[5];
	}

	this.globalMappingElement = globalMappingElement;
    }

    public GlobalThreadDataElement(GlobalMappingElement globalMappingElement, boolean ue, int capacity){
	if(ue){
	    userevent = true;
	    doubleList = new double[capacity*4];
	}
	else{
	    doubleList = new double[capacity*5];
	}

	this.globalMappingElement = globalMappingElement;
    }
    //####################################
    //End - Contructor(s).
    //####################################
  
    //####################################
    //Public section.
    //####################################
    public GlobalMappingElement getGlobalMappingElement(){
	return globalMappingElement;}

    public String getMappingName(){
	return globalMappingElement.getMappingName();}

    public void setMappingName(String mappingName){} //Don't set Name.
  
    public void setMappingID(int mappingID){} //Don't set ID.

    public int getMappingID(){
	return globalMappingElement.getMappingID();}

    public boolean isGroupMember(int groupID){
	return globalMappingElement.isGroupMember(groupID);}

    public int[] getGroups(){
	return globalMappingElement.getGroups();}
  
    public void setInclusiveValue(int metric, double inDouble){
	this.insertDouble(metric,0,inDouble);}
  
    public double getInclusiveValue(int metric){
	return this.getDouble(metric,0);}
  
    public void setExclusiveValue(int metric, double inDouble){
	this.insertDouble(metric,1,inDouble);}
    
    public double getExclusiveValue(int metric){
	return this.getDouble(metric,1);}
    
    public void setInclusivePercentValue(int metric, double inDouble){
	this.insertDouble(metric,2,inDouble);}
  
    public double getInclusivePercentValue(int metric){
	return this.getDouble(metric,2);}
  
    public void setExclusivePercentValue(int metric, double inDouble){
	this.insertDouble(metric,3,inDouble);}
    
    public double getExclusivePercentValue(int metric){
	return this.getDouble(metric,3);}
  
    public void setNumberOfCalls(int inInt){
	numberOfCalls = inInt;}
  
    public int getNumberOfCalls(){
	return numberOfCalls;}
  
    public void setNumberOfSubRoutines(int inInt){
	numberOfSubRoutines = inInt;}
  
    public int getNumberOfSubRoutines(){
	return numberOfSubRoutines;}
  
    public void setUserSecPerCall(int metric, double inDouble){
	this.insertDouble(metric,4,inDouble);}
  
    public double getUserSecPerCall(int metric){
	return this.getDouble(metric,4);}
  
  
    public static int getPositionOfName(){
	return 103;
    }
  
    public static String getTStatStringHeading(String metricType) {
	try {
	    return UtilFncs.lpad("%"+metricType,7) + 
		UtilFncs.lpad(metricType,16) + 
		UtilFncs.lpad("Total "+metricType,18) +
		UtilFncs.lpad("#Calls",14) +
		UtilFncs.lpad("#Subrs",14) + 
		UtilFncs.lpad("Total "+metricType+"/Call",21) 
		+ "   ";	    
	} catch (Exception e) {
	    UtilFncs.systemError(e, null, "GTDE01");
	}
	return "An error occurred processing this string!"; 
    }
  
    public String getTStatString(int type, int metric) {
	try{

	    String tmpString;

	    DecimalFormat dF = new DecimalFormat("##0.0");
	    tmpString = UtilFncs.lpad(dF.format(this.getInclusivePercentValue(metric)),7);
      
	    tmpString = tmpString + "  " + UtilFncs.getOutputString(type,this.getExclusiveValue(metric),14);
	    tmpString = tmpString + "  " + UtilFncs.getOutputString(type,this.getInclusiveValue(metric),16);
	    tmpString = tmpString + "  " + UtilFncs.formatDouble(this.getNumberOfCalls(),12);
	    tmpString = tmpString + "  " + UtilFncs.formatDouble(this.getNumberOfSubRoutines(),12);
	    tmpString = tmpString + "  " + UtilFncs.getOutputString(type,this.getUserSecPerCall(metric),19);
      
	    //Everything should be added now except the function name.
	    return tmpString;
	} catch(Exception e) {
	    UtilFncs.systemError(e, null, "GTDE01");
	}
    
	return "An error occurred processing this string!"; 
    }
  
    //User event interface.
    public boolean userevent(){
	return userevent;}
  
    public void setUserEventNumberValue(int inInt){
	userEventNumberValue = inInt;
    }
  
    public int getUserEventNumberValue(){
	return userEventNumberValue;
    }
  
    public void setUserEventMinValue(double inDouble){
	doubleList[0]=inDouble;}
  
    public double getUserEventMinValue(){
	return doubleList[0];}
    
    public void setUserEventMaxValue(double inDouble){
	doubleList[1]=inDouble;}
  
    public double getUserEventMaxValue(){
	return doubleList[1];}
  
    public void setUserEventMeanValue(double inDouble){
	doubleList[2]=inDouble;}
  
    public double getUserEventMeanValue(){
	return doubleList[2];}
  
    public void setUserEventSumSquared(double inDouble){
	doubleList[3]=inDouble;}
  
    public double getUserEventSumSquared(){
	return doubleList[3];}

  
    public static String getUserEventStatStringHeading(){

	int w = 18;
	return UtilFncs.pad("NumSamples",w) + 
	    UtilFncs.pad("Max",w) + 
	    UtilFncs.pad("Min",w) +
	    UtilFncs.pad("Mean",w) +
	    UtilFncs.pad("Std. Dev",w);	    

	// this is great fun to maintain, what is the point of this stuff?
	/*	try{

	    
	    int width = 16;
	    int initialBufferLength = 91;
	    int position = 0;
	    char [] statStringArray = new char[initialBufferLength];
	    char [] tmpArray;
	    String tmpString;
      
	    insertSpaces(statStringArray , 0, 91);
      
	    tmpArray = ("NumSamples").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 18;
	    tmpArray = ("MaxValue").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 36;
	    tmpArray = ("MinValue").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 54;
	    tmpArray = ("MeanValue").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }

	    return new String(statStringArray);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "GTDE01");
	}
	return "An error occurred processing this string!"; 
	*/
    }
  
    public String getUserEventStatString(int precision){
	try{
	    int initialBufferLength = 90;
	    int position = 0;
	    char [] statStringArray = new char[initialBufferLength];
	    char [] tmpArray;
	    String tmpString;
      
	    this.insertSpaces(statStringArray , 0, 90);
      
	    tmpArray = (Integer.toString(this.getUserEventNumberValue()).toCharArray());
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 18;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getUserEventMaxValue(),
										  precision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 36;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getUserEventMinValue(),
										  precision)));
	    tmpArray = tmpString.toCharArray();                       
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 54;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getUserEventMeanValue(),
										  precision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      

	    // compute the standard deviation
	    // this should be computed just once, somewhere else
	    // but it's here, for now
	    double sumsqr = this.getUserEventSumSquared();
	    double numEvents = this.getUserEventNumberValue();
	    double mean = this.getUserEventMeanValue();

	    double stddev = java.lang.Math.sqrt(java.lang.Math.abs( sumsqr/numEvents)
				- ( mean * mean ));

	    position = 72;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(stddev,
										  precision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }


	    //Everything should be added now except the function name.
	    return new String(statStringArray);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "GTDE01");
	}
    
	return "An error occurred processing this string!";
    }
  

    public void addCall(double exclusive, double inclusive){
	if(calls==null)
	    calls = new Vector();
	double[] arr = new double[2];
	arr[0] = exclusive;
	arr[1] = inclusive;
	calls.add(arr);
    }
    
    public boolean isCallPathObject(){
	return globalMappingElement.isCallPathObject();}

    public int addParent(int id){
	//This function is used in the building
	//of the parent/child relations from the
	//global mapping. As such, it does not have to
	//be quite as careful as the standard addParent
	//function (which is left in below).
	if(parents==null){
	    parents = new Vector();
	    callPathIDSParents = new Vector();
	}
	parents.add(new Integer(id));
	Vector tmpVector = new Vector();
	callPathIDSParents.add(tmpVector);
	return (parents.size()-1);
    }

    public void addParentCallPathID(int location, int pathID){
	Vector tmpVector = (Vector) callPathIDSParents.elementAt(location);
	tmpVector.add(new Integer(pathID));
    }

    public void addParent(int id,int pathID){
	//Check to see if this parent is already present,
	//if so, add only the callpath to the system.
	int location = UtilFncs.exists(parents,id);
	if(location == -1){
	    if(parents==null){
		parents = new Vector();
		callPathIDSParents = new Vector();
	    }
	    
	    parents.add(new Integer(id));
	    Vector tmpVector = new Vector();
	    tmpVector.add(new Integer(pathID));
	    callPathIDSParents.add(tmpVector);

	}
	else{
	    Vector tmpVector = (Vector) callPathIDSParents.elementAt(location);
	    tmpVector.add(new Integer(pathID));
	}
    }

    public Vector getParents(){
	return parents;
    }

    public Vector getChildren(){
	return children;
    }

    public ListIterator getParentsIterator(){
	return new DataSessionIterator(parents);
    }

    public ListIterator getChildrenIterator(){
	return new DataSessionIterator(children);
    }

    public ListIterator getCallPathIDParents(int id){
	//The argument represents the id of the parent.
	//Get the location of the parent first.
	int location = UtilFncs.exists(parents,id);
	//Now return the callpath id list for that parent.
	return new DataSessionIterator((Vector)callPathIDSParents.elementAt(location));
    }

    public ListIterator getCallPathIDChildren(int id){
	//The argument represents the id of the child.
	//Get the location of the child first.
	int location = UtilFncs.exists(children,id);
	//Now return the callpath id list for that child.
	return new DataSessionIterator((Vector)callPathIDSChildren.elementAt(location));
    }

    public int addChild(int id){
	//This function is used in the building
	//of the parent/child relations from the
	//global mapping. As such, it does not have to
	//be quite as careful as the standard addParent
	//function (which is left in below).
	if(children==null){
		children = new Vector();
		callPathIDSChildren = new Vector();
	}
	children.add(new Integer(id));
	Vector tmpVector = new Vector();
	callPathIDSChildren.add(tmpVector);
	return (children.size()-1);
    }

    public void addChildCallPathID(int location, int pathID){
	Vector tmpVector = (Vector) callPathIDSChildren.elementAt(location);
	tmpVector.add(new Integer(pathID));
    }

    public void addChild(int id,int pathID){
	//Check to see if this child is already present,
	//if so, add only the callpath to the system.
	int location = UtilFncs.exists(children,id);
	if(location == -1){
	    if(children==null){
		children = new Vector();
		callPathIDSChildren = new Vector();
	    }
	    
	    children.add(new Integer(id));
	    Vector tmpVector = new Vector();
	    tmpVector.add(new Integer(pathID));
	    callPathIDSChildren.add(tmpVector);
	}
	else{
	    Vector tmpVector = (Vector) callPathIDSChildren.elementAt(location);
	    tmpVector.add(new Integer(pathID));
	}
    }

	public int getStorageSize() {
	    if (userevent == true) 
			return doubleList.length / 4;
		else
			return doubleList.length / 5;
	}

    public void incrementStorage(){
	if(userevent)
	    UtilFncs.systemError(null, null, "Error: Attempt to increase storage on a user event object!");
	int currentLength = doubleList.length;
	//can use a little space here ... space for speed! :-)
	double[] newArray = new double[currentLength+5];
	
	for(int i=0;i<currentLength;i++){
	    newArray[i] = doubleList[i];
	}
	doubleList = newArray;
    }

    //####################################
    //Private section.
    //####################################
    private void insertDouble(int metric, int offset, double inDouble){
	int actualLocation = (metric*5)+offset;
	try{
	    doubleList[actualLocation] = inDouble;
	}
	catch(Exception e){
		e.printStackTrace();
		System.out.println("inDouble: " + inDouble);
		System.out.println("metric: " + metric);
		System.out.println("offset: " + offset);
		System.out.println("actualLocation: " + actualLocation);
		System.out.println("doubleList size: " + doubleList.length);
	    UtilFncs.systemError(e, null, "GTDE06");
	}
    }
  
    private double getDouble(int metric, int offset){
	int actualLocation = (metric*5)+offset;
	try{
	    return doubleList[actualLocation];
	}
	catch(Exception e){
		e.printStackTrace();
		e.printStackTrace();
		System.out.println("metric: " + metric);
		System.out.println("offset: " + offset);
		System.out.println("actualLocation: " + actualLocation);
		System.out.println("doubleList size: " + doubleList.length);
	    UtilFncs.systemError(e, null, "GTDE06");
	}
	return -1;
    }
    
    private static int insertSpaces(char[] inArray, int position, int number){
	for(int i=0;i<number;i++){
	    inArray[position] = '\u0020';
	    position++;
	}
	return position;
    }

    //####################################
    //End - Private section.
    //####################################
    
    //####################################
    //Instance data.
    //####################################
    private GlobalMappingElement globalMappingElement = null;
    private boolean mappingExists = false;
    private double[] doubleList;
    private int numberOfCalls = 0;
    private int numberOfSubRoutines = 0;
    private int userEventNumberValue = 0;
    private boolean userevent = false;

    private Vector calls = null;

    private Vector parents = null;
    private Vector children = null;
    private Vector callPathIDSParents = null;
    private Vector callPathIDSChildren = null;
    //####################################
    //End - Instance data.
    //####################################
}
