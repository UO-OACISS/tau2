/* 
   GlobalThreadDataElement.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description: 
*/

package paraprof;

import java.util.*;
import java.io.*;
import java.text.*;

public class GlobalThreadDataElement{
    
    //####################################
    //Contructor(s).
    //####################################
    public GlobalThreadDataElement(GlobalMappingElement globalMappingElement, boolean ue){
	if(ue){
	    userevent = true;
	    doubleList = new double[3];
	}
	else{
	    doubleList = new double[5];
	}

	this.globalMappingElement = globalMappingElement;
    }

    public GlobalThreadDataElement(GlobalMappingElement globalMappingElement, boolean ue, int capacity){
	if(ue){
	    userevent = true;
	    doubleList = new double[capacity*3];
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

    public int getMappingID(){
	return globalMappingElement.getGlobalID();}

    public String getMappingName(){
	return globalMappingElement.getMappingName();}
  
    public void setMappingExists(){
	mappingExists = true;}
  
    public boolean getMappingExists(){
	return mappingExists;}
  
    public boolean isGroupMember(int groupID){
	return globalMappingElement.isGroupMember(groupID);}
  
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
	return 97;
    }
  
    public static String getTStatStringHeading(String metricType){
	try{
	    int initialBufferLength = 103;
	    int position = 0;
	    char [] statStringArray = new char[initialBufferLength];
	    char [] tmpArray;
	    String tmpString;
      
	    insertSpaces(statStringArray , 0, 103);
      
	    tmpArray = ("%"+metricType).toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 9;
	    tmpArray = (metricType).toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 27;
	    tmpArray = ("total "+metricType).toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 45;
	    tmpArray = ("#calls").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 63;
	    tmpArray = ("#subrs").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 81;
	    tmpArray = ("total "+metricType+"/call").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 99;
	    tmpArray = ("name").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    return new String(statStringArray);
	}
	catch(Exception e)
	    {
		UtilFncs.systemError(e, null, "GTDE01");
	    }
    
	return "An error occured pocessing this string!"; 
    }
  
    public String getTStatString(int type, int metric, int precision){
	try{
	    int initialBufferLength = 99;
	    int position = 0;
	    char [] statStringArray = new char[initialBufferLength];
	    char [] tmpArray;
	    String tmpString;
      
	    this.insertSpaces(statStringArray , 0, 99);
      
	    DecimalFormat dF = new DecimalFormat();
	    dF.applyPattern("##0.0");
	    tmpArray = (dF.format(this.getInclusivePercentValue(metric))).toCharArray();
      
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 9;
	    tmpString = UtilFncs.getOutputString(type,this.getExclusiveValue(metric),precision);

	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 27;
	    tmpString = UtilFncs.getOutputString(type,this.getInclusiveValue(metric),precision);

	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 45;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getNumberOfCalls(),
										  precision)));
	    tmpArray = tmpString.toCharArray();                       
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 63;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getNumberOfSubRoutines(),
										  precision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 81;
	    tmpString = UtilFncs.getOutputString(type,this.getUserSecPerCall(metric),precision);

	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    //Everything should be added now except the function name.
	    String firstPart = new String(statStringArray);
	    return firstPart + this.getMappingName();
	}
	catch(Exception e)
	    {
		UtilFncs.systemError(e, null, "GTDE01");
	    }
    
	return "An error occured pocessing this string!"; 
    }
  
    //User event interface.
    public String getUserEventName(){
	return globalMappingElement.getMappingName();}
  
    public int getUserEventID(){
	return globalMappingElement.getGlobalID();}
  
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
  
  
    public static String getUserEventStatStringHeading(){
	try{
	    int initialBufferLength = 82;
	    int position = 0;
	    char [] statStringArray = new char[initialBufferLength];
	    char [] tmpArray;
	    String tmpString;
      
	    insertSpaces(statStringArray , 0, 82);
      
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
      
	    position = 72;
	    tmpArray = ("Event Name").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    return new String(statStringArray);
	}
	catch(Exception e)
	    {
		UtilFncs.systemError(e, null, "GTDE01");
	    }
    
	return "An error occured pocessing this string!"; 
    }
  
    public String getUserEventStatString(int precision){
	try{
	    int initialBufferLength = 72;
	    int position = 0;
	    char [] statStringArray = new char[initialBufferLength];
	    char [] tmpArray;
	    String tmpString;
      
	    this.insertSpaces(statStringArray , 0, 72);
      
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
      
	    //Everything should be added now except the function name.
	    String firstPart = new String(statStringArray);
	    return firstPart + this.getUserEventName();
	}
	catch(Exception e)
	    {
		UtilFncs.systemError(e, null, "GTDE01");
	    }
    
	return "An error occured pocessing this string!";
    }
  
    public static int getPositionOfUserEventName(){
	return 72;
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
	return new ParaProfIterator(parents);
    }

    public ListIterator getChildrenIterator(){
	return new ParaProfIterator(children);
    }

    public ListIterator getCallPathIDParents(int id){
	//The argument represents the id of the parent.
	//Get the location of the parent first.
	int location = UtilFncs.exists(parents,id);
	//Now return the callpath id list for that parent.
	return new ParaProfIterator((Vector)callPathIDSParents.elementAt(location));
    }

    public ListIterator getCallPathIDChildren(int id){
	//The argument represents the id of the child.
	//Get the location of the child first.
	int location = UtilFncs.exists(children,id);
	//Now return the callpath id list for that child.
	return new ParaProfIterator((Vector)callPathIDSChildren.elementAt(location));
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

    public void incrementStorage(){
	if(userevent)
	    UtilFncs.systemError(null, null, "Error: Attemp to increase storage on a user event object!");
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
	    UtilFncs.systemError(e, null, "GTDE06");
	}
    }
  
    private double getDouble(int metric, int offset){
	int actualLocation = (metric*5)+offset;
	try{
	    return doubleList[actualLocation];
	}
	catch(Exception e){
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
    GlobalMappingElement globalMappingElement = null;
    boolean mappingExists = false;
    private double[] doubleList;
    private int numberOfCalls = 0;
    private int numberOfSubRoutines = 0;
    int userEventNumberValue = 0;
    boolean userevent = false;

    Vector calls = null;

    private Vector parents = null;
    private Vector children = null;
    private Vector callPathIDSParents = null;
    private Vector callPathIDSChildren = null;
    //####################################
    //End - Instance data.
    //####################################
}
