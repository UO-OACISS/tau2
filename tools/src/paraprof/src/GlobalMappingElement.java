/* 
   GlobalMappingElement.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;
import java.text.*;

public class GlobalMappingElement implements Serializable, Comparable{
    //Constructors.
    public GlobalMappingElement(Trial inTrial){
	trial = inTrial;
    }

    public void incrementStorage(){
	int currentLength = doubleList.length;
	double[] newArray = new double[currentLength+14];
	for(int i=0;i<currentLength;i++){
	    newArray[i] = doubleList[i];
	}
	doubleList = newArray;
    }
  
    public void setMappingName(String inMappingName){
	mappingName = inMappingName;
    }
  
    public String getMappingName(){
	return mappingName;
    }
  
    public void setGlobalID(int inGlobalID){
	globalID = inGlobalID;
    }
  
    public int getGlobalID(){
	return globalID;
    }
  
    public void addGroup(int id){
	if(groups==null)
	    groups = new int[5];
	else if(groups.length<numberOfGroups){
	    int currentLength = groups.length;
	    int[] newArray = new int[currentLength+5];
	    for(int i=0;i<currentLength;i++){
		newArray[i] = groups[i];
	    }
	    groups = newArray;	    
	}
	//Safe to add.
	groups[numberOfGroups] = id;
	numberOfGroups++;
    }

    public boolean isGroupMember(int inGroupID){
	GlobalMapping tmpGM = trial.getGlobalMapping();
    
	boolean tmpBool = tmpGM.getIsAllExceptGroupOn();
	boolean tmpBoolResult = false;
    
	for(int i=0;i<numberOfGroups;i++){
	    if(groups[i] == inGroupID){
		tmpBoolResult = true;
		break;
	    }
	}
	
	if(!tmpBool)
	    return tmpBoolResult;
	else
	    return (!tmpBoolResult);
    }

    public void addParent(int id){
	
	if(parents==null)
	    parents = new int[5];
	else if(parents.length<numberOfParents){
	    int currentLength = parents.length;
	    int[] newArray = new int[currentLength+5];
	    for(int i=0;i<currentLength;i++){
		newArray[i] = parents[i];
	    }
	    parents = newArray;	    
	}
	//Safe to add.
	parents[numberOfParents] = id;
	numberOfParents++;
    }

    public int[] getParents(){
	return parents;
    }

    public int getNumberOfParents(){
	return numberOfParents;
    }

    public int[] getChildren(){
	return children;
    }

    public int getNumberOfChildren(){
	return numberOfChildren;
    }

    public void addChild(int id){
	if(children==null)
	    children = new int[5];
	else if(children.length<numberOfChildren){
	    int currentLength = children.length;
	    int[] newArray = new int[currentLength+5];
	    for(int i=0;i<currentLength;i++){
		newArray[i] = children[i];
	    }
	    children = newArray;	    
	}
	//Safe to add.
	children[numberOfChildren] = id;
	numberOfChildren++;
    }
    
    public void setColorFlag(boolean inBoolean){
	colorFlag = inBoolean;
    }
  
    public boolean isColorFlagSet(){
	return colorFlag;
    }
  
    public void setGenericColor(Color inColor){
	genericMappingColor = inColor;
    }
  
    public void setSpecificColor(Color inColor){
	specificMappingColor = inColor;
    }
  
    public Color getMappingColor(){
	if(colorFlag)
	    return specificMappingColor;
	else
	    return genericMappingColor;
    }
  
    public Color getGenericColor(){
	return genericMappingColor;
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
  
    public void setMaxNumberOfCalls(int inInt){
	maxNumberOfCalls = inInt;}
  
    public int getMaxNumberOfCalls(){
	return maxNumberOfCalls;}
  
    public void setMaxNumberOfSubRoutines(int inInt){
	maxNumberOfSubRoutines = inInt;}
  
    public int getMaxNumberOfSubRoutines(){
	return maxNumberOfSubRoutines;}
  
    public void setMaxUserSecPerCall(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,4,inDouble);}
  
    public double getMaxUserSecPerCall(int dataValueLocation){
	return this.getDouble(dataValueLocation,4);}
    
    //User event section.
    public void setMaxUserEventNumberValue(int inInt){
	maxUserEventNumberValue = inInt;}
  
    public int getMaxUserEventNumberValue(){
	return maxUserEventNumberValue;}
  
    public void setMaxUserEventMinValue(double inDouble){
	maxUserEventMinValue = inDouble;}
  
    public double getMaxUserEventMinValue(){
	return maxUserEventMinValue;}
  
    public void setMaxUserEventMaxValue(double inDouble){
	maxUserEventMaxValue = inDouble;}
  
    public double getMaxUserEventMaxValue(){
	return maxUserEventMaxValue;}
  
    public void setMaxUserEventMeanValue(double inDouble){
	maxUserEventMeanValue = inDouble;}
  
    public double getMaxUserEventMeanValue(){
	return maxUserEventMeanValue;
    }
  
  
  
    //Mean section.
    public void setMeanInclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,5,inDouble);}
  
    public double getMeanInclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,5);}
  
    public void setMeanExclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,6,inDouble);}
  
    public double getMeanExclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,6);}
  
    public void setMeanInclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,7,inDouble);}
  
    public double getMeanInclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,7);}
  
    public void setMeanExclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,8,inDouble);}
  
    public double getMeanExclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,8);}
  
    public void setMeanNumberOfCalls(double inDouble){
	meanNumberOfCalls = inDouble;}
  
    public double getMeanNumberOfCalls(){
	return meanNumberOfCalls;}
  
    public void setMeanNumberOfSubRoutines(double inDouble){
	meanNumberOfSubRoutines = inDouble;}
  
    public double getMeanNumberOfSubRoutines(){
	return meanNumberOfSubRoutines;}
  
    public void setMeanUserSecPerCall(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,9,inDouble);}
  
    public double getMeanUserSecPerCall(int dataValueLocation){
  	return this.getDouble(dataValueLocation,9);}
  
    public void setTotalInclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,10,inDouble);}
  
    public double getTotalInclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,10);}
  
    public void setTotalExclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,11,inDouble);}
  
    public double getTotalExclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,11);}
  
    public void setTotalInclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,12,inDouble);}
  
    public double getTotalInclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,12);}
  
    public void setTotalExclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,13,inDouble);}
  
    public double getTotalExclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,13);}
    
  
    //Stat Strings. 
    public String getMeanTotalStatString(int dataValueLocation){
  
	try{
	    int defaultNumberPrecision = ParaProf.defaultNumberPrecision;
	    int initialBufferLength = 97;
	    int position = 0;
	    char [] statStringArray = new char[initialBufferLength];
	    char [] tmpArray;
	    String tmpString;
      
	    this.insertSpaces(statStringArray , 0, 97);
      
	    DecimalFormat dF = new DecimalFormat();
	    dF.applyPattern("##0.0");
	    tmpArray = (dF.format(this.getMeanInclusivePercentValue(dataValueLocation))).toCharArray();
      
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 7;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getMeanExclusiveValue(dataValueLocation),
										  defaultNumberPrecision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 25;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getMeanInclusiveValue(dataValueLocation),
										  defaultNumberPrecision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 43;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getMeanNumberOfCalls(),
										  defaultNumberPrecision)));
	    tmpArray = tmpString.toCharArray();                       
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 61;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getMeanNumberOfSubRoutines(),
										  defaultNumberPrecision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 79;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getMeanUserSecPerCall(dataValueLocation),
										  defaultNumberPrecision)));
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
		ParaProf.systemError(e, null, "GTDE01");
	    }
    
	return "An error occured pocessing this string!";
    }
  
    private int insertSpaces(char[] inArray, int position, int number){
	for(int i=0;i<number;i++){
	    inArray[position] = '\u0020';
	    position++;
	}
	return position;
    }
    
    public void setMeanValuesSet(boolean inBoolean)
    {
	meanValuesSet = inBoolean;
    }
  
    public boolean getMeanValuesSet()
    {
	return meanValuesSet;
    }
  
    public void setDrawCoords(int inXBeg, int inXEnd, int inYBeg, int inYEnd)
    {
	xBeginPosition = inXBeg;
	xEndPosition = inXEnd;
	yBeginPosition = inYBeg;
	yEndPosition = inYEnd;
    }
  
    public int getXBeg()
    {
	return xBeginPosition;
    }
  
    public int getXEnd()
    {
	return xEndPosition;
    }
  
    public int getYBeg()
    {
	return yBeginPosition;
    }
  
    public int getYEnd()
    {
	return yEndPosition;
    }
  
    //Functions used to calculate the mean values for derived values (such as flops)
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
    
    public void setCounter(int inInt){
	counter = inInt;}
    
    public void incrementCounter(){
	counter++;}
    
    public int getCounter(){
	return counter;}
  
  
    public int compareTo(Object inObject){
	return mappingName.compareTo((String)inObject);
    }

    private void insertDouble(int dataValueLocation, int offset, double inDouble){
	int actualLocation = (dataValueLocation*14)+offset;
	try{
	    doubleList[actualLocation] = inDouble;
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "GME01");
	}
    }
  
    private double getDouble(int dataValueLocation, int offset){
	int actualLocation = (dataValueLocation*14)+offset;
	try{
	    return doubleList[actualLocation];
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "GME02");
	}
	return -1;
    }
  
    //Instance elmements.
  
    private Trial trial = null;
  
    //Global Mapping reference.
    String mappingName = null;
    int globalID = -1;     //Global ID for this mapping.
  
    int[] groups = null;
    int numberOfGroups = 0;
    int[] parents = null;
    int numberOfParents = 0;
    int[] children = null;
    int numberOfChildren = 0;

    //Color Settings.
    boolean colorFlag = false;
    Color genericMappingColor = null;
    Color specificMappingColor = null;
  
    private double[] doubleList = new double[14];
    private int maxNumberOfCalls = 0;
    private int maxNumberOfSubRoutines = 0;
    private int maxUserEventNumberValue = 0;
    private double maxUserEventMinValue = 0;
    private double maxUserEventMaxValue = 0;
    private double maxUserEventMeanValue = 0;
    private double meanNumberOfCalls = 0;
    private double meanNumberOfSubRoutines = 0;
  
    //Drawing coordinates for this Global mapping element.
    int xBeginPosition;
    int xEndPosition;
    int yBeginPosition;
    int yEndPosition;
  
    boolean meanValuesSet = false;
  
    //Instance values used to calculate the mean values for derived values (such as flops)
    int counter = 0;
    double totalExclusiveValue = 0;
    double totalInclusiveValue = 0;
}
