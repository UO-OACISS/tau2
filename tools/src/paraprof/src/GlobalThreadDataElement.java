/* 
  
GlobalThreadDataElement
  
Title:      ParaProf
Author:     Robert Bell
Description:  
*/

package paraprof;

import java.util.*;
import java.io.*;
import java.text.*;

public class GlobalThreadDataElement
{
    //Constructor.
    public GlobalThreadDataElement(Trial inTrial, boolean inUserElement){ 
	trial = inTrial;
	mappingID = -1;
    
	if(inUserElement){
	    userElement = true;
	    doubleList = new double[3];
	}
	else{
	    doubleList = new double[5];
	}
    }
  
    //Rest of the public functions.
    public String getMappingName(){
	GlobalMappingElement tmpGME = (GlobalMappingElement) (trial.getGlobalMapping()).getGlobalMappingElement(mappingID, 0);
	return tmpGME.getMappingName();}
  
    public void setMappingID(int inMappingID){
	mappingID = inMappingID;}
  
    public void setMappingExists(){
	mappingExists = true;}
  
    public boolean getMappingExists(){
	return mappingExists;}
  
    public int getMappingID(){
	return mappingID;}
  
    public void setInclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,0,inDouble);}
  
    public double getInclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,0);}
  
    public void setExclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,1,inDouble);}
    
    public double getExclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,1);}
    
    public void setInclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,2,inDouble);}
  
    public double getInclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,2);}
  
    public void setExclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,3,inDouble);}
    
    public double getExclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,3);}
  
    public void setNumberOfCalls(int inInt){
	numberOfCalls = inInt;}
  
    public int getNumberOfCalls(){
	return numberOfCalls;}
  
    public void setNumberOfSubRoutines(int inInt){
	numberOfSubRoutines = inInt;}
  
    public int getNumberOfSubRoutines(){
	return numberOfSubRoutines;}
  
    public void setUserSecPerCall(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,4,inDouble);}
  
    public double getUserSecPerCall(int dataValueLocation){
	return this.getDouble(dataValueLocation,4);}
  
  
    public static int getPositionOfName(){
	return 97;
    }
  
    public static String getTStatStringHeading(){
	try{
	    int defaultNumberPrecision = ParaProf.defaultNumberPrecision;
	    int initialBufferLength = 101;
	    int position = 0;
	    char [] statStringArray = new char[initialBufferLength];
	    char [] tmpArray;
	    String tmpString;
      
	    insertSpaces(statStringArray , 0, 100);
      
	    tmpArray = ("%time").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 7;
	    tmpArray = ("counts").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 25;
	    tmpArray = ("total counts").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 43;
	    tmpArray = ("#calls").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 61;
	    tmpArray = ("#subrs").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 79;
	    tmpArray = ("count/call").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 97;
	    tmpArray = ("name").toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    return new String(statStringArray);
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "GTDE01");
	    }
    
	return "An error occured pocessing this string!"; 
    }
  
    public String getTStatString(int dataValueLocation){
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
	    tmpArray = (dF.format(this.getInclusivePercentValue(dataValueLocation))).toCharArray();
      
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 7;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getExclusiveValue(dataValueLocation),
										  defaultNumberPrecision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 25;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getInclusiveValue(dataValueLocation),
										  defaultNumberPrecision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 43;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getNumberOfCalls(),
										  defaultNumberPrecision)));
	    tmpArray = tmpString.toCharArray();                       
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 61;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getNumberOfSubRoutines(),
										  defaultNumberPrecision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 79;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getUserSecPerCall(dataValueLocation),
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
  
    //User event interface.
    public String getUserEventName(){
	GlobalMappingElement tmpGME = (GlobalMappingElement) (trial.getGlobalMapping()).getGlobalMappingElement(mappingID, 2);
	return tmpGME.getMappingName();}
  
    public void setUserEventID(int inUserEventID){
	mappingID = inUserEventID;}
  
    public int getUserEventID(){
	return mappingID;}
  
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
	    int defaultNumberPrecision = ParaProf.defaultNumberPrecision;
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
		ParaProf.systemError(e, null, "GTDE01");
	    }
    
	return "An error occured pocessing this string!"; 
    }
  
    public String getUserEventStatString(){
	try{
	    int defaultNumberPrecision = ParaProf.defaultNumberPrecision;
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
										  defaultNumberPrecision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 36;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getUserEventMinValue(),
										  defaultNumberPrecision)));
	    tmpArray = tmpString.toCharArray();                       
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 54;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getUserEventMeanValue(),
										  defaultNumberPrecision)));
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
		ParaProf.systemError(e, null, "GTDE01");
	    }
    
	return "An error occured pocessing this string!";
    }
  
    public static int getPositionOfUserEventName(){
	return 72;
    }
  
    private static int insertSpaces(char[] inArray, int position, int number){
	for(int i=0;i<number;i++){
	    inArray[position] = '\u0020';
	    position++;
	}
	return position;
    }
  
    private void insertDouble(int dataValueLocation, int offset, double inDouble){
	int actualLocation = (dataValueLocation*5)+offset;
	try{
	    doubleList[actualLocation] = inDouble;
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "GTDE06");
	}
    }
  
    private double getDouble(int dataValueLocation, int offset){
	int actualLocation = (dataValueLocation*5)+offset;
	try{
	    return doubleList[actualLocation];
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "GTDE06");
	}
	return -1;
    }
  
    public void incrementStorage(){
	if(userElement)
	    System.out.println("We are trying to increase storage on a user event object!");
	
	int currentLength = doubleList.length;

	//can use a little space here ... space for speed! :-)
	double[] newArray = new double[currentLength+5];
	    
	for(int i=0;i<currentLength;i++){
	    newArray[i] = doubleList[i];
	}
	doubleList = newArray;
    }
    
  
    //Instance data.
    private Trial trial = null;
    boolean mappingExists = false;
    int mappingID;
    private double[] doubleList;
    private int numberOfCalls = 0;
    private int numberOfSubRoutines = 0;
    int userEventNumberValue = 0;
    boolean userElement = false;
}




