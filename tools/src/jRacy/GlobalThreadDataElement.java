/* 
	
	GlobalThreadDataElement
	
	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;
import java.text.*;

public class GlobalThreadDataElement implements Serializable 
{
	//Constructor.
	public GlobalThreadDataElement(Trial inTrial, boolean inUserElement){	
		trial = inTrial;
		globalMappingReference = trial.getGlobalMapping();
		mappingID = -1;
		
		if(inUserElement)
			userElement = true;
		else{
			doubleList4 = new double[1];
			doubleList5 = new double[1];
		}
	}
	
	//Rest of the public functions.
	public String getMappingName(){
		tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
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
		this.insertDouble(1,dataValueLocation,inDouble);}
	
	public double getInclusiveValue(int dataValueLocation){
		return this.getDouble(1,dataValueLocation);}
	
	public void setExclusiveValue(int dataValueLocation, double inDouble){
		this.insertDouble(2,dataValueLocation,inDouble);}
		
	public double getExclusiveValue(int dataValueLocation){
		return this.getDouble(2,dataValueLocation);}
		
	public void setInclusivePercentValue(int dataValueLocation, double inDouble){
		this.insertDouble(3,dataValueLocation,inDouble);}
	
	public double getInclusivePercentValue(int dataValueLocation){
		return this.getDouble(3,dataValueLocation);}
	
	public void setExclusivePercentValue(int dataValueLocation, double inDouble){
		this.insertDouble(4,dataValueLocation,inDouble);}
		
	public double getExclusivePercentValue(int dataValueLocation){
		return this.getDouble(4,dataValueLocation);}
	
	public void setNumberOfCalls(int inInt){
		numberOfCalls = inInt;}
	
	public int getNumberOfCalls(){
		return numberOfCalls;}
	
	public void setNumberOfSubRoutines(int inInt){
		numberOfSubRoutines = inInt;}
	
	public int getNumberOfSubRoutines(){
		return numberOfSubRoutines;}
	
	public void setUserSecPerCall(int dataValueLocation, double inDouble){
		this.insertDouble(5,dataValueLocation,inDouble);}
	
	public double getUserSecPerCall(int dataValueLocation){
		return this.getDouble(5,dataValueLocation);}
	
	public String getTStatString(int dataValueLocation){
		try{
			int initialBufferLength = 80;
			int position = 0;
			char [] statStringArray = new char[initialBufferLength];
			char [] tmpArray;
			
			DecimalFormat dF = new DecimalFormat();
			dF.applyPattern("##0.0");
			tmpArray = (dF.format(this.getInclusivePercentValue(dataValueLocation))).toCharArray();
			
			for(int i=0;i<tmpArray.length;i++){
				statStringArray[position] = tmpArray[i];
				position++;
			}
			position = this.insertSpaces(statStringArray , position, 2);
			
			dF.applyPattern("0.######E0");
			tmpArray = (dF.format(this.getExclusiveValue(dataValueLocation))).toCharArray();
			for(int i=0;i<tmpArray.length;i++){
				statStringArray[position] = tmpArray[i];
				position++;
			}
			position = this.insertSpaces(statStringArray , position, 2);
			
			tmpArray = (dF.format(this.getInclusiveValue(dataValueLocation))).toCharArray();
			for(int i=0;i<tmpArray.length;i++){
				statStringArray[position] = tmpArray[i];
				position++;
			}
			position = this.insertSpaces(statStringArray , position, 2);
			
			tmpArray = (Integer.toString(this.getNumberOfCalls())).toCharArray();
			for(int i=0;i<tmpArray.length;i++){
				statStringArray[position] = tmpArray[i];
				position++;
			}
			position = this.insertSpaces(statStringArray , position, 2);
			
			tmpArray = (Integer.toString(this.getNumberOfSubRoutines())).toCharArray();
			for(int i=0;i<tmpArray.length;i++){
				statStringArray[position] = tmpArray[i];
				position++;
			}
			position = this.insertSpaces(statStringArray , position, 2);
			
			tmpArray = (dF.format(this.getUserSecPerCall(dataValueLocation))).toCharArray();
			for(int i=0;i<tmpArray.length;i++){
				statStringArray[position] = tmpArray[i];
				position++;
			}
			position = this.insertSpaces(statStringArray , position, 2);
			
			//Fill in the rest of the buffer.
			while(position < initialBufferLength){
				statStringArray[position] = '\u0020';
				position++;
			}
			
			//Everything should be added now except the function name.
			String firstPart = new String(statStringArray);
			return firstPart + this.getMappingName();
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "GTDE01");
		}
		
		return "An error occured pocessing this string!";	
	}
	
	//User event interface.
	public String getUserEventName(){
		tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 2);
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
		this.insertDouble(1,0,inDouble);}
	
	public double getUserEventMinValue(){
		return this.getDouble(1,0);}
	
	public void setUserEventMaxValue(double inDouble){
		this.insertDouble(2,0,inDouble);}
	
	public double getUserEventMaxValue(){
		return this.getDouble(2,0);}
	
	public void setUserEventMeanValue(double inDouble){
		this.insertDouble(3,0,inDouble);}
	
	public double getUserEventMeanValue(){
		return this.getDouble(3,0);}
	
	public String getUserEventStatString(){
		try{
			int initialBufferLength = 80;
			int position = 0;
			char [] statStringArray = new char[initialBufferLength];
			char [] tmpArray;
			
			
			tmpArray = (Integer.toString(this.getUserEventNumberValue()).toCharArray());
			for(int i=0;i<tmpArray.length;i++){
				statStringArray[position] = tmpArray[i];
				position++;
			}
			position = this.insertSpaces(statStringArray , position, 2);
			
			DecimalFormat dF = new DecimalFormat();
			dF.applyPattern("0.######E0");
			tmpArray = (dF.format(this.getUserEventMaxValue())).toCharArray();
			for(int i=0;i<tmpArray.length;i++){
				statStringArray[position] = tmpArray[i];
				position++;
			}
			position = this.insertSpaces(statStringArray , position, 2);
			
			tmpArray = (dF.format(this.getUserEventMinValue())).toCharArray();
			for(int i=0;i<tmpArray.length;i++){
				statStringArray[position] = tmpArray[i];
				position++;
			}
			position = this.insertSpaces(statStringArray , position, 2);
			
			tmpArray = (dF.format(this.getUserEventMeanValue())).toCharArray();
			for(int i=0;i<tmpArray.length;i++){
				statStringArray[position] = tmpArray[i];
				position++;
			}
			position = this.insertSpaces(statStringArray , position, 2);
			
			//Fill in the rest of the buffer.
			while(position < initialBufferLength){
				statStringArray[position] = '\u0020';
				position++;
			}
			
			//Everything should be added now except the function name.
			String firstPart = new String(statStringArray);
			return firstPart + this.getUserEventName();
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "GTDE04");
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
	
	private void insertDouble(int listNumber, int dataValueLocation, double inDouble){
		try{
			switch(listNumber){
			case(1):
				doubleList1[dataValueLocation] = inDouble;
				break;
			case(2):
				doubleList2[dataValueLocation] = inDouble;
				break;			
			case(3):
				doubleList3[dataValueLocation] = inDouble;
				break;
			case(4):
				doubleList4[dataValueLocation] = inDouble;
				break;
			case(5):
				doubleList5[dataValueLocation] = inDouble;
				break;
			default:
				jRacy.systemError(null, null, "GTDE03");
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "GTDE04");
		}
	}
	
	private double getDouble(int listNumber, int dataValueLocation){
		try{
			switch(listNumber){
			case(1):
				return doubleList1[dataValueLocation];
				
			case(2):
				return doubleList2[dataValueLocation];			
			case(3):
				return doubleList3[dataValueLocation];
			case(4):
				return doubleList4[dataValueLocation];
			case(5):
				return doubleList5[dataValueLocation];
			default:
				jRacy.systemError(null, null, "GTDE05");
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "GTDE06");
		}
		
		return -1;
	}
	
	public void incrementStorage(){
		if(userElement)
			System.out.println("We are trying to increase storage on a user event object!");
		//Since all arrays are kept at the same size, just use one of them to get the current length.
		int currentLength = doubleList1.length;

		//can use a little space here ... space for speed! :-)
		double[] newArray1 = new double[currentLength+1];
		double[] newArray2 = new double[currentLength+1];
		double[] newArray3 = new double[currentLength+1];
		double[] newArray4 = new double[currentLength+1];
		double[] newArray5 = new double[currentLength+1];
		
		for(int i=0;i<currentLength;i++){
			newArray1[i] = doubleList1[i];
			newArray2[i] = doubleList2[i];
			newArray3[i] = doubleList3[i];
			newArray4[i] = doubleList4[i];
			newArray5[i] = doubleList5[i];
		}
		
		doubleList1 = newArray1;
		doubleList2 = newArray2;
		doubleList3 = newArray3;
		doubleList4 = newArray4;
		doubleList5 = newArray5;
	}
		
	
	//Instance data.
	
	private Trial trial = null;
	
	//Global Mapping reference.
	GlobalMapping globalMappingReference;
	
	//A global mapping element reference.
	GlobalMappingElement tmpGME;
	
	//Set if mapping exists on this thread.
	boolean mappingExists = false;
	
	//Mapping ID
	int mappingID;
	
	//Named data values.
	private double[] doubleList1 = new double[1];	//inclusive/minValue
	private double[] doubleList2 = new double[1];	//exclusive/maxValue
	private double[] doubleList3 = new double[1];	//inclusivePercent/meanValue
	private double[] doubleList4;					//exclusivePercent
	private int numberOfCalls = 0;
	private int numberOfSubRoutines = 0;
	private double[] doubleList5;
	int userEventNumberValue = 0;
	
	boolean userElement = false;
}




