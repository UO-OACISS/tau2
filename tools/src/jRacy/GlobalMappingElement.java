/* 
	GlobalMappingElement.java

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

public class GlobalMappingElement implements Serializable 
{
	//Constructors.
	public GlobalMappingElement(ExperimentRun inExpRun)
	{
		expRun = inExpRun;
		
		mappingName = null;
		globalID = -1;
		groups = new int[10];
		numberOfGroups = 0;
		colorFlag = false;
		genericMappingColor = null;
		specificMappingColor = null;
		
		this.addDefaultToVectors();
	}
	
	public void addDefaultToVectors(){
		maxInclusiveValueList.add(new Double(0));
		maxExclusiveValueList.add(new Double(0));
		maxInclusivePercentValueList.add(new Double(0));
		maxExclusivePercentValueList.add(new Double(0));
		maxUserSecPerCallList.add(new Double(0));
		
		maxUserEventNumberValueList.add(new Integer(0));
		maxUserEventMinValueList.add(new Double(0));
		maxUserEventMaxValueList.add(new Double(0));
		maxUserEventMeanValueList.add(new Double(0));
		
		meanInclusiveValueList.add(new Double(0));
		meanExclusiveValueList.add(new Double(0));
		meanInclusivePercentValueList.add(new Double(0));
		meanExclusivePercentValueList.add(new Double(0));
		meanUserSecPerCallList.add(new Double(0));
	
	
		totalInclusiveValueList.add(new Double(0));
		totalExclusiveValueList.add(new Double(0));
		totalInclusivePercentValueList.add(new Double(0));
		totalExclusivePercentValueList.add(new Double(0));
		
		meanTotalStatStringList.add(new String(""));
		totalTotalStatStringList.add(new String(""));
	}
	
	public void setMappingName(String inMappingName)
	{
		mappingName = inMappingName;
	}
	
	public String getMappingName()
	{
		return mappingName;
	}
	
	public void setGlobalID(int inGlobalID)
	{
		globalID = inGlobalID;
	}
	
	public int getGlobalID()
	{
		return globalID;
	}
	
	public boolean addGroup(int inGroupID)
	{
		
		if(numberOfGroups < 10)
		{
			groups[numberOfGroups] = inGroupID;
			numberOfGroups++;
			
			return true;
		}
		
		return false;
	}
	
	public boolean isGroupMember(int inGroupID)
	{
		GlobalMapping tmpGM = expRun.getGlobalMapping();
		
		boolean tmpBool = tmpGM.getIsAllExceptGroupOn();
		boolean tmpBoolResult = false;
		
		for(int i=0;i<numberOfGroups;i++)
		{
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
	
	public void setColorFlag(boolean inBoolean)
	{
		colorFlag = inBoolean;
	}
	
	public boolean isColorFlagSet()
	{
		return colorFlag;
	}
	
	public void setGenericColor(Color inColor)
	{
		genericMappingColor = inColor;
	}
	
	public void setSpecificColor(Color inColor)
	{
		specificMappingColor = inColor;
	}
	
	public Color getMappingColor()
	{
		if(colorFlag)
			return specificMappingColor;
		else
			return genericMappingColor;
	}
	
	public Color getGenericColor()
	{
		return genericMappingColor;
	}
	
	public void setMaxInclusiveValue(int dataValueLocation, double inInclusiveValue){
		Double tmpDouble = new Double(inInclusiveValue);
		maxInclusiveValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMaxInclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) maxInclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxExclusiveValue(int dataValueLocation, double inExclusiveValue){
		Double tmpDouble = new Double(inExclusiveValue);
		maxExclusiveValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMaxExclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) maxExclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxInclusivePercentValue(int dataValueLocation, double inInclusivePercentValue){
		
		Double tmpDouble = new Double(inInclusivePercentValue);
		maxInclusivePercentValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMaxInclusivePercentValue(int dataValueLocation){
	
		Double tmpDouble = (Double) maxInclusivePercentValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxExclusivePercentValue(int dataValueLocation, double inExclusivePercentValue){
	
		Double tmpDouble = new Double(inExclusivePercentValue);
		maxExclusivePercentValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMaxExclusivePercentValue(int dataValueLocation){
		
		Double tmpDouble = (Double) maxExclusivePercentValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxNumberOfCalls(int inInt){
		maxNumberOfCalls = inInt;}
	
	public int getMaxNumberOfCalls(){
		return maxNumberOfCalls;}
	
	public void setMaxNumberOfSubRoutines(int inInt){
		maxNumberOfSubRoutines = inInt;}
	
	public int getMaxNumberOfSubRoutines(){
		return maxNumberOfSubRoutines;}
	
	public void setMaxUserSecPerCall(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		maxUserSecPerCallList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMaxUserSecPerCall(int dataValueLocation){
		Double tmpDouble = (Double) maxUserSecPerCallList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
		
	//User event section.
	public void setMaxUserEventNumberValue(int dataValueLocation, int inUserEventNumberValue){
		Integer tmpInt = new Integer(inUserEventNumberValue);
		maxUserEventNumberValueList.setElementAt(tmpInt, dataValueLocation);}
	
	public int getMaxUserEventNumberValue(int dataValueLocation){
		Integer tmpInt = (Integer) maxUserEventNumberValueList.elementAt(dataValueLocation);
		return tmpInt.intValue();}
	
	public void setMaxUserEventMinValue(int dataValueLocation, double inUserEventMinValue){
		
		Double tmpDouble = new Double(inUserEventMinValue);
		maxUserEventMinValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMaxUserEventMinValue(int dataValueLocation){
		Double tmpDouble = (Double) maxUserEventMinValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxUserEventMaxValue(int dataValueLocation, double inUserEventMaxValue){
		Double tmpDouble = new Double(inUserEventMaxValue);
		maxUserEventMaxValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMaxUserEventMaxValue(int dataValueLocation){
		Double tmpDouble = (Double) maxUserEventMaxValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxUserEventMeanValue(int dataValueLocation, double inUserEventMeanValue){
		Double tmpDouble = new Double(inUserEventMeanValue);
		maxUserEventMeanValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMaxUserEventMeanValue(int dataValueLocation){
		Double tmpDouble = (Double) maxUserEventMeanValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	
	
	//Mean section.
	public void setMeanInclusiveValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		meanInclusiveValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMeanInclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) meanInclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMeanExclusiveValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		meanExclusiveValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMeanExclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) meanExclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMeanInclusivePercentValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		meanInclusivePercentValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMeanInclusivePercentValue(int dataValueLocation){
		Double tmpDouble = (Double) meanInclusivePercentValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMeanExclusivePercentValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		meanExclusivePercentValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMeanExclusivePercentValue(int dataValueLocation){
		Double tmpDouble = (Double) meanExclusivePercentValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMeanNumberOfCalls(double inDouble){
		meanNumberOfCalls = inDouble;}
	
	public double getMeanNumberOfCalls(){
		return meanNumberOfCalls;}
	
	public void setMeanNumberOfSubRoutines(double inDouble){
		meanNumberOfSubRoutines = inDouble;}
	
	public double getMeanNumberOfSubRoutines(int dataValueLocation){
		return meanNumberOfSubRoutines;}
	
	public void setMeanUserSecPerCall(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		meanUserSecPerCallList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMeanUserSecPerCall(int dataValueLocation){
		Double tmpDouble = (Double) meanUserSecPerCallList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	
	
	public void setTotalInclusiveValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		totalInclusiveValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getTotalInclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) totalInclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setTotalExclusiveValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		totalExclusiveValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getTotalExclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) totalExclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setTotalInclusivePercentValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		totalInclusivePercentValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getTotalInclusivePercentValue(int dataValueLocation){
		Double tmpDouble = (Double) totalInclusivePercentValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setTotalExclusivePercentValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		totalExclusivePercentValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getTotalExclusivePercentValue(int dataValueLocation){
		Double tmpDouble = (Double) totalExclusivePercentValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	
	
	//Stat Strings.
	public void setMeanTotalStatString(int dataValueLocation, String inString){
		meanTotalStatStringList.setElementAt(inString, dataValueLocation);}
	
	public String getMeanTotalStatString(int dataValueLocation){
		return (String) meanTotalStatStringList.elementAt(dataValueLocation);}
		
	public void setTotalTotalStatString(int dataValueLocation, String inString){
		totalTotalStatStringList.setElementAt(inString, dataValueLocation);}
	
	public String getTotalTotalStatString(int dataValueLocation){
		return (String) totalTotalStatStringList.elementAt(dataValueLocation);}
	

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
	
	
	
	//Instance elmements.
	
	private ExperimentRun expRun = null;
	
	//Global Mapping reference.
	String mappingName;
	int globalID;			//Global ID for this mapping.
	
	int[] groups;
	int numberOfGroups;
	
	//Color Settings.
	boolean colorFlag;
	Color genericMappingColor;
	Color specificMappingColor;
	
	private Vector maxInclusiveValueList = new Vector();
	private Vector maxExclusiveValueList = new Vector();
	private Vector maxInclusivePercentValueList = new Vector();
	private Vector maxExclusivePercentValueList = new Vector();
	private int maxNumberOfCalls = 0;
	private int maxNumberOfSubRoutines = 0;
	private Vector maxUserSecPerCallList = new Vector();
	
	private Vector maxUserEventNumberValueList = new Vector();
	private Vector maxUserEventMinValueList = new Vector();
	private Vector maxUserEventMaxValueList = new Vector();
	private Vector maxUserEventMeanValueList = new Vector();
	
	private Vector meanInclusiveValueList = new Vector();
	private Vector meanExclusiveValueList = new Vector();
	private Vector meanInclusivePercentValueList = new Vector();
	private Vector meanExclusivePercentValueList = new Vector();
	private double meanNumberOfCalls = 0;
	private double meanNumberOfSubRoutines = 0;
	private Vector meanUserSecPerCallList = new Vector();
	
	
	private Vector totalInclusiveValueList = new Vector();
	private Vector totalExclusiveValueList = new Vector();
	private Vector totalInclusivePercentValueList = new Vector();
	private Vector totalExclusivePercentValueList = new Vector();
	
	private Vector meanTotalStatStringList = new Vector();
	private Vector totalTotalStatStringList = new Vector();
	
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