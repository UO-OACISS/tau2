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

public class GlobalThreadDataElement implements Serializable 
{
	//Constructor.
	public GlobalThreadDataElement(ExperimentRun inExpRun, boolean userElement){	
		expRun = inExpRun;
		globalMappingReference = expRun.getGlobalMapping();
		mappingID = -1;
		
		if(!userElement)
			this.addDefaultToVectors();
		else
			this.addDefaultToVectorsUE();
	}
	
	public void addDefaultToVectors(){
		inclusiveValueList.add(new Double(0));
		exclusiveValueList.add(new Double(0));
		inclusivePercentValueList.add(new Double(0));
		exclusivePercentValueList.add(new Double(0));
		userSecPerCallList.add(new Double(0));
		
		tStatStringList.add(new String(""));
		
		userEventNumberValueList.add(new Integer(0));
		userEventMinValueList.add(new Double(0));
		userEventMaxValueList.add(new Double(0));
		userEventMeanValueList.add(new Double(0));
		userEventStatStringList.add(new String(""));
	}
	
	public void addDefaultToVectorsUE(){
		userEventNumberValueList.add(new Integer(0));
		userEventMinValueList.add(new Double(0));
		userEventMaxValueList.add(new Double(0));
		userEventMeanValueList.add(new Double(0));
		userEventStatStringList.add(new String(""));
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
	
	public void setInclusiveValue(int dataValueLocation, double inInclusiveValue){
		Double tmpDouble = new Double(inInclusiveValue);
		inclusiveValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getInclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) inclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setExclusiveValue(int dataValueLocation, double inExclusiveValue){
		Double tmpDouble = new Double(inExclusiveValue);
		exclusiveValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getExclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) exclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setInclusivePercentValue(int dataValueLocation, double inInclusivePercentValue){
		
		Double tmpDouble = new Double(inInclusivePercentValue);
		inclusivePercentValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getInclusivePercentValue(int dataValueLocation){
		Double tmpDouble = (Double) inclusivePercentValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setExclusivePercentValue(int dataValueLocation, double inExclusivePercentValue){
		Double tmpDouble = new Double(inExclusivePercentValue);
		exclusivePercentValueList.setElementAt(tmpDouble, dataValueLocation);}
	public double getExclusivePercentValue(int dataValueLocation){
		Double tmpDouble = (Double) exclusivePercentValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setNumberOfCalls(int inInt){
		numberOfCalls = inInt;}
	
	public int getNumberOfCalls(){
		return numberOfCalls;}
	
	public void setNumberOfSubRoutines(int inInt){
		numberOfSubRoutines = inInt;}
	
	public int getNumberOfSubRoutines(){
		return numberOfSubRoutines;}
	
	public void setUserSecPerCall(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		userSecPerCallList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getUserSecPerCall(int dataValueLocation){
		Double tmpDouble = (Double) userSecPerCallList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setTStatString(int dataValueLocation, String inString){
		tStatStringList.setElementAt(inString, dataValueLocation);}
	
	public String getTStatString(int dataValueLocation){
		return (String) tStatStringList.elementAt(dataValueLocation);}
	
	//User event interface.
	public String getUserEventName(){
		tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 2);
		return tmpGME.getMappingName();}
	
	public void setUserEventID(int inUserEventID){
		userEventID = inUserEventID;}
	
	public int getUserEventID(){
		return userEventID;}
	
	public void setUserEventNumberValue(int dataValueLocation, int inUserEventNumberValue){
		Integer tmpInt = new Integer(inUserEventNumberValue);
		userEventNumberValueList.setElementAt(tmpInt, dataValueLocation);}
	
	public int getUserEventNumberValue(int dataValueLocation){
		Integer tmpInt = (Integer) userEventNumberValueList.elementAt(dataValueLocation);
		return tmpInt.intValue();}
	
	public void setUserEventMinValue(int dataValueLocation, double inUserEventMinValue){
		Double tmpDouble = new Double(inUserEventMinValue);
		userEventMinValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getUserEventMinValue(int dataValueLocation){
		Double tmpDouble = (Double) userEventMinValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setUserEventMaxValue(int dataValueLocation, double inUserEventMaxValue){
		Double tmpDouble = new Double(inUserEventMaxValue);
		userEventMaxValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getUserEventMaxValue(int dataValueLocation){
		Double tmpDouble = (Double) userEventMaxValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setUserEventMeanValue(int dataValueLocation, double inUserEventMeanValue){
		Double tmpDouble = new Double(inUserEventMeanValue);
		userEventMeanValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getUserEventMeanValue(int dataValueLocation){
		Double tmpDouble = (Double) userEventMeanValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setUserEventStatString(int dataValueLocation, String inString){
		userEventStatStringList.setElementAt(inString, dataValueLocation);}
	
	public String getUserEventStatString(int dataValueLocation){
		return (String) userEventStatStringList.elementAt(dataValueLocation);}

	//Instance data.
	
	private ExperimentRun expRun = null;
	
	//Global Mapping reference.
	GlobalMapping globalMappingReference;
	
	//A global mapping element reference.
	GlobalMappingElement tmpGME;
	
	//Set if mapping exists on this thread.
	boolean mappingExists = false;
	
	//Mapping ID
	int mappingID;
	
	//Named data values.
	private Vector inclusiveValueList = new Vector();
	private Vector exclusiveValueList = new Vector();
	private Vector inclusivePercentValueList = new Vector();
	private Vector exclusivePercentValueList = new Vector();
	private int numberOfCalls = 0;
	private int numberOfSubRoutines = 0;
	private Vector userSecPerCallList = new Vector();
	
	//The total statics string.
	private Vector tStatStringList = new Vector();
	
	
	//User event section.
	private String userEventName;
	private int userEventID;
	private Vector userEventNumberValueList = new Vector();
	private Vector userEventMinValueList = new Vector();
	private Vector userEventMaxValueList = new Vector();
	private Vector userEventMeanValueList = new Vector();
	private Vector userEventStatStringList = new Vector();
}




