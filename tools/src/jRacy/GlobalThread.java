/* 
	GlobalThread.java
	
	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;

import java.util.*;
import java.io.*;

public class GlobalThread implements Serializable 
{
	//Constructor.
	public GlobalThread()
	{
		threadDataList = new Vector();
		userThreadDataList = new Vector();
		
		this.addDefaultToVectors();
	}
	
	//Rest of the public functions.
	
	public void addDefaultToVectors(){
		maxInclusiveValueList.add(new Double(0));
		maxExclusiveValueList.add(new Double(0));
		maxInclusivePercentValueList.add(new Double(0));
		maxExclusivePercentValueList.add(new Double(0));
		maxUserSecPerCallList.add(new Double(0));
	}
	
	
	//The following function adds a thread data element to
	//the threadDataList
	void addThreadDataElement(GlobalThreadDataElement inGTDE)
	{
		threadDataList.addElement(inGTDE);
	}
	
	void addThreadDataElement(GlobalThreadDataElement inGTDE, int inPosition)
	{
		threadDataList.setElementAt(inGTDE, inPosition);
	}
	
	//The following function adds a thread data element to
	//the userThreadDataList
	void addUserThreadDataElement(GlobalThreadDataElement inGTDE)
	{
		userThreadDataList.addElement(inGTDE);
	}
	
	Vector getThreadDataList()
	{
		return threadDataList;
	}
	
	Vector getUserThreadDataList()
	{
		return userThreadDataList;
	}
	
	public void setMaxInclusiveValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		maxInclusiveValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMaxInclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) maxInclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxExclusiveValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		maxExclusiveValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMaxExclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) maxExclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxInclusivePercentValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		maxInclusivePercentValueList.setElementAt(tmpDouble, dataValueLocation);}
	
	public double getMaxInclusivePercentValue(int dataValueLocation){
		Double tmpDouble = (Double) maxInclusivePercentValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxExclusivePercentValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
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

	//Max mean values.
	private Vector maxInclusiveValueList = new Vector();
	private Vector maxExclusiveValueList = new Vector();
	private Vector maxInclusivePercentValueList = new Vector();
	private Vector maxExclusivePercentValueList = new Vector();
	double totalExclusiveValue = 0;
	double totalInclusiveValue = 0;
	private int maxNumberOfCalls = 0;
	private int maxNumberOfSubRoutines = 0;
	private Vector maxUserSecPerCallList = new Vector();
}