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
	public GlobalThreadDataElement()
	{	
		
		globalMappingReference = jRacy.staticSystemData.getGlobalMapping();
		
		inclusiveValue = 0;
		exclusiveValue = 0;
		inclusiveMicroValue = 0;
		exclusiveMicroValue = 0;
		inclusivePercentValue = 0;
		exclusivePercentValue = 0;
		
		functionID = -1;
		
		tStatString = null;
		
		userEventName = null;
		userEventNumberValue = 0;
		userEventMinValue = 0;
		userEventMaxValue = 0;
		userEventMeanValue = 0;
		userEventStatString = null;
	}
	
	//Rest of the public functions.
	public String getFunctionName()
	{
		tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(functionID);
		
		return tmpGME.getFunctionName();
	}
	
	public void setFunctionID(int inFunctionID)
	{
		functionID = inFunctionID;
	}
	
	public void setFunctionExists()
	{
		functionExists = true;
	}
	
	public boolean getFunctionExists()
	{
		return functionExists;
	}
	
	public int getFunctionID()
	{
		return functionID;
	}
	
	public void setInclusiveValue(double inInclusiveValue)
	{
		inclusiveValue = inInclusiveValue;
	}
	
	public double getInclusiveValue()
	{
		return inclusiveValue;
	}
	
	public void setExclusiveValue(double inExclusiveValue)
	{
		exclusiveValue = inExclusiveValue;
	}
	
	public double getExclusiveValue()
	{
		return exclusiveValue;
	}
	
	public void setInclusiveMicroValue(double inInclusiveMicroValue)
	{
		inclusiveMicroValue = inInclusiveMicroValue;
	}
	
	public double getInclusiveMicroValue()
	{
		return inclusiveMicroValue;
	}
	
	public void setExclusiveMicroValue(double inExclusiveMicroValue)
	{
		exclusiveMicroValue = inExclusiveMicroValue;
	}
	
	public double getExclusiveMicroValue()
	{
		return exclusiveMicroValue;
	}
	
	public void setInclusivePercentValue(double inInclusivePercentValue)
	{
		inclusivePercentValue = inInclusivePercentValue;
	}
	
	public double getInclusivePercentValue()
	{
		return inclusivePercentValue;
	}
	
	public void setExclusivePercentValue(double inExclusivePercentValue)
	{
		exclusivePercentValue = inExclusivePercentValue;
	}
	
	public double getExclusivePercentValue()
	{
		return exclusivePercentValue;
	}
	
	public void setTStatString(String inString)
	{
		tStatString = inString;
	}
	
	public String getTStatString()
	{
		return tStatString;
	}
	
	//User event interface.
	public void setUserEventName(String inUserEventName)
	{
		userEventName = inUserEventName;
	}
	
	public String getUserEventName()
	{
		return userEventName;
	}
	
	public void setUserEventID(int inUserEventID)
	{
		userEventID = inUserEventID;
	}
	
	public int getUserEventID()
	{
		return userEventID;
	}
	
	public void setUserEventNumberValue(int inUserEventNumberValue)
	{
		userEventNumberValue = inUserEventNumberValue;
	}
	
	public int getUserEventNumberValue()
	{
		return userEventNumberValue;
	}
	
	public void setUserEventMinValue(double inUserEventMinValue)
	{
		userEventMinValue = inUserEventMinValue;
	}
	
	public double getUserEventMinValue()
	{
		return userEventMinValue;
	}
	
	public void setUserEventMaxValue(double inUserEventMaxValue)
	{
		userEventMaxValue = inUserEventMaxValue;
	}
	
	public double getUserEventMaxValue()
	{
		return userEventMaxValue;
	}
	
	public void setUserEventMeanValue(double inUserEventMeanValue)
	{
		userEventMeanValue = inUserEventMeanValue;
	}
	
	public double getUserEventMeanValue()
	{
		return userEventMeanValue;
	}
	
	public void setUserEventStatString(String inString)
	{
		userEventStatString = inString;
	}
	
	public String getUserEventStatString()
	{
		return userEventStatString;
	}

	//Instance data.
	
	//Global Mapping reference.
	GlobalMapping globalMappingReference;
	
	//A global mapping element reference.
	GlobalMappingElement tmpGME;
	
	//Set if function exists on this thread.
	boolean functionExists = false;
	
	//Function ID
	int functionID;
	
	//Named data values.
	private double inclusiveValue;
	private double exclusiveValue;
	private double inclusivePercentValue;
	private double exclusivePercentValue;
	private double inclusiveMicroValue;
	private double exclusiveMicroValue;
	
	//The total statics string.
	String tStatString;
	
	
	//User event section.
	private String userEventName;
	private int userEventID;
	private int userEventNumberValue;
	private double userEventMinValue;
	private double userEventMaxValue;
	private double userEventMeanValue;
	String userEventStatString;
}




