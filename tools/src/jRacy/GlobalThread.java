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
	}
	
	//Rest of the public functions.
	
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
	
	public void setMaxInclusiveValue(double inDouble)
	{
		maxInclusiveValue = inDouble;
	}
	
	public double getMaxInclusiveValue()
	{
		return maxInclusiveValue;
	}
	
	public void setMaxExclusiveValue(double inDouble)
	{
		maxExclusiveValue = inDouble;
	}
	
	public double getMaxExclusiveValue()
	{
		return maxExclusiveValue;
	}
	
	public void setMaxInclusivePercentValue(double inDouble)
	{
		maxInclusivePercentValue = inDouble;
	}
	
	public double getMaxInclusivePercentValue()
	{
		return maxInclusivePercentValue;
	}
	
	public void setMaxExclusivePercentValue(double inDouble)
	{
		maxExclusivePercentValue = inDouble;
	}
	
	public double getMaxExclusivePercentValue()
	{
		return maxExclusivePercentValue;
	}
	
	
	//Instance data.
	Vector threadDataList;
	Vector userThreadDataList;
	
	//Max values on this thread.
	double maxInclusiveValue = 0;
	double maxExclusiveValue = 0;
	double maxInclusivePercentValue = 0;
	double maxExclusivePercentValue = 0;
	
}