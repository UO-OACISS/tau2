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
	public GlobalMappingElement()
	{
		mappingName = null;
		globalID = -1;
		groups = new int[10];
		numberOfGroups = 0;
		colorFlag = false;
		groupColorFlag = false;
		genericMappingColor = null;
		specificMappingColor = null;
		Color groupMappingColor = null;
		
		meanExclusiveValue = 0.0;
		totalExclusiveValue = 0.0;
		meanExclusivePercentValue = 0.0;
		totalExclusivePercentValue = 0.0;
		
		meanInclusiveValue = 0.0;
		totalInclusiveValue = 0.0;
		meanInclusivePercentValue = 0.0;
		totalInclusivePercentValue = 0.0;
		
		meanTotalStatString = null;
		totalTotalStatString = null;
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
		GlobalMapping tmpGM = jRacy.staticSystemData.getGlobalMapping();
		
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
	
	public void setGroupColorFlag(boolean inBoolean)
	{
		groupColorFlag = inBoolean;
	}
	
	public boolean isColorFlagSet()
	{
		return colorFlag;
	}
	
	public boolean isGroupColorFlagSet()
	{
		return groupColorFlag;
	}
	
	public void setGenericColor(Color inColor)
	{
		genericMappingColor = inColor;
	}
	
	public void setSpecificColor(Color inColor)
	{
		specificMappingColor = inColor;
	}
	
	public void setGroupColor(Color inColor)
	{
		groupMappingColor = inColor;
	}
	
	public Color getMappingColor()
	{
		//Group coloring takes priority.
		if(groupColorFlag)
			return groupMappingColor;
		else if(colorFlag)
			return specificMappingColor;
		else
			return genericMappingColor;
	}
	
	public Color getGenericColor()
	{
		return genericMappingColor;
	}
	
	//Exclusive part.	
	public void setMeanExclusiveValue(double inMeanExclusiveValue)
	{
		meanExclusiveValue = inMeanExclusiveValue;
	}
	
	public double getMeanExclusiveValue()
	{
		return meanExclusiveValue;
	}
	
	public void setMeanExclusivePercentValue(double inMeanExclusivePercentValue)
	{
		meanExclusivePercentValue = inMeanExclusivePercentValue;
	}
	
	public double getMeanExclusivePercentValue()
	{
		return meanExclusivePercentValue;
	}
	
	public void setTotalExclusiveValue(double inTotalExclusiveValue)
	{
		totalExclusiveValue = inTotalExclusiveValue;
	}
	
	public double getTotalExclusiveValue()
	{
		return totalExclusiveValue;
	}
	
	public void setTotalExclusivePercentValue(double inTotalExclusivePercentValue)
	{
		totalExclusivePercentValue = inTotalExclusivePercentValue;
	}
	
	public double getTotalExclusivePercentValue(double inTotalExclusivePercentValue)
	{
		return totalExclusivePercentValue;
	}
	
	
	//Inclusive part.
	
	public void setMeanInclusiveValue(double inMeanInclusiveValue)
	{
		meanInclusiveValue = inMeanInclusiveValue;
	}
	
	public double getMeanInclusiveValue()
	{
		return meanInclusiveValue;
	}
	
	public void setMeanInclusivePercentValue(double inMeanInclusivePercentValue)
	{
		meanInclusivePercentValue = inMeanInclusivePercentValue;
	}
	
	public double getMeanInclusivePercentValue()
	{
		return meanInclusivePercentValue;
	}
	
	public void setTotalInclusiveValue(double inTotalInclusiveValue)
	{
		totalInclusiveValue = inTotalInclusiveValue;
	}
	
	public double getTotalInclusiveValue()
	{
		return totalInclusiveValue;
	}
	
	public void setTotalInclusivePercentValue(double inTotalInclusivePercentValue)
	{
		totalInclusivePercentValue = inTotalInclusivePercentValue;
	}
	
	public double getTotalInclusivePercentValue()
	{
		return totalInclusivePercentValue;
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
	
	
	public void setMaxInclusiveValue(double inMaxInclusiveValue)
	{
		maxInclusiveValue = inMaxInclusiveValue;
	}
	
	public double getMaxInclusiveValue()
	{
		return maxInclusiveValue;
	}
	
	public void setMaxExclusiveValue(double inMaxExclusiveValue)
	{
		maxExclusiveValue = inMaxExclusiveValue;
	}
	
	public double getMaxExclusiveValue()
	{
		return maxExclusiveValue;
	}
	
	
	public void setMaxInclusiveMicroValue(double inMaxInclusiveMicroValue)
	{
		maxInclusiveMicroValue = inMaxInclusiveMicroValue;
	}
	
	public double getMaxInclusiveMicroValue()
	{
		return maxInclusiveMicroValue;
	}
	
	public void setMaxExclusiveMicroValue(double inMaxExclusiveMicroValue)
	{
		maxExclusiveMicroValue = inMaxExclusiveMicroValue;
	}
	
	public double getMaxExclusiveMicroValue()
	{
		return maxExclusiveMicroValue;
	}
	
	public void setMaxInclusivePercentValue(double inMaxInclusivePercentValue)
	{
		maxInclusivePercentValue = inMaxInclusivePercentValue;
	}
	
	public double getMaxInclusivePercentValue()
	{
		return maxInclusivePercentValue;
	}
	
	public void setMaxExclusivePercentValue(double inMaxExclusivePercentValue)
	{
		maxExclusivePercentValue = inMaxExclusivePercentValue;
	}
	
	public double getMaxExclusivePercentValue()
	{
		return maxExclusivePercentValue;
	}
	
	
	//Total stat strings.
	
	public void setMeanTotalStatString(String inMeanTotalStatString)
	{
		meanTotalStatString = inMeanTotalStatString;
	}
	
	public String getMeanTotalStatString()
	{
		return meanTotalStatString;
	}
	
	public void setTotalTotalStatString(String inTotalTotalStatString)
	{
		totalTotalStatString = inTotalTotalStatString;
	}
	
	public String getTotalTotalStatString()
	{
		return totalTotalStatString;
	}

	//Instance elmements.
	
	//Global Mapping reference.
	String mappingName;
	int globalID;			//Global ID for this mapping.
	
	int[] groups;
	int numberOfGroups;
	
	//Color Settings.
	boolean colorFlag;
	boolean groupColorFlag;
	Color genericMappingColor;
	Color specificMappingColor;
	Color groupMappingColor;
	
	double maxInclusiveValue = 0;
	double maxExclusiveValue = 0;
	double maxInclusiveMicroValue = 0;
	double maxExclusiveMicroValue = 0;
	double maxInclusivePercentValue = 0;
	double maxExclusivePercentValue = 0;
	
	double meanExclusiveValue;
	double totalExclusiveValue;
	double meanExclusivePercentValue;
	double totalExclusivePercentValue;
	
	double meanInclusiveValue;
	double totalInclusiveValue;
	double meanInclusivePercentValue;
	double totalInclusivePercentValue;
	
	//Drawing coordinates for this Global mapping element.
	int xBeginPosition;
	int xEndPosition;
	int yBeginPosition;
	int yEndPosition;
	
	boolean meanValuesSet = false;
	
	String meanTotalStatString;
	String totalTotalStatString;
}