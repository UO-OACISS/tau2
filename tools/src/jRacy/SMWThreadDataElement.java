/* 
	SMWThreadDataElement.java

	Title:			jRacy
	Author:			Robert Bell
	Description:	This class holds the mapping data for one of the mappings on this thread.
					It also holds all the drawing information for this mapping.
*/

package jRacy;

import java.util.*;
import java.lang.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;

public class SMWThreadDataElement implements Comparable
{
	//Constructor.
	public SMWThreadDataElement(GlobalThreadDataElement inGTDEReference)
	{	
		gTDEReference = inGTDEReference;
		globalMappingReference = jRacy.staticSystemData.getGlobalMapping();
		
		value = 0;
		
		xBeginPosition = 0;
		xEndPosition = 0;
		yBeginPosition = 0;
		yEndPosition = 0;
		
		tDWXEndPosition = 0;
		tDWXBegPosition = 0;
		tDWYEndPosition = 0;
		tDWYBegPosition = 0;
		
		fDWXEndPosition = 0;
		fDWXBegPosition = 0;
		fDWYEndPosition = 0;
		fDWYBegPosition = 0;
		
		sortByMappingID = false;
		sortByName = true;
		sortByValue = false;
		sortByReverse = false;
	}
	
	//Rest of the public functions.
	public GlobalThreadDataElement getGTDE()
	{
		return gTDEReference;
	}
		
	public String getMappingName()
	{
		tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
		
		return tmpGME.getMappingName();
	}
	
	public void setMappingID(int inMappingID)
	{
		mappingID = inMappingID;
	}
	
	public int getMappingID()
	{
		return mappingID;
	}
	
	public Color getMappingColor()
	{
		tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
		return tmpGME.getMappingColor();
	}
	
	public boolean getMappingExists()
	{
		return gTDEReference.getMappingExists();
	}
	
	public boolean isGroupMember(int inGroupID)
	{
		return globalMappingReference.isGroupMember(mappingID, inGroupID, 0);
	}
	
	public double getInclusiveValue()
	{
		return gTDEReference.getInclusiveValue();
	}
	
	public double getExclusiveValue()
	{
		return gTDEReference.getExclusiveValue();
	}
	
	public double getInclusiveMicroValue()
	{
		return gTDEReference.getInclusiveMicroValue();
	}
	
	public double getExclusiveMicroValue()
	{
		return gTDEReference.getExclusiveMicroValue();
	}
	
	public double getInclusivePercentValue()
	{
		return gTDEReference.getInclusivePercentValue();
	}
	
	public double getExclusivePercentValue()
	{
		return gTDEReference.getExclusivePercentValue();
	}
	
	public String getTStatString()
	{
		return gTDEReference.getTStatString();
	}
	
	//User event interface.
	public String getUserEventName()
	{
		return gTDEReference.getUserEventName();
	}
	
	public int getUserEventID()
	{
		return gTDEReference.getUserEventID();
	}
	
	public int getUserEventNumberValue()
	{
		return gTDEReference.getUserEventNumberValue();
	}
	
	public double getUserEventMinValue()
	{
		return gTDEReference.getUserEventMinValue();
	}
	
	public double getUserEventMaxValue()
	{
		return gTDEReference.getUserEventMaxValue();
	}
	
	public double getUserEventMeanValue()
	{
		return gTDEReference.getUserEventMeanValue();
	}
	
	public String getUserEventStatString()
	{
		return gTDEReference.getUserEventStatString();
	}

	
	public int compareTo(Object inObject)
	{
		//Note that list will never call to compare against mapping id.  This
		//is because all the mappings are already sorted on the system.
		double tmpDouble = 0;
		
		if(sortByMappingID)
		{
			if(!sortByReverse)
				return (mappingID - (((SMWThreadDataElement)inObject).getMappingID()));
			else
				return ((((SMWThreadDataElement)inObject).getMappingID()) - mappingID);
		} 
				
		else if(sortByName)
		{
			if(!sortByReverse)
				return (this.getMappingName()).compareTo(((SMWThreadDataElement)inObject).getMappingName());
			else
				return (((SMWThreadDataElement) inObject).getMappingName()).compareTo(this.getMappingName());
		}
		
		//If here, means that we are in sort by value.
		tmpDouble = (value - (((SMWThreadDataElement)inObject).getValue()));
		if(tmpDouble < 0.00)
			if(!sortByReverse)
				return -1;
			else
				return 1;
		if(tmpDouble == 0.00)
			return 0;
		
		if(!sortByReverse)
				return 1;
			else
				return -1;
	}
	
	public void setValue(double inValue)
	{
		value = inValue;
	}
	
	public double getValue()
	{
		return value;
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
	
	public void setTDWDrawCoords(int inTDWXBeg, int inTDWXEnd, int inTDWYBeg, int inTDWYEnd)
	{
		tDWXBegPosition = inTDWXBeg;
		tDWXEndPosition = inTDWXEnd;
		tDWYBegPosition = inTDWYBeg;
		tDWYEndPosition = inTDWYEnd;
	}
		
	public int getTDWYBeg()
	{
		return tDWYBegPosition;
	}
	
	public int getTDWYEnd()
	{
		return tDWYEndPosition;
	}
	
	public int getTDWXBeg()
	{
		return tDWXBegPosition;
	}
	
	public int getTDWXEnd()
	{
		return tDWXEndPosition;
	}
	
	public void setFDWDrawCoords(int inFDWXBeg, int inFDWXEnd, int inFDWYBeg, int inFDWYEnd)
	{
		fDWXBegPosition = inFDWXBeg;
		fDWXEndPosition = inFDWXEnd;
		fDWYBegPosition = inFDWYBeg;
		fDWYEndPosition = inFDWYEnd;
	}
		
	public int getFDWYBeg()
	{
		return fDWYBegPosition;
	}
	
	public int getFDWYEnd()
	{
		return fDWYEndPosition;
	}
	
	public int getFDWXBeg()
	{
		return fDWXBegPosition;
	}
	
	public int getFDWXEnd()
	{
		return fDWXEndPosition;
	}
	
	public boolean getStatDrawnTo()
	{
		return statDrawnTo;
	}
	
	public void setStatDrawnTo(boolean inBoolean)
	{
		statDrawnTo = inBoolean;
	}
	
	public void setHighlighted(boolean inBool)
	{
		highlighted = inBool;
	}
	
	public boolean isHighlighted()
	{
		return highlighted;
	}
	
	public void setSortByMappingID()
	{
		sortByMappingID = true;
		sortByName = false;
		sortByValue = false;
	}
	
	public void setSortByName()
	{
		sortByMappingID = false;
		sortByName = true;
		sortByValue = false;
	}
	
	public void setSortByValue()
	{
		sortByMappingID = false;
		sortByName = false;
		sortByValue = true;
	}
	
	public void setSortByReverse(boolean inBool)
	{
		sortByReverse = inBool;
	}

	//Instance data.

	//Global Thread Data Element Reference.
	GlobalThreadDataElement gTDEReference;
	
	//Global Mapping reference.
	GlobalMapping globalMappingReference;
	
	//A global mapping element reference.
	GlobalMappingElement tmpGME;
	
	//Mapping ID
	int mappingID;

	//Named data values.
	double value;
	
	//Drawing coordinates for this thread data object.
	int xBeginPosition;
	int xEndPosition;
	int yBeginPosition;
	int yEndPosition;
	
	//Drawing coordinates the thread data window.
	int tDWXEndPosition;
	int tDWXBegPosition;
	int tDWYEndPosition;
	int tDWYBegPosition;
	
	//Drawing coordinates the mapping data window.
	int fDWXEndPosition;
	int fDWXBegPosition;
	int fDWYEndPosition;
	int fDWYBegPosition;
	
	boolean statDrawnTo;
	
	//Boolean indicating whether or not this object is highlighted.
	boolean highlighted = false;
	
	//
	boolean sortByMappingID;
	boolean sortByName;
	boolean sortByValue;
	boolean sortByReverse;
	
	boolean compareOnMappingName;
}