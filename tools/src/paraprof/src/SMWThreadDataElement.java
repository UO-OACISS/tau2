/* 
   SMWThreadDataElement.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  This class holds the mapping data for one of the mappings on this thread.
   It also holds all the drawing information for this mapping.
*/

package paraprof;

import java.util.*;
import java.lang.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;

public class SMWThreadDataElement implements Comparable{
    //Constructor.
    public SMWThreadDataElement(Trial inTrial, GlobalThreadDataElement inGTDEReference){ 
	trial = inTrial;
    
	gTDEReference = inGTDEReference;
	globalMappingReference = trial.getGlobalMapping();
    
	value = 0;
    
	xBeginPosition = 0;
	xEndPosition = 0;
	yBeginPosition = 0;
	yEndPosition = 0;
    
	tDWXEndPosition = 0;
	tDWXBegPosition = 0;
	tDWYEndPosition = 0;
	tDWYBegPosition = 0;
    
	mDWXEndPosition = 0;
	mDWXBegPosition = 0;
	mDWYEndPosition = 0;
	mDWYBegPosition = 0;
    
    
	int sortSetting = 0;
    
	sortByMappingID = false;
	sortByName = true;
	sortByValue = false;
	sortByReverse = false;
    }
  
    //Rest of the public functions.
    public GlobalThreadDataElement getGTDE(){
	return gTDEReference;
    }
    
    public String getMappingName(){
	tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
	return tmpGME.getMappingName();
    }
  
    public void setMappingID(int inMappingID){
	mappingID = inMappingID;}
  
    public int getMappingID(){
	return mappingID;}
  
    public Color getMappingColor(){
	tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
	return tmpGME.getMappingColor();
    }
  
    public boolean getMappingExists(){
	return gTDEReference.getMappingExists();}
  
    public boolean isGroupMember(int inGroupID){
	return globalMappingReference.isGroupMember(mappingID, inGroupID, 0);}

    public boolean isCallPathObject(){
	tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
	return tmpGME.isCallPathObject();
    }

    public Vector getParents(){
	return gTDEReference.getParents();}

    public Vector getChildren(){
	return gTDEReference.getChildren();}

    public ListIterator getParentsIterator(){
	return gTDEReference.getParentsIterator();}

    public ListIterator getChildrenIterator(){
 	return gTDEReference.getChildrenIterator();}

    public ListIterator getCallPathIDParents(int id){
	return gTDEReference.getCallPathIDParents(id);}

    public ListIterator getCallPathIDChildren(int id){
	return gTDEReference.getCallPathIDChildren(id);}
  
    public double getInclusiveValue(){
	return gTDEReference.getInclusiveValue(trial.getCurValLoc());}
  
    public double getExclusiveValue(){
	return gTDEReference.getExclusiveValue(trial.getCurValLoc());}
  
    public double getInclusivePercentValue(){
	return gTDEReference.getInclusivePercentValue(trial.getCurValLoc());}
  
    public double getExclusivePercentValue(){
	return gTDEReference.getExclusivePercentValue(trial.getCurValLoc());}
  
    public int getNumberOfCalls(){
	return gTDEReference.getNumberOfCalls();}
  
    public int getNumberOfSubRoutines(){
	return gTDEReference.getNumberOfSubRoutines();}
  
    public double getUserSecPerCall(){
	return gTDEReference.getUserSecPerCall(trial.getCurValLoc());}
  
    public String getTStatString(int type){
	return gTDEReference.getTStatString(type, trial.getCurValLoc());}
  
    //User event interface.
    public String getUserEventName(){
	tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 2);
    	return tmpGME.getMappingName();
    }
  
    public int getUserEventID(){
	return gTDEReference.getUserEventID();}
  
    public Color getUserEventMappingColor(){
	tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 2);
	return tmpGME.getMappingColor();
    }
  
    public int getUserEventNumberValue(){
	return gTDEReference.getUserEventNumberValue();}
  
    public double getUserEventMinValue(){
	return gTDEReference.getUserEventMinValue();}
  
    public double getUserEventMaxValue(){
	return gTDEReference.getUserEventMaxValue();}
  
    public double getUserEventMeanValue(){
	return gTDEReference.getUserEventMeanValue();}
  
    public String getUserEventStatString(){
	return gTDEReference.getUserEventStatString();}

    /*
      1 FIdD FunctionID
      2 FIdA FunctionID
      3 ND Name
      4 NA Name
      5 MD Millisec
      6 MA Millisec
    */
  
  
    public int compareTo(Object inObject){
    
	double tmpDouble = 0;
    
	switch(sortSetting){
	case(1):
	    return ((((SMWThreadDataElement)inObject).getMappingID()) - mappingID);
	case(2):
	    return (mappingID - (((SMWThreadDataElement)inObject).getMappingID()));
	case(3):
	    return (((SMWThreadDataElement) inObject).getMappingName()).compareTo(this.getMappingName());
	case(4):
	    return (this.getMappingName()).compareTo(((SMWThreadDataElement)inObject).getMappingName());
	case(5):
	    tmpDouble = (value - (((SMWThreadDataElement)inObject).getValue()));
	    if(tmpDouble < 0.00)
		return 1;
	    else if(tmpDouble == 0.00)
		return 0;
	    else
		return -1;
	case(6):
	    tmpDouble = (value - (((SMWThreadDataElement)inObject).getValue()));
	    if(tmpDouble < 0.00)
		return -1;
	    else if(tmpDouble == 0.00)
		return 0;
	    else
		return 1;
          
	default:
	    return 0;
	}
    }
  
    public int testCompareTo(Object inObject){
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
  
    public void setValue(double inValue){
	value = inValue;}
  
    public double getValue(){
	return value;}
  
    public void setDrawCoords(int inXBeg, int inXEnd, int inYBeg, int inYEnd){
	xBeginPosition = inXBeg;
	xEndPosition = inXEnd;
	yBeginPosition = inYBeg;
	yEndPosition = inYEnd;
    }
  
    public int getXBeg(){
	return xBeginPosition;}
  
    public int getXEnd(){
	return xEndPosition;}
  
    public int getYBeg(){
	return yBeginPosition;}
  
    public int getYEnd(){
	return yEndPosition;}
  
    public void setTDWDrawCoords(int inTDWXBeg, int inTDWXEnd, int inTDWYBeg, int inTDWYEnd){
	tDWXBegPosition = inTDWXBeg;
	tDWXEndPosition = inTDWXEnd;
	tDWYBegPosition = inTDWYBeg;
	tDWYEndPosition = inTDWYEnd;
    }
    
    public int getTDWYBeg(){
	return tDWYBegPosition;}
  
    public int getTDWYEnd(){
	return tDWYEndPosition;}
  
    public int getTDWXBeg(){
	return tDWXBegPosition;}
  
    public int getTDWXEnd(){
	return tDWXEndPosition;}
  
    public void setMDWDrawCoords(int inMDWXBeg, int inMDWXEnd, int inMDWYBeg, int inMDWYEnd){
	mDWXBegPosition = inMDWXBeg;
	mDWXEndPosition = inMDWXEnd;
	mDWYBegPosition = inMDWYBeg;
	mDWYEndPosition = inMDWYEnd;
    }
    
    public int getMDWYBeg(){
	return mDWYBegPosition;}
  
    public int getMDWYEnd(){
	return mDWYEndPosition;}
  
    public int getMDWXBeg(){
	return mDWXBegPosition;}
  
    public int getMDWXEnd(){
	return mDWXEndPosition;}
  
    //User Event Window.
    public void setUEWDrawCoords(int inUEWXBeg, int inUEWXEnd, int inUEWYBeg, int inUEWYEnd){
	uEWXBegPosition = inUEWXBeg;
	uEWXEndPosition = inUEWXEnd;
	uEWYBegPosition = inUEWYBeg;
	uEWYEndPosition = inUEWYEnd;
    }
    
    public int getUEWYBeg(){
	return uEWYBegPosition;}
  
    public int getUEWYEnd(){
	return uEWYEndPosition;}
  
    public int getUEWXBeg(){
	return uEWXBegPosition;}
  
    public int getUEWXEnd(){
	return uEWXEndPosition;}
  
    public boolean getStatDrawnTo(){
	return statDrawnTo;}
  
    public void setStatDrawnTo(boolean inBoolean){
	statDrawnTo = inBoolean;}
  
    public void setHighlighted(boolean inBool){
	highlighted = inBool;}
  
    public boolean isHighlighted(){
	return highlighted;}
  
  
    public void setSortSetting(int inInt){
	sortSetting = inInt;}
  
    public void setSortByMappingID(){
	sortByMappingID = true;
	sortByName = false;
	sortByValue = false;
    }
  
    public void setSortByName(){
	sortByMappingID = false;
	sortByName = true;
	sortByValue = false;
    }
  
    public void setSortByValue(){
	sortByMappingID = false;
	sortByName = false;
	sortByValue = true;
    }
  
    public void setSortByReverse(boolean inBool){
	sortByReverse = inBool;}

    //Instance data.
  
    private Trial trial = null;

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
    int mDWXEndPosition;
    int mDWXBegPosition;
    int mDWYEndPosition;
    int mDWYBegPosition;
  
    //Drawing coordinates the mapping data window.
    int uEWXEndPosition;
    int uEWXBegPosition;
    int uEWYEndPosition;
    int uEWYBegPosition;
  
    boolean statDrawnTo;
  
    //Boolean indicating whether or not this object is highlighted.
    boolean highlighted = false;
  
    //
    int sortSetting;
    boolean sortByMappingID;
    boolean sortByName;
    boolean sortByValue;
    boolean sortByReverse;
  
    boolean compareOnMappingName;
}
