/* 
   SMWThreadDataElement.java

   Title:      ParaProf
   Author:     Robert Bell
   Description: The primary functions of this class are:
                1)Pass data calls onto the objects which contain function
		  userevent, mean, and other data.
		2)Implement the Comparable interface to allow it to be sorted.
		3)Hold drawing information.

		Thus, it can be considered loosly as representing a particular object
		that will be drawn onto the screen at some point. It is not set up to
		represent MULTIPLE occurrences of drawing or sorting information. That is,
		it can hold only one set of drawing and sorting data at a time. Different
		windows must create their own instances of this object to avoid conflicts.

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
    public SMWThreadDataElement(ParaProfTrial trial, int nodeID, int contextID, int threadID, Object obj){ 
	
	this.trial = trial;
	this.nodeID = nodeID;
	this.contextID = contextID;
	this.threadID = threadID;
	//At present, obj should either be of type GlobalThreadDataElement or
	//GlobalMappingElement.
	if(obj instanceof GlobalThreadDataElement){
	    this.globalThreadDataElement = (GlobalThreadDataElement) obj;
	    this.globalMapping = trial.getGlobalMapping();
	    if(globalThreadDataElement.userevent())
		this.globalMappingElement = globalMapping.getGlobalMappingElement(globalThreadDataElement.getMappingID(), 2);
	    else
		this.globalMappingElement = globalMapping.getGlobalMappingElement(globalThreadDataElement.getMappingID(), 0);
	    
	}
	else if(obj instanceof GlobalMappingElement){
	    globalMapping = trial.getGlobalMapping();
	    this.globalMappingElement = (GlobalMappingElement) obj;
	}
	else{
	    UtilFncs.systemError(null, null, "Unexpected object type - SMWTDE value: " + obj.getClass().getName());
	}
    }
  
    //Rest of the public functions.
    public int getNodeID(){
	return nodeID;}

    public int getContextID(){
	return contextID;}

    public int getThreadID(){
	return threadID;}

    public GlobalThreadDataElement getGTDE(){
	return globalThreadDataElement;}
    
    public String getMappingName(){
	return globalMappingElement.getMappingName();}
  
    public int getMappingID(){
	return globalMappingElement.getMappingID();}
  
    public Color getColor(){
	return globalMappingElement.getColor();}
  
    public boolean getMappingExists(){
	return globalThreadDataElement.getMappingExists();}
  
    public boolean isGroupMember(int groupID){
	return globalMapping.isGroupMember(this.getMappingID(), groupID, 0);}

    public boolean isCallPathObject(){
	return globalMappingElement.isCallPathObject();}

    //####################################
    //Function interface.
    //####################################
    public Vector getParents(){
	return globalThreadDataElement.getParents();}

    public Vector getChildren(){
	return globalThreadDataElement.getChildren();}

    public ListIterator getParentsIterator(){
	return globalThreadDataElement.getParentsIterator();}

    public ListIterator getChildrenIterator(){
 	return globalThreadDataElement.getChildrenIterator();}

    public ListIterator getCallPathIDParents(int id){
	return globalThreadDataElement.getCallPathIDParents(id);}

    public ListIterator getCallPathIDChildren(int id){
	return globalThreadDataElement.getCallPathIDChildren(id);}
  
    public double getInclusiveValue(){
	return globalThreadDataElement.getInclusiveValue(trial.getSelectedMetricID());}
  
    public double getExclusiveValue(){
	return globalThreadDataElement.getExclusiveValue(trial.getSelectedMetricID());}
  
    public double getInclusivePercentValue(){
	return globalThreadDataElement.getInclusivePercentValue(trial.getSelectedMetricID());}
  
    public double getExclusivePercentValue(){
	return globalThreadDataElement.getExclusivePercentValue(trial.getSelectedMetricID());}
  
    public int getNumberOfCalls(){
	return globalThreadDataElement.getNumberOfCalls();}
  
    public int getNumberOfSubRoutines(){
	return globalThreadDataElement.getNumberOfSubRoutines();}
  
    public double getUserSecPerCall(){
	return globalThreadDataElement.getUserSecPerCall(trial.getSelectedMetricID());}
  
    public String getTStatString(int type, int precision){
	return globalThreadDataElement.getTStatString(type, trial.getSelectedMetricID(), precision);}
    //####################################
    //End - Function interface.
    //####################################
  
    //####################################
    //Userevent interface.
    //####################################
    public int getUserEventNumberValue(){
	return globalThreadDataElement.getUserEventNumberValue();}
  
    public double getUserEventMinValue(){
	return globalThreadDataElement.getUserEventMinValue();}
  
    public double getUserEventMaxValue(){
	return globalThreadDataElement.getUserEventMaxValue();}
  
    public double getUserEventMeanValue(){
	return globalThreadDataElement.getUserEventMeanValue();}
  
    public String getUserEventStatString(int precision){
	return globalThreadDataElement.getUserEventStatString(precision);}
    //####################################
    //End - Userevent interface.
    //####################################

    //####################################
    //Mean interface.
    //####################################
    public boolean getMeanValuesSet(){
	return globalMappingElement.getMeanValuesSet();}
  
    public double getMeanExclusiveValue(){
	return globalMappingElement.getMeanExclusiveValue(trial.getSelectedMetricID());}
  
    public double getMeanExclusivePercentValue(){
	return globalMappingElement.getMeanExclusivePercentValue(trial.getSelectedMetricID());}
  
    public double getMeanInclusiveValue(){
	return globalMappingElement.getMeanInclusiveValue(trial.getSelectedMetricID());}
    
    public double getMeanInclusivePercentValue(){
	return globalMappingElement.getMeanInclusivePercentValue(trial.getSelectedMetricID());}
  
    public double getMeanNumberOfCalls(){
	return globalMappingElement.getMeanNumberOfCalls();}
  
    public double getMeanNumberOfSubRoutines(){
	return globalMappingElement.getMeanNumberOfSubRoutines();}
  
    public double getMeanUserSecPerCall(){
	return globalMappingElement.getMeanUserSecPerCall(trial.getSelectedMetricID());}

    public String getMeanTotalStatString(int type, int precision){
	return globalMappingElement.getMeanTotalStatString(type, trial.getSelectedMetricID(), precision);}
    //####################################
    //End - Mean interface.
    //####################################


    /*
      (0) name
      (2) exclusive
      (4) inclusive
      (6) number of calls
      (8) number of subroutines
      (10) per call value
      (12) userevent number value
      (14) userevent min value
      (16) userevent max value
      (18) userevent mean value
      (20) mean exclusive
      (22) mean inclusive
      (24) mean number of calls
      (26) mean number of subroutines
      (28) mean per call value
      (30) n,c,t.

      The even values represent these items sorted in decending order,
      the odd values in ascending order. Thus (0) is name decending, and
      (1) is name ascending. Set sortType to the integer value required.
    */

    public int compareTo(Object inObject){
	switch(sortType){
	case 0:
	    return (((SMWThreadDataElement) inObject).getMappingName()).compareTo(this.getMappingName());
	case 1:
	    return (this.getMappingName()).compareTo(((SMWThreadDataElement)inObject).getMappingName());
	case 2:
	    return compareToHelper(((SMWThreadDataElement)inObject).getExclusiveValue(),this.getExclusiveValue());
	case 3:
	    return compareToHelper(this.getExclusiveValue(),((SMWThreadDataElement)inObject).getExclusiveValue());
	case 4:
	    return compareToHelper(((SMWThreadDataElement)inObject).getInclusiveValue(),this.getInclusiveValue());
	case 5:
	    return compareToHelper(this.getInclusiveValue(),((SMWThreadDataElement)inObject).getInclusiveValue());
	case 6:
	    return compareToHelper(((SMWThreadDataElement)inObject).getNumberOfCalls(),this.getNumberOfCalls());
	case 7:
	    return compareToHelper(this.getNumberOfCalls(),((SMWThreadDataElement)inObject).getNumberOfCalls());
	case 8:
	    return compareToHelper(((SMWThreadDataElement)inObject).getNumberOfSubRoutines(),this.getNumberOfSubRoutines());
	case 9:
	    return compareToHelper(this.getNumberOfSubRoutines(),((SMWThreadDataElement)inObject).getNumberOfSubRoutines());
	case 10:
	    return compareToHelper(((SMWThreadDataElement)inObject).getUserSecPerCall(),this.getUserSecPerCall());
	case 11:
	    return compareToHelper(this.getUserSecPerCall(),((SMWThreadDataElement)inObject).getUserSecPerCall());
	case 12:
	    return compareToHelper(((SMWThreadDataElement)inObject).getUserEventNumberValue(),this.getUserEventNumberValue());
	case 13:
	    return compareToHelper(this.getUserEventNumberValue(),((SMWThreadDataElement)inObject).getUserEventNumberValue());
	case 14:
	    return compareToHelper(((SMWThreadDataElement)inObject).getUserEventMinValue(),this.getUserEventMinValue());
	case 15:
	    return compareToHelper(this.getUserEventMinValue(),((SMWThreadDataElement)inObject).getUserEventMinValue());
	case 16:
	    return compareToHelper(((SMWThreadDataElement)inObject).getUserEventMaxValue(),this.getUserEventMaxValue());
	case 17:
	    return compareToHelper(this.getUserEventMaxValue(),((SMWThreadDataElement)inObject).getUserEventMaxValue());
	case 18:
	    return compareToHelper(((SMWThreadDataElement)inObject).getUserEventMeanValue(),this.getUserEventMeanValue());
	case 19:
	    return compareToHelper(this.getUserEventMeanValue(),((SMWThreadDataElement)inObject).getUserEventMeanValue());
	case 20:
	    return compareToHelper(((SMWThreadDataElement)inObject).getMeanExclusiveValue(),this.getMeanExclusiveValue());
	case 21:
	    return compareToHelper(this.getMeanExclusiveValue(),((SMWThreadDataElement)inObject).getMeanExclusiveValue());
	case 22:
	    return compareToHelper(((SMWThreadDataElement)inObject).getMeanInclusiveValue(),this.getMeanInclusiveValue());
	case 23:
	    return compareToHelper(this.getMeanInclusiveValue(),((SMWThreadDataElement)inObject).getMeanInclusiveValue());
	case 24:
	    return compareToHelper(((SMWThreadDataElement)inObject).getMeanNumberOfCalls(),this.getMeanNumberOfCalls());
	case 25:
	    return compareToHelper(this.getMeanNumberOfCalls(),((SMWThreadDataElement)inObject).getMeanNumberOfCalls());
	case 26:
	    return compareToHelper(((SMWThreadDataElement)inObject).getMeanNumberOfSubRoutines(),this.getMeanNumberOfSubRoutines());
	case 27:
	    return compareToHelper(this.getMeanNumberOfSubRoutines(),((SMWThreadDataElement)inObject).getMeanNumberOfSubRoutines());
	case 28:
	    return compareToHelper(((SMWThreadDataElement)inObject).getMeanUserSecPerCall(),this.getMeanUserSecPerCall());
	case 29:
	    return compareToHelper(this.getMeanUserSecPerCall(),((SMWThreadDataElement)inObject).getMeanUserSecPerCall());
	case 30:
	    SMWThreadDataElement sMWThreadDataElement = (SMWThreadDataElement) inObject;
	    if(sMWThreadDataElement.getNodeID()!=this.getNodeID())
		return sMWThreadDataElement.getNodeID() - this.getNodeID();
	    else if(sMWThreadDataElement.getContextID()!=this.getContextID())
		return sMWThreadDataElement.getContextID() - this.getContextID();
	    else
		return sMWThreadDataElement.getThreadID() - this.getThreadID();
	case 31:
	    sMWThreadDataElement = (SMWThreadDataElement) inObject;
	    if(sMWThreadDataElement.getNodeID()!=this.getNodeID())
		return this.getNodeID() - sMWThreadDataElement.getNodeID();
	    else if(sMWThreadDataElement.getContextID()!=this.getContextID())
		return this.getContextID() - sMWThreadDataElement.getContextID();
	    else
		return this.getThreadID() - sMWThreadDataElement.getThreadID();
 	default:
	    UtilFncs.systemError(null, null, "Unexpected sort type - SMWTDE value: " + sortType);
	}
	return 0;
    }
    
    private int compareToHelper(double d1, double d2){
	double result = d1 - d2;
	if(result < 0.00)
	    return -1;
	else if(result == 0.00)
	    return 0;
	else
	    return 1;
    }
 
    public void setDrawCoords(int xBeg, int xEnd, int yBeg, int yEnd){
	this.xBeg = xBeg;
	this.xEnd = xEnd;
	this.yBeg = yBeg;
	this.yEnd = yEnd;
    }
  
    public int getXBeg(){
	return xBeg;}
  
    public int getXEnd(){
	return xEnd;}
  
    public int getYBeg(){
	return yBeg;}
  
    public int getYEnd(){
	return yEnd;}
  
    public boolean getStatDrawnTo(){
	return statDrawnTo;}
  
    public void setStatDrawnTo(boolean statDrawnTo){
	this.statDrawnTo = statDrawnTo;}
  
    public void setHighlighted(boolean highlighted){
	this.highlighted = highlighted;}
  
    public boolean isHighlighted(){
	return highlighted;}
  
  
    public void setSortType(int sortType){
	this.sortType = sortType;}
  
    //####################################
    //Instance data.
    //####################################
  
    private ParaProfTrial trial = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;

    //Global Thread Data Element Reference.
    GlobalThreadDataElement globalThreadDataElement;
  
    //Global Mapping reference.
    GlobalMapping globalMapping;
    GlobalMappingElement globalMappingElement;
  
    //Drawing coordinates for this thread data object.
    int xBeg = 0;
    int xEnd = 0;
    int yBeg = 0;
    int yEnd = 0;

    boolean statDrawnTo;
  
    //Boolean indicating whether or not this object is highlighted.
    boolean highlighted = false;
  
    int sortType;
    //####################################
    //End - Instance data.
    //####################################
}
