/* 
  SMWMeanDataElement.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package ParaProf;

import java.util.*;
import java.lang.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;

public class SMWMeanDataElement implements Comparable
{

  //Constructor.
  public SMWMeanDataElement(Trial inTrial)
  { 
    trial = inTrial;
    
    globalMappingReference = trial.getGlobalMapping();
    
    value = 0;
    
    xBeginPosition = 0;
    xEndPosition = 0;
    yBeginPosition = 0;
    yEndPosition = 0;
    
    sortByMappingID = false;
    sortByName = true;
    sortByValue = false;
    sortByReverse = false;
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
  
  public boolean isGroupMember(int inGroupID)
  {
    return globalMappingReference.isGroupMember(mappingID, inGroupID, 0);
  }
  
  public int compareTo(Object inObject){
    
    double tmpDouble = 0;
    
    switch(sortSetting){
      case(1):
        return ((((SMWMeanDataElement)inObject).getMappingID()) - mappingID);
      case(2):
        return (mappingID - (((SMWMeanDataElement)inObject).getMappingID()));
      case(3):
        return (((SMWMeanDataElement) inObject).getMappingName()).compareTo(this.getMappingName());
      case(4):
        return (this.getMappingName()).compareTo(((SMWMeanDataElement)inObject).getMappingName());
      case(5):
        tmpDouble = (value - (((SMWMeanDataElement)inObject).getValue()));
        if(tmpDouble < 0.00)
          return 1;
        else if(tmpDouble == 0.00)
          return 0;
        else
          return -1;
      case(6):
        tmpDouble = (value - (((SMWMeanDataElement)inObject).getValue()));
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
  
  public int testCompareTo(Object inObject)
  {
    //Note that list will never call to compare against mapping id.  This
    //is because all the mappings are already sorted on the system.
    double tmpDouble = 0;
    
    if(sortByMappingID)
    {
      if(!sortByReverse)
        return (mappingID - (((SMWMeanDataElement)inObject).getMappingID()));
      else
        return ((((SMWMeanDataElement)inObject).getMappingID()) - mappingID);
    } 
        
    else if(sortByName)
    {
      if(!sortByReverse)
        return (this.getMappingName()).compareTo(((SMWMeanDataElement)inObject).getMappingName());
      else
        return (((SMWMeanDataElement) inObject).getMappingName()).compareTo(this.getMappingName());
    }
    
    //If here, means that we are in sort by value.
    tmpDouble = (value - (((SMWMeanDataElement)inObject).getValue()));
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
  
  public boolean getMeanValuesSet()
  {
    tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
    
    return tmpGME.getMeanValuesSet();
  }
  
  public double getMeanExclusiveValue()
  {
    tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
    
    return tmpGME.getMeanExclusiveValue(trial.getCurValLoc());
  }
  
  public double getMeanExclusivePercentValue()
  {
    tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
    
    return tmpGME.getMeanExclusivePercentValue(trial.getCurValLoc());
  }
  
  public double getMeanInclusiveValue()
  {
    tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
    
    return tmpGME.getMeanInclusiveValue(trial.getCurValLoc());
  }
  
  public double getMeanInclusivePercentValue()
  {
    tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
    
    return tmpGME.getMeanInclusivePercentValue(trial.getCurValLoc());
  }
  
  public String getMeanTotalStatString()
  {
    tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
    
    return tmpGME.getMeanTotalStatString(trial.getCurValLoc());
  }
  
  public double getMeanNumberOfCalls(){
    tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
    return tmpGME.getMeanNumberOfCalls();
  }
  
  public double getMeanNumberOfSubRoutines(){
    tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
    return tmpGME.getMeanNumberOfSubRoutines();
  }
  
  public double getMeanUserSecPerCall(){
    tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
    return tmpGME.getMeanUserSecPerCall(trial.getCurValLoc());
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
  
  public void setHighlighted(boolean inBool)
  {
    highlighted = inBool;
  }
  
  public boolean isHighlighted()
  {
    return highlighted;
  }
  
  public void setSortSetting(int inInt){
    sortSetting = inInt;
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
  
  private Trial trial = null;
  
  //A global mapping element reference.
  GlobalMappingElement tmpGME;
  
  //Global Mapping reference.
  GlobalMapping globalMappingReference;
  
  //Mapping ID
  int mappingID;

  //Named data values.
  double value;  
  
  //Drawing coordinates for this thread data object.
  int xBeginPosition;
  int xEndPosition;
  int yBeginPosition;
  int yEndPosition;
  
  //Boolean indicating whether or not this object is highlighted.
  boolean highlighted = false;
  
  int sortSetting = 0;
  
  boolean sortByMappingID;
  boolean sortByName;
  boolean sortByValue;
  boolean sortByReverse;
  
  boolean compareOnMappingName;
}
