/* 
  GlobalMappingElement.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package ParaProf;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;
import java.text.*;

public class GlobalMappingElement implements Serializable, Comparable
{
  //Constructors.
  public GlobalMappingElement(Trial inTrial)
  {
    trial = inTrial;
    
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
    GlobalMapping tmpGM = trial.getGlobalMapping();
    
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
  public void setMaxUserEventNumberValue(int inInt){
    maxUserEventNumberValue = inInt;}
  
  public int getMaxUserEventNumberValue(){
    return maxUserEventNumberValue;}
  
  public void setMaxUserEventMinValue(double inDouble){
    maxUserEventMinValue = inDouble;}
  
  public double getMaxUserEventMinValue(){
    return maxUserEventMinValue;}
  
  public void setMaxUserEventMaxValue(double inDouble){
    maxUserEventMaxValue = inDouble;}
  
  public double getMaxUserEventMaxValue(){
    return maxUserEventMaxValue;}
  
  public void setMaxUserEventMeanValue(double inDouble){
    maxUserEventMeanValue = inDouble;}
  
  public double getMaxUserEventMeanValue(){
    return maxUserEventMeanValue;
  }
  
  
  
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
  
  public double getMeanNumberOfSubRoutines(){
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
  public String getMeanTotalStatString(int dataValueLocation){
  
    try{
      int defaultNumberPrecision = ParaProf.defaultNumberPrecision;
      int initialBufferLength = 97;
      int position = 0;
      char [] statStringArray = new char[initialBufferLength];
      char [] tmpArray;
      String tmpString;
      
      this.insertSpaces(statStringArray , 0, 97);
      
      DecimalFormat dF = new DecimalFormat();
      dF.applyPattern("##0.0");
      tmpArray = (dF.format(this.getMeanInclusivePercentValue(dataValueLocation))).toCharArray();
      
      for(int i=0;i<tmpArray.length;i++){
        statStringArray[position] = tmpArray[i];
        position++;
      }
      
      position = 7;
      tmpString = new String(Double.toString(
              UtilFncs.adjustDoublePresision(this.getMeanExclusiveValue(dataValueLocation),
                              defaultNumberPrecision)));
      tmpArray = tmpString.toCharArray();
      for(int i=0;i<tmpArray.length;i++){
        statStringArray[position] = tmpArray[i];
        position++;
      }
      
      position = 25;
      tmpString = new String(Double.toString(
              UtilFncs.adjustDoublePresision(this.getMeanInclusiveValue(dataValueLocation),
                              defaultNumberPrecision)));
      tmpArray = tmpString.toCharArray();
      for(int i=0;i<tmpArray.length;i++){
        statStringArray[position] = tmpArray[i];
        position++;
      }
      
      position = 43;
      tmpString = new String(Double.toString(
              UtilFncs.adjustDoublePresision(this.getMeanNumberOfCalls(),
                              defaultNumberPrecision)));
      tmpArray = tmpString.toCharArray();                       
      for(int i=0;i<tmpArray.length;i++){
        statStringArray[position] = tmpArray[i];
        position++;
      }
      
      position = 61;
      tmpString = new String(Double.toString(
              UtilFncs.adjustDoublePresision(this.getMeanNumberOfSubRoutines(),
                              defaultNumberPrecision)));
      tmpArray = tmpString.toCharArray();
      for(int i=0;i<tmpArray.length;i++){
        statStringArray[position] = tmpArray[i];
        position++;
      }
      
      position = 79;
      tmpString = new String(Double.toString(
              UtilFncs.adjustDoublePresision(this.getMeanUserSecPerCall(dataValueLocation),
                              defaultNumberPrecision)));
      tmpArray = tmpString.toCharArray();
      for(int i=0;i<tmpArray.length;i++){
        statStringArray[position] = tmpArray[i];
        position++;
      }
      
      //Everything should be added now except the function name.
      String firstPart = new String(statStringArray);
      return firstPart + this.getMappingName();
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "GTDE01");
    }
    
    return "An error occured pocessing this string!";
  }
  
  private int insertSpaces(char[] inArray, int position, int number){
    for(int i=0;i<number;i++){
      inArray[position] = '\u0020';
      position++;
    }
    return position;
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
  
  
  public int compareTo(Object inObject){
    return mappingName.compareTo((String)inObject);
  }
  
  
  
  //Instance elmements.
  
  private Trial trial = null;
  
  //Global Mapping reference.
  String mappingName;
  int globalID;     //Global ID for this mapping.
  
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
  
  private int maxUserEventNumberValue = 0;
  private double maxUserEventMinValue = 0;
  private double maxUserEventMaxValue = 0;
  private double maxUserEventMeanValue = 0;
  
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
