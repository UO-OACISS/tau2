/* 
   GlobalMapping.java

   Title:      ParaProf
   Author:     Robert Bell
  
  
   Description:

   Things to do:
   1) Fix the naming scheme.  Would like what is now called a GlobalMappingElement to be just called
   a mapping.  Thus mappingID wouuld make sense, as would mapping name.  Then need to find a good
   name for the containers that hold the mappings.
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;


public class GlobalMapping{

    public GlobalMapping(){
	mappings[0] = new Vector();
	mappings[1] = new Vector();
	mappings[2] = new Vector();

	sortedMappings[0] = new Vector();
	sortedMappings[1] = new Vector();
	sortedMappings[2] = new Vector();
    }

    protected void increaseVectorStorage(){
	maxMeanInclusiveValueList.add(new Double(0));
	maxMeanExclusiveValueList.add(new Double(0));
	maxMeanInclusivePercentValueList.add(new Double(0));
	maxMeanExclusivePercentValueList.add(new Double(0));
	maxMeanUserSecPerCallList.add(new Double(0));
    }
    
    public int addGlobalMapping(String mappingName, int mappingSelection){
	int mappingID = -1;
	int pos = getSortedMappingPosition(mappingName, mappingSelection);

	//If the returned id is less than 0, this means that this mapping name
	//is NOT present in the given mapping.
	if(pos < 0){
	    //Add a new GlobalMappingElement at the end of the given mapping. 
	    //Note: We could insert into a given position that matched the
	    //sorted list.  This would violate one principle that we want to 
	    //maintain however - once a mapping is assigned an id, don't 
	    //change it.  This is important for runtime situations, whilst 
	    //still being able to use the id as a location into the mapping for
	    //efficient lookup.
	    mappingID = mappings[mappingSelection].size();
	    GlobalMappingElement globalMappingElement = new GlobalMappingElement();
	    globalMappingElement.setMappingName(mappingName);
	    globalMappingElement.setGlobalID(mappingID);

	    mappings[mappingSelection].addElement(globalMappingElement);

	    //Now insert into the sorted list.
	    GlobalSortedMappingElement globalSortedMappingElement = new GlobalSortedMappingElement(mappingName, mappingID);
	    sortedMappings[mappingSelection].insertElementAt(globalSortedMappingElement, (-(pos+1)));
	}
	else{
	    GlobalSortedMappingElement globalSortedMappingElement = getGlobalSortedMappingElement(pos, mappingSelection);
	    mappingID = globalSortedMappingElement.getMappingID();
	}
	return mappingID;
    }

    public int getMappingID(String mappingName, int mappingSelection){
	//The id lookup takes place on the sorted list.
	GlobalSortedMappingElement globalSortedMappingElement = null;
	globalSortedMappingElement = this.getGlobalSortedMappingElement(mappingName, mappingSelection);
	if(globalSortedMappingElement!=null)
	    return globalSortedMappingElement.getMappingID();
	else
	    return -1;
    }

    public GlobalMappingElement getGlobalMappingElement(int mappingID, int mappingSelection){
	GlobalMappingElement globalMappingElement = null;
	try{
	    globalMappingElement = (GlobalMappingElement) mappings[mappingSelection].elementAt(mappingID);}
	catch(Exception e){
	    ParaProf.systemError(e, null, "GM04");}
	return globalMappingElement;
    }

    public GlobalMappingElement getGlobalMappingElement(String mappingName, int mappingSelection){
	return (GlobalMappingElement) mappings[mappingSelection].elementAt(this.getMappingID(mappingName, mappingSelection));}

    public int getNumberOfMappings(int mappingSelection){
	return mappings[mappingSelection].size();}
    
    public Vector getMapping(int mappingSelection){
	return mappings[mappingSelection];}

    public ListIterator getMappingIterator(int mappingSelection){
	return new ParaProfIterator(mappings[mappingSelection]);}
  
    //######
    //Group functions
    //######
    public void addGroup(int mappingID, int groupID, int mappingSelection){
	GlobalMappingElement globalMappingElement = this.getGlobalMappingElement(mappingID, mappingSelection);
	globalMappingElement.addGroup(groupID);
    }
  
    public boolean isGroupMember(int mappingID, int groupID, int mappingSelection){
	GlobalMappingElement globalMappingElement = this.getGlobalMappingElement(mappingID, mappingSelection);
	if(globalMappingElement.isGroupMember(groupID))
	    return true;
	else
	    return false;
    }

    public void setIsSelectedGroupOn(boolean inBool){
	isSelectedGroupOn = inBool;}
  
    public boolean getIsSelectedGroupOn(){
	return isSelectedGroupOn;}
  
    public void setIsAllExceptGroupOn(boolean inBool){
	isAllExceptGroupOn = inBool;}
  
    public boolean getIsAllExceptGroupOn(){
	return isAllExceptGroupOn;}
  
    public void setSelectedGroupID(int inInt){
	selectedGroupID = inInt;}
  
    public int getSelectedGroupID(){
	return selectedGroupID;}
    //######
    //End - Group functions
    //######

    //######
    //Functions setting max and total values.
    //######
    public void setMeanExclusiveValueAt(int dataValueLocation, double value, int mappingID, int mappingSelection){
	GlobalMappingElement globalMappingElement = this.getGlobalMappingElement(mappingID, mappingSelection);
	globalMappingElement.setMeanExclusiveValue(dataValueLocation, value);
    }

    public void setMeanInclusiveValueAt(int dataValueLocation, double value, int mappingID, int mappingSelection){
	GlobalMappingElement globalMappingElement = this.getGlobalMappingElement(mappingID, mappingSelection);
	globalMappingElement.setMeanInclusiveValue(dataValueLocation, value);
    }

    public void setTotalExclusiveValueAt(int dataValueLocation, double value, int mappingID, int mappingSelection){
	GlobalMappingElement globalMappingElement = this.getGlobalMappingElement(mappingID, mappingSelection);
	globalMappingElement.setTotalExclusiveValue(dataValueLocation, value);
    }

    public void setTotalInclusiveValueAt(int dataValueLocation, double value, int mappingID, int mappingSelection){
	GlobalMappingElement globalMappingElement = this.getGlobalMappingElement(mappingID, mappingSelection);
	globalMappingElement.setTotalInclusiveValue(dataValueLocation, value);
    }
    //######
    //End - Functions setting max and total values.
    //######
    
    //######
    //Function to compute mean values.
    //######
    public void computeMeanData(int mappingSelection, int metric){
	ListIterator l = this.getMappingIterator(mappingSelection);
	double exclusiveTotal = 0.0;
	GlobalMappingElement globalMappingElement = null;
	while(l.hasNext()){
	    globalMappingElement = (GlobalMappingElement) l.next();
	    if((globalMappingElement.getCounter()) != 0){
		double d = (globalMappingElement.getTotalExclusiveValue())/(globalMappingElement.getCounter());
		//Increment the total values.
		exclusiveTotal+=d;
		globalMappingElement.setMeanExclusiveValue(metric, d);
		if((this.getMaxMeanExclusiveValue(metric) < d))
		    this.setMaxMeanExclusiveValue(metric, d);
					
		d = (globalMappingElement.getTotalInclusiveValue())/(globalMappingElement.getCounter());
		globalMappingElement.setMeanInclusiveValue(metric, d);
		if((this.getMaxMeanInclusiveValue(metric) < d))
		    this.setMaxMeanInclusiveValue(metric, d);
	    }
	}
				
	double inclusiveMax = this.getMaxMeanInclusiveValue(metric);
				
	l = this.getMappingIterator(mappingSelection);
	while(l.hasNext()){
	    globalMappingElement = (GlobalMappingElement) l.next();
				    
	    if(exclusiveTotal!=0){
		double tmpDouble = ((globalMappingElement.getMeanExclusiveValue(metric))/exclusiveTotal) * 100;
		globalMappingElement.setMeanExclusivePercentValue(metric, tmpDouble);
		if((this.getMaxMeanExclusivePercentValue(metric) < tmpDouble))
		    this.setMaxMeanExclusivePercentValue(metric, tmpDouble);
	    }
				    
	    if(inclusiveMax!=0){
		double tmpDouble = ((globalMappingElement.getMeanInclusiveValue(metric))/inclusiveMax) * 100;
		globalMappingElement.setMeanInclusivePercentValue(metric, tmpDouble);
		if((this.getMaxMeanInclusivePercentValue(metric) < tmpDouble))
		    this.setMaxMeanInclusivePercentValue(metric, tmpDouble);
	    }
	    globalMappingElement.setMeanValuesSet(true);
	}
    }
    //######
    //End - Function to compute mean values.
    //######

    //######
    //Functions for max mean values.
    //######
    public void setMaxMeanInclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanInclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    public void setMaxMeanExclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanExclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    public void setMaxMeanInclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanInclusivePercentValueList.add(dataValueLocation, tmpDouble);}
  
    public void setMaxMeanExclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanExclusivePercentValueList.add(dataValueLocation, tmpDouble);}

    public void setMaxMeanNumberOfCalls(double inDouble){
	maxMeanNumberOfCalls = inDouble;}
  
    public void setMaxMeanNumberOfSubRoutines(double inDouble){
	maxMeanNumberOfSubRoutines = inDouble;}

    public void setMaxMeanUserSecPerCall(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanUserSecPerCallList.add(dataValueLocation, tmpDouble);}

    public double getMaxMeanExclusiveValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanExclusiveValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}

    public double getMaxMeanInclusiveValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanInclusiveValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}

    public double getMaxMeanInclusivePercentValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanInclusivePercentValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}

    public double getMaxMeanExclusivePercentValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanExclusivePercentValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}

    public double getMaxMeanNumberOfCalls(){
	return maxMeanNumberOfCalls;}
  
    public double getMaxMeanNumberOfSubRoutines(){
	return maxMeanNumberOfSubRoutines;}
  
    public double getMaxMeanUserSecPerCall(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanUserSecPerCallList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
    //######
    //End - Function to set max mean values.
    //######

    //######
    //Functions managing the sortedMapppings.
    //######
    //Returns element whose mapping name matches the given mapping name in the specified list.
    private GlobalSortedMappingElement getGlobalSortedMappingElement(String mappingName, int mappingSelection){
	int pos = this.getSortedMappingPosition(mappingName, mappingSelection);
	if(pos>=0)
	    return this.getGlobalSortedMappingElement(pos, mappingSelection);
	else
	    return null;
    }

    //Returns element at the specified position in the specified list.
    private GlobalSortedMappingElement getGlobalSortedMappingElement(int pos, int mappingSelection){
	return (GlobalSortedMappingElement) sortedMappings[mappingSelection].elementAt(pos);}

    private int getSortedMappingPosition(String mappingName, int mappingSelection){
	return Collections.binarySearch(sortedMappings[mappingSelection], 
					new GlobalSortedMappingElement(mappingName, -1));}
    //######
    //End - Functions managing the sortedMapppings.
    //######

    public void setColors(ColorChooser c, int mappingSelection){
	//If the mapping selection is equal to -1, then set the colors in all the mappings,
	//otherwise, just set the ones for the specified mapping.

	if((mappingSelection == -1) || (mappingSelection == 0)){
	    int numberOfColors = c.getNumberOfColors();
	    for(Enumeration e = mappings[0].elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		globalMappingElement.setMappingColor(c.getColorInLocation((globalMappingElement.getGlobalID()) % numberOfColors));
	    }
	}
	
	if((mappingSelection == -1) || (mappingSelection == 1)){
	    int numberOfColors = c.getNumberOfMappingGroupColors();
	    for(Enumeration e = mappings[1].elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		globalMappingElement.setMappingColor(c.getMappingGroupColorInLocation((globalMappingElement.getGlobalID()) % numberOfColors));
	    }
	}

	if((mappingSelection == -1) || (mappingSelection == 2)){
	    int numberOfColors = c.getNumberOfColors();
	    for(Enumeration e = mappings[2].elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		globalMappingElement.setMappingColor(c.getColorInLocation((globalMappingElement.getGlobalID()) % numberOfColors));
	    }
	}
    }

    //####################################
    //Instance data.
    //####################################
    //An array of Vectors each of which holds GlobalMappingElements.
    private Vector[] mappings = new Vector[3];
    //An array of Vectors each of which holds GlobalSortedMappingElements.
    private Vector[] sortedMappings = new Vector[3];

    private Vector maxMeanInclusiveValueList = new Vector();
    private Vector maxMeanExclusiveValueList = new Vector();
    private Vector maxMeanInclusivePercentValueList = new Vector();
    private Vector maxMeanExclusivePercentValueList = new Vector();
    private double maxMeanNumberOfCalls = 0;
    private double maxMeanNumberOfSubRoutines = 0;
    private Vector maxMeanUserSecPerCallList = new Vector();
    
    private boolean isSelectedGroupOn = false;
    private boolean isAllExceptGroupOn = false;
    private int selectedGroupID = -1;
    //####################################
    //End - Instance data.
    //#################################### 

}
