/* 
   Name:       GlobalMapping.java
   Author:     Robert Bell
  
  
   Description:

   Things to do:
   1) Fix the naming scheme.  Would like what is now called a GlobalMappingElement to be just called
   a mapping.  Thus mappingID wouuld make sense, as would mapping name.  Then need to find a good
   name for the containers that hold the mappings.
*/

package edu.uoregon.tau.dms.dss;

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

    public void increaseVectorStorage(){
	//	System.out.println ("increaseVectorStorage called!\n");
	maxMeanInclusiveValueList.add(new Double(0));
	maxMeanExclusiveValueList.add(new Double(0));
	maxMeanInclusivePercentValueList.add(new Double(0));
	maxMeanExclusivePercentValueList.add(new Double(0));
	maxMeanUserSecPerCallList.add(new Double(0));
    }
    
    public int addGlobalMapping(String mappingName, int mappingSelection, int capacity){
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
	    GlobalMappingElement globalMappingElement = new GlobalMappingElement(capacity);
	    globalMappingElement.setMappingName(mappingName);
	    globalMappingElement.setMappingID(mappingID);

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
	    UtilFncs.systemError(e, null, "GM04");}
	return globalMappingElement;
    }

    public GlobalMappingElement getGlobalMappingElement(String mappingName, int mappingSelection){
	return (GlobalMappingElement) mappings[mappingSelection].elementAt(this.getMappingID(mappingName, mappingSelection));}

    public int getNumberOfMappings(int mappingSelection){
	return mappings[mappingSelection].size();}
    
    public Vector getMapping(int mappingSelection){
	return mappings[mappingSelection];}

    public ListIterator getMappingIterator(int mappingSelection){
	return new DataSessionIterator(mappings[mappingSelection]);}
  
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

    public boolean displayMapping(int mappingID){
	switch(groupFilter){
	case 0:
	    //No specific group selection is required.
	    return true;
	case 1:
	    //Show this group only.
	    if(this.isGroupMember(mappingID, this.getSelectedGroupID(), 0))
		return true;
	    else
		return false;
	case 2:
	    //Show all groups except this one.
	    if(this.isGroupMember(mappingID, this.getSelectedGroupID(), 0))
		return false;
	    else
		return true;
	default:
	    //Default case behaves as case 0.
	    return true;
	}
    }

    public void setSelectedGroupID(int selectedGroupID){
	this.selectedGroupID = selectedGroupID;}
  
    public int getSelectedGroupID(){
	return selectedGroupID;}

    public void setGroupFilter(int groupFilter){
	this.groupFilter = groupFilter;}

    public int getGroupFilter(){
	return groupFilter;}
    //######
    //End - Group functions
    //######

    //######
    //Functions setting mean and total values.
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
    //Functions for max mean values.
    //######
    public void setMaxMeanInclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	//maxMeanInclusiveValueList.add(dataValueLocation, tmpDouble);
	maxMeanInclusiveValueList.setElementAt(tmpDouble,dataValueLocation);
    }
  
    public void setMaxMeanExclusiveValue(int dataValueLocation, double inDouble){
	//System.out.println ("setMaxMeanExclusiveValue(" + dataValueLocation + ", " + inDouble + ")");
	Double tmpDouble = new Double(inDouble);
	//maxMeanExclusiveValueList.add(dataValueLocation, tmpDouble);
	maxMeanExclusiveValueList.setElementAt(tmpDouble,dataValueLocation);
    }
  
    public void setMaxMeanInclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	//maxMeanInclusivePercentValueList.add(dataValueLocation, tmpDouble);
	maxMeanInclusivePercentValueList.setElementAt(tmpDouble,dataValueLocation);
    }
  
    public void setMaxMeanExclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	//maxMeanExclusivePercentValueList.add(dataValueLocation, tmpDouble);
	maxMeanExclusivePercentValueList.setElementAt(tmpDouble,dataValueLocation);
    }

    public void setMaxMeanNumberOfCalls(double inDouble){
	maxMeanNumberOfCalls = inDouble;}
  
    public void setMaxMeanNumberOfSubRoutines(double inDouble){
	maxMeanNumberOfSubRoutines = inDouble;}

    public void setMaxMeanUserSecPerCall(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	//maxMeanUserSecPerCallList.add(dataValueLocation, tmpDouble);
	maxMeanUserSecPerCallList.setElementAt(tmpDouble,dataValueLocation);
    }

    public double getMaxMeanExclusiveValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanExclusiveValueList.elementAt(dataValueLocation);
	//System.out.println ("getMaxMeanExclusiveValue(" + dataValueLocation + ") = " + tmpDouble);
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
    
    private int selectedGroupID = -1;
    private int groupFilter = 0;
    //####################################
    //End - Instance data.
    //#################################### 

}
