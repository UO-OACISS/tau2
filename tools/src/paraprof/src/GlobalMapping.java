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

    public GlobalMapping(Trial trial){
	this.trial = trial;

	mappings[0] = new Vector();
	mappings[1] = new Vector();
	mappings[2] = new Vector();

	sortedMappings[0] = new Vector();
	sortedMappings[1] = new Vector();
	sortedMappings[2] = new Vector();
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
	    GlobalMappingElement globalMappingElement = new GlobalMappingElement(trial);
	    globalMappingElement.setMappingName(mappingName);
	    globalMappingElement.setGlobalID(mappingID);

	    ColorChooser c = trial.getColorChooser();
	    int numOfColors = c.getNumberOfColors();
	    int numOfGroupColors = c.getNumberOfMappingGroupColors();
	    if((mappingSelection == 0) || (mappingSelection == 2))
		globalMappingElement.setGenericColor(c.getColorInLocation(mappingID % numOfColors));
	    else
		globalMappingElement.setGenericColor(c.getMappingGroupColorInLocation(mappingID % numOfGroupColors));

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

    public void updateGenericColors(int mappingSelection){
	if(mappingSelection == 0){
	    int tmpInt = trial.getColorChooser().getNumberOfColors();
	    for(Enumeration e = mappings[mappingSelection].elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		int mappingID = globalMappingElement.getGlobalID();
		globalMappingElement.setGenericColor(trial.getColorChooser().getColorInLocation(mappingID % tmpInt));
	    }
	}
	else{
	    int tmpInt = trial.getColorChooser().getNumberOfMappingGroupColors();
	    for(Enumeration e = mappings[mappingSelection].elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		int mappingID = globalMappingElement.getGlobalID();
		globalMappingElement.setGenericColor(trial.getColorChooser().getMappingGroupColorInLocation(mappingID % tmpInt));
	    }
	}   
    }

    //####################################
    //Instance data.
    //####################################
    private Trial trial = null;
  
    //An array of Vectors each of which holds GlobalMappingElements.
    private Vector[] mappings = new Vector[3];
    //An array of Vectors each of which holds GlobalSortedMappingElements.
    private Vector[] sortedMappings = new Vector[3];
    
    private boolean isSelectedGroupOn = false;
    private boolean isAllExceptGroupOn = false;
    private int selectedGroupID = -1;
    //####################################
    //End - Instance data.
    //#################################### 

}
