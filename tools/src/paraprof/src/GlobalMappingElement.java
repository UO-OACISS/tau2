/* 
   GlobalMappingElement.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;
import java.text.*;

public class GlobalMappingElement implements Mapping, Serializable, Comparable{
    //######
    //Constructors.
    //######
    public GlobalMappingElement(int capacity){
	doubleList = new double[capacity*14];}
    //######
    //End - Constructors.
    //######
    
    public void setMappingName(String mappingName){
	this.mappingName = mappingName;}
  
    public String getMappingName(){
	return mappingName;}
  
    public void setMappingID(int mappingID){
	this.mappingID = mappingID;}
  
    public int getMappingID(){
	return mappingID;}

    //######
    //Storage control.
    //######
    public void incrementStorage(){
	int currentLength = doubleList.length;
	double[] newArray = new double[currentLength+14];
	for(int i=0;i<currentLength;i++){
	    newArray[i] = doubleList[i];
	}
	doubleList = newArray;
    }

    public void incrementStorage(int increase){
	int currentLength = doubleList.length;
	double[] newArray = new double[currentLength+(increase*14)];
	for(int i=0;i<currentLength;i++){
	    newArray[i] = doubleList[i];
	}
	doubleList = newArray;
    }
    //######
    //End - Storage control.
    //######
  
    //######
    //Group section.
    //######
    public void addGroup(int id){
	//Don't add group if already a member.
	if(this.isGroupMember(id))
	    return;
	
	if(groups==null)
	    groups = new int[5];
	else if(groups.length<=numberOfGroups){
	    int currentLength = groups.length;
	    int[] newArray = new int[currentLength+5];
	    for(int i=0;i<currentLength;i++){
		newArray[i] = groups[i];
	    }
	    groups = newArray;	    
	}
	//Safe to add.
	groups[numberOfGroups] = id;
	numberOfGroups++;
    }

    public boolean isGroupMember(int inGroupID){
	for(int i=0;i<numberOfGroups;i++){
	    if(groups[i] == inGroupID){
		return true;
	    }
	}
	return false;
    }

    public int[] getGroups(){
	return groups;}

    public void setGroupsSet(boolean groupsSet){
	this.groupsSet = groupsSet;}

    public boolean groupsSet(){
	return groupsSet;}
    //######
    //End - Group section.
    //######

    //######
    //Members section.
    //######
    public void addMember(GlobalMappingElement globalMappingElement){
	if(members==null)
	    members = new Vector();
	members.add(globalMappingElement);
    }
    //######
    //End - Members section.
    //######

    //######
    //Call path section.
    //######
    public void addParent(int id,int pathID){
	//Check to see if this parent is already present,
	//if so, add only the callpath to the system.
	int location = UtilFncs.exists(parents,id);
	if(location == -1){
	    if(parents==null){
		parents = new Vector();
		callPathIDSParents = new Vector();
	    }
	    
	    parents.add(new Integer(id));
	    Vector tmpVector = new Vector();
	    tmpVector.add(new Integer(pathID));
	    callPathIDSParents.add(tmpVector);

	}
	else{
	    Vector tmpVector = (Vector) callPathIDSParents.elementAt(location);
	    tmpVector.add(new Integer(pathID));
	}
    }

    public Vector getParents(){
	return parents;}

    public Vector getChildren(){
	return children;}

    public ListIterator getParentsIterator(){
	return new ParaProfIterator(parents);}

    public ListIterator getChildrenIterator(){
	return new ParaProfIterator(children);}

    public ListIterator getCallPathIDParents(int id){
	//The argument represents the id of the parent.
	//Get the location of the parent first.
	int location = UtilFncs.exists(parents,id);
	//Now return the callpath id list for that parent.
	return new ParaProfIterator((Vector)callPathIDSParents.elementAt(location));
    }

    public ListIterator getCallPathIDChildren(int id){
	//The argument represents the id of the child.
	//Get the location of the child first.
	int location = UtilFncs.exists(children,id);
	//Now return the callpath id list for that parent.
	return new ParaProfIterator((Vector)callPathIDSChildren.elementAt(location));
    }

    public void addChild(int id,int pathID){
	//Check to see if this child is already present,
	//if so, add only the callpath to the system.
	int location = UtilFncs.exists(children,id);
	if(location == -1){
	    if(children==null){
		children = new Vector();
		callPathIDSChildren = new Vector();
	    }
	    
	    children.add(new Integer(id));
	    Vector tmpVector = new Vector();
	    tmpVector.add(new Integer(pathID));
	    callPathIDSChildren.add(tmpVector);
	}
	else{
	    Vector tmpVector = (Vector) callPathIDSChildren.elementAt(location);
	    tmpVector.add(new Integer(pathID));
	}
    }

    public void setCallPathObject(boolean b){
	callPathObject = b;}

    public boolean isCallPathObject(){
	return callPathObject;}
    //######
    //End - Call path section.
    //######

    //######
    //End - Colour section.
    //######
    public void setColor(Color color){
	this.color = color;}

    public Color getColor(){
	if(colorFlag)
	    return specificColor;
	else
	    return color;
    }
    
    public void setColorFlag(boolean colorFlag){
	this.colorFlag = colorFlag;}
  
    public boolean isColorFlagSet(){
	return colorFlag;}
  
    public void setSpecificColor(Color specificColor){
	this.specificColor = specificColor;}
    //######
    //End - Colour section.
    //######
  
    //######
    //Max values section.
    //######
    public void setMaxInclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,0,inDouble);}
  
    public double getMaxInclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,0);}
  
    public void setMaxExclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,1,inDouble);}
  
    public double getMaxExclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,1);}
  
    public void setMaxInclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,2,inDouble);}
  
    public double getMaxInclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,2);}
  
    public void setMaxExclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,3,inDouble);}
  
    public double getMaxExclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,3);}
  
    public void setMaxNumberOfCalls(int inInt){
	maxNumberOfCalls = inInt;}
  
    public int getMaxNumberOfCalls(){
	return maxNumberOfCalls;}
  
    public void setMaxNumberOfSubRoutines(int inInt){
	maxNumberOfSubRoutines = inInt;}
  
    public int getMaxNumberOfSubRoutines(){
	return maxNumberOfSubRoutines;}
  
    public void setMaxUserSecPerCall(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,4,inDouble);}
  
    public double getMaxUserSecPerCall(int dataValueLocation){
	return this.getDouble(dataValueLocation,4);}
    //######
    //End - Max values section.
    //######
    
    //######
    //Userevent max section.
    //######
    public void setMaxUserEventNumberValue(int maxUserEventNumberValue){
	this.maxUserEventNumberValue = maxUserEventNumberValue;}
  
    public int getMaxUserEventNumberValue(){
	return maxUserEventNumberValue;}
  
    public void setMaxUserEventMinValue(double maxUserEventMinValue){
	this.maxUserEventMinValue = maxUserEventMinValue;}
  
    public double getMaxUserEventMinValue(){
	return maxUserEventMinValue;}
  
    public void setMaxUserEventMaxValue(double maxUserEventMaxValue){
	this.maxUserEventMaxValue = maxUserEventMaxValue;}
  
    public double getMaxUserEventMaxValue(){
	return maxUserEventMaxValue;}
  
    public void setMaxUserEventMeanValue(double maxUserEventMeanValue){
	this.maxUserEventMeanValue = maxUserEventMeanValue;}
  
    public double getMaxUserEventMeanValue(){
	return maxUserEventMeanValue;
    }
    //######
    //End - Userevent max section.
    //######

    //######
    //Mean section.
    //######
    public void setMeanInclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,5,inDouble);}
  
    public double getMeanInclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,5);}
  
    public void setMeanExclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,6,inDouble);}
  
    public double getMeanExclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,6);}
  
    public void setMeanInclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,7,inDouble);}
  
    public double getMeanInclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,7);}
  
    public void setMeanExclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,8,inDouble);}
  
    public double getMeanExclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,8);}
  
    public void setMeanNumberOfCalls(double inDouble){
	meanNumberOfCalls = inDouble;}
  
    public double getMeanNumberOfCalls(){
	return meanNumberOfCalls;}
  
    public void setMeanNumberOfSubRoutines(double inDouble){
	meanNumberOfSubRoutines = inDouble;}
  
    public double getMeanNumberOfSubRoutines(){
	return meanNumberOfSubRoutines;}
  
    public void setMeanUserSecPerCall(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,9,inDouble);}
  
    public double getMeanUserSecPerCall(int dataValueLocation){
  	return this.getDouble(dataValueLocation,9);}

    public String getMeanTotalStatString(int type, int dataValueLocation, int precision){
  	try{
	    int initialBufferLength = 99;
	    int position = 0;
	    char [] statStringArray = new char[initialBufferLength];
	    char [] tmpArray;
	    String tmpString;
      
	    this.insertSpaces(statStringArray , 0, 99);
      
	    DecimalFormat dF = new DecimalFormat();
	    dF.applyPattern("##0.0");
	    tmpArray = (dF.format(this.getMeanInclusivePercentValue(dataValueLocation))).toCharArray();
      
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 9;
	    tmpString = UtilFncs.getOutputString(type,this.getMeanExclusiveValue(dataValueLocation),precision);

	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 27;
	    tmpString = UtilFncs.getOutputString(type,this.getMeanInclusiveValue(dataValueLocation),precision);

	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 45;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getMeanNumberOfCalls(),
										  precision)));
	    tmpArray = tmpString.toCharArray();                       
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 63;
	    tmpString = new String(Double.toString(
						   UtilFncs.adjustDoublePresision(this.getMeanNumberOfSubRoutines(),
										  precision)));
	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    position = 81;
	    tmpString = UtilFncs.getOutputString(type,this.getMeanUserSecPerCall(dataValueLocation),precision);

	    tmpArray = tmpString.toCharArray();
	    for(int i=0;i<tmpArray.length;i++){
		statStringArray[position] = tmpArray[i];
		position++;
	    }
      
	    //Everything should be added now except the function name.
	    String firstPart = new String(statStringArray);
	    return firstPart + this.getMappingName();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "GTDE01");
	}
    	return "An error occured pocessing this string!";
    }

    public void setMeanValuesSet(boolean meanValuesSet){
	this.meanValuesSet = meanValuesSet;}
  
    public boolean getMeanValuesSet(){
	return meanValuesSet;}
    //######
    //End - Mean section.
    //######
  
    //######
    //Total section.
    //######
    public void setTotalInclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,10,inDouble);}
  
    public double getTotalInclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,10);}
  
    public void setTotalExclusiveValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,11,inDouble);}
  
    public double getTotalExclusiveValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,11);}
  
    public void setTotalInclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,12,inDouble);}
  
    public double getTotalInclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,12);}
  
    public void setTotalExclusivePercentValue(int dataValueLocation, double inDouble){
	this.insertDouble(dataValueLocation,13,inDouble);}
  
    public double getTotalExclusivePercentValue(int dataValueLocation){
	return this.getDouble(dataValueLocation,13);}
    //######
    //End - Total section.
    //######
  
    //######
    //Draw section.
    //######
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
    //######
    //End - Draw section.
    //######
  
    //######
    //Usage?
    //######
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
    //######
    //End - Usage?
    //######
    
    //####################################
    //Interface code.
    //####################################
    
    //######
    //Comparable section.
    //######
    public int compareTo(Object inObject){
	return mappingName.compareTo((String)inObject);}
    //######
    //End - Comparable section.
    //######

    //####################################
    //End - Interface code.
    //####################################

    //######
    //Private section.
    //######
    private void insertDouble(int dataValueLocation, int offset, double inDouble){
	int actualLocation = (dataValueLocation*14)+offset;
	try{
	    doubleList[actualLocation] = inDouble;
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "GME01");
	}
    }
  
    private double getDouble(int dataValueLocation, int offset){
	int actualLocation = (dataValueLocation*14)+offset;
	try{
	    return doubleList[actualLocation];
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "GME02");
	}
	return -1;
    }

    private int insertSpaces(char[] inArray, int position, int number){
	for(int i=0;i<number;i++){
	    inArray[position] = '\u0020';
	    position++;
	}
	return position;
    }
    //######
    //End - Private section.
    //######

    //####################################
    //Instance data.
    //####################################
    private String mappingName = null;
    private int mappingID = -1;
  
    private int[] groups = null;
    private int numberOfGroups = 0;
    private Vector parents = null;
    private Vector children = null;
    private Vector callPathIDSParents = null;
    private Vector callPathIDSChildren = null;
    private boolean callPathObject = false;

    //Color Settings.
    private boolean colorFlag = false;
    private Color color = null;
    private Color specificColor = null;
  
    private double[] doubleList = null;
    private int maxNumberOfCalls = 0;
    private int maxNumberOfSubRoutines = 0;
    private int maxUserEventNumberValue = 0;
    private double maxUserEventMinValue = 0;
    private double maxUserEventMaxValue = 0;
    private double maxUserEventMeanValue = 0;
    private double meanNumberOfCalls = 0;
    private double meanNumberOfSubRoutines = 0;
  
    //Drawing coordinates for this Global mapping element.
    private int xBeg;
    private int xEnd;
    private int yBeg;
    private int yEnd;
  
    private boolean meanValuesSet = false;
    private boolean groupsSet = false;
  
    //Instance values used to calculate the mean values for derived values (such as flops)
    private int counter = 0;
    private double totalExclusiveValue = 0;
    private double totalInclusiveValue = 0;

    private Vector members = null;
    //####################################
    //End - Instance data.
    //####################################
}
