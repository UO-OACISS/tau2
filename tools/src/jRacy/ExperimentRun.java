/*
	StaticSystemData.java
	
	
	Title:			jRacy
	Author:			Robert Bell
	Description:	This class is the heart of Racy's static data system.
					This class is rather an ongoing project.  Much work needs
					to be done with respect to data format.
					The use of tokenizers here could impact the performance
					with large data sets, but for now, this will be sufficient.
					The naming and creation of the tokenizers has been done mainly
					to improve the readability of the code.
					
					It must also be noted that the correct funtioning of this
					class is heavily dependent on the format of the pprof -d format.
					It is NOT resistant to change in that format at all.
*/

package jRacy;

import java.io.*;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;

public class ExperimentRun implements Serializable
{
	//Constructor.
	public ExperimentRun()
	{
		globalMapping = new GlobalMapping(this);
		systemEvents = new SystemEvents();
		StaticServerList = new Vector();
		positionOfName = -1;
		positionOfUserEventName = -1;
		counterName = null;
		heading = null;
		userEventHeading = null;
		isUserEventHeadingSet = false;
	}
	
	public ColorChooser getColorChooser(){
		return clrChooser;
	}
	
	public Preferences getPreferences(){
		return preferences;
	}
	
	public void setProfilePathName(String inString){
		profilePathName = inString;
	}
	
	public void setProfilePathNameReverse(String inString){
		profilePathNameReverse = inString;
	}
	
	public String getProfilePathName(){
		return profilePathName;
	}
	
	public String getProfilePathNameReverse(){
		return profilePathNameReverse;
	}
	
	//This sets the name for the run.  The run name is different
	//from the list of run value names.  A run has one run name
	//,but can have multiple run value names.
	public void setRunName(String inString){
		runName = inString;
	}
	
	
	public void addRunValueName(String inString){
		runValueNameList.add(inString);
	}
	
	public Vector getRunValueNameList(){
		return runValueNameList;
	}
	
	public boolean isExperimentRunValueNamePresent(String inString){
		return false;
	}
	
	public int getExperimentRunValueNamePosition(String inString){
		
		int counter = 0;
		
		for(Enumeration e = runValueNameList.elements(); e.hasMoreElements() ;)
		{
			String tmpString = (String) e.nextElement();
			if(inString.equals(tmpString.toString()))
				return counter;
			counter++;
		}
		
		return -1;
	}
	
	public String toString(){
		return runName;
	}
	
	public Vector getStaticServerList()
	{
		return StaticServerList;
	}
	
	public StaticMainWindow getStaticMainWindow(){
		return sMW;
	}
	
	public void showStaticMainWindow(){
	
		sMW = new StaticMainWindow(this);
	
		sMW.setVisible(true);
		systemEvents.addObserver(sMW);
		
		jRacy.runHasBeenOpened = true;
	}
	
	public SystemEvents getSystemEvents(){
		return systemEvents;
	}
	
	
	public void setCurRunValLoc(int inInt){
		currentRunValueLocation = inInt;
	}
	
	public void setCurRunValLoc(String inString){
		int tmpInt = this.getExperimentRunValueNamePosition(inString);
		currentRunValueLocation = tmpInt;
	}
	
	public int getCurRunValLoc(){
		return currentRunValueLocation;
	}
	
	
	public void addDefaultToVectors(){
		maxMeanInclusiveValueList.add(new Double(0));
		maxMeanExclusiveValueList.add(new Double(0));
		maxMeanInclusivePercentValueList.add(new Double(0));
		maxMeanExclusivePercentValueList.add(new Double(0));
		maxMeanUserSecPerCallList.add(new Double(0));
	}
	
	//The following funtion initializes the GlobalMapping object.
	//Since we are in the static mode, the number of mappings is known,
	//therefore, the appropriate number of GlobalMappingElements are created.
	void initializeGlobalMapping(int inNumberOfMappings, int mappingSelection)
	{
		for(int i=0; i<inNumberOfMappings; i++)
		{
			//globalMapping.addGlobalMapping("Error ... the mapping name has not been set!");
			globalMapping.addGlobalMapping(null, mappingSelection);
		}
	}
	
	//Rest of the public functions.
	GlobalMapping getGlobalMapping()
	{
		return globalMapping;
	}
	
	public String getCounterName()
	{
		return (String) runValueNameList.elementAt(currentRunValueLocation);
	}
	
	//The core public function of this class.  It reads pprof dump files ... that is pprof
	//run with the -d option.  If any changes occur to the "pprof -d" file output format,
	//the working of this function might be affected.
	public void buildStaticData(boolean firstRun, File inFile)
	{
		String inputString = null;
		try
		{
			BufferedReader br = new BufferedReader(new FileReader(inFile));
			
			//Some useful strings.
			//String inputString;
			String tokenString;
			String mappingNameString = null;
			String groupNamesString = null;
			String userEventNameString = null;
			
			StringTokenizer genericTokenizer;
			
			int mappingID = -1;
			int userEventID = -1;
			double value = -1;
			double percentValue = -1;
			int node = -1;
			int context = -1;
			int thread = -1;
			
			GlobalMappingElement tmpGlobalMappingElement;
			
			GlobalServer currentGlobalServer = null;
			GlobalContext currentGlobalContext = null;
			GlobalThread currentGlobalThread = null;
			GlobalThreadDataElement tmpGlobalThreadDataElement = null;
			
			int lastNode = -1;
			int lastContext = -1;
			int lastThread = -1;
			
			int counter = 0;
			
			
			//A loop counter.
			bSDCounter = 0;
			
			//Another one.
			int i=0;
			int maxID = 0;
			//Read in the file line by line!
			while((inputString = br.readLine()) != null)
			{	
				//Skip over processing the first line ... not needed.
				if(bSDCounter>0)
				{
					//Set up some tokenizers.
					//I want one that will first parse the line so that I can look for certian words.
					genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
					
					//Now I want to search for the tokens which interest me.
					if(!(bSDCounter==1))
					{	
						//Now, skip line three!
						if(!(bSDCounter==2))
						{
							
							//A lot of work goes on in this section.  Certain lines are searched for,
							//and then action is taken depending on those lines.
							
							//Check to See if the String begins with a t.
							if(checkForBeginningT(inputString))
							{
								counter++;
								if(checkForExclusiveWithTOrM(inputString))
								{	
									//Grab the mapping name.
									mappingNameString = getMappingName(inputString);
									
									//Grab the mapping ID.
									mappingID = getMappingID(inputString);
									if(mappingID > maxID)
										maxID = mappingID;
									
									if(firstRun){	
										//Grab the group names.
										groupNamesString = getGroupNames(inputString);
										if(groupNamesString != null){
											StringTokenizer st = new StringTokenizer(groupNamesString, " |");
										    while (st.hasMoreTokens()){
										         String tmpString = st.nextToken();
										         if(tmpString != null){
										         	//The potential new group is added here.  If the group is already present, the the addGlobalMapping
										         	//function will just return the already existing group id.  See the GlobalMapping class for more details.
										         	int tmpInt = globalMapping.addGlobalMapping(tmpString, 1);
										         	//The group is either already present, or has just been added in the above line.  Now, using the addGroup
										         	//function, update this mapping to be a member of this group.
										         	globalMapping.addGroup(mappingID, tmpInt, 0);
										         	if((tmpInt != -1) && (jRacy.debugIsOn))
										         		System.out.println("Adding " + tmpString + " group with id: " + tmpInt + " to mapping: " + mappingNameString);
										      	 }   	
											}    
										}
									}
									
									if(firstRun){
										//Now that we have the mapping name and id, fill in the global mapping element
										//for this mapping.  I am assuming here that pprof's output lists only the
										//global ids.
										if(!(globalMapping.setMappingNameAt(mappingNameString, mappingID, 0)))
											System.out.println("There was an error adding mapping to the global mapping");
									}
									
									//Grab the value.
									value = getValue(inputString);
									
									//Set the value for this mapping.
									if(!(globalMapping.setTotalExclusiveValueAt(currentRunValueLocation, value, mappingID, 0)))
										System.out.println("There was an error setting Exc/Inc total time");	
								}
								else if(checkForInclusiveWithTOrM(inputString))
								{
									//Grab the mapping ID.
									mappingID = getMappingID(inputString);
									//Grab the value.
									value = getValue(inputString);
									//Set the value for this mapping.
									if(!(globalMapping.setTotalInclusiveValueAt(currentRunValueLocation, value, mappingID, 0)))
										System.out.println("There was an error setting Exc/Inc total time");
								}
							} //End - Check to See if the String begins with a t.
							//Check to See if the String begins with a mt.
							else if(checkForBeginningM(inputString))
							{
								if(checkForExclusiveWithTOrM(inputString))
								{
									//Grab the mapping ID.
									mappingID = getMappingID(inputString);
									//Grab the value.
									value = getValue(inputString);
									percentValue = getPercentValue(inputString);
									
									//Grab the correct global mapping element.
									tmpGlobalMappingElement = globalMapping.getGlobalMappingElement(mappingID, 0);
									
									//Now set the values correctly.
									if((this.getMaxMeanExclusiveValue(currentRunValueLocation)) < value)
									{
										this.setMaxMeanExclusiveValue(currentRunValueLocation, value);
									}
									
									if((this.getMaxMeanExclusivePercentValue(currentRunValueLocation)) < percentValue)
									{
										this.setMaxMeanExclusivePercentValue(currentRunValueLocation, percentValue);
									}
									
									tmpGlobalMappingElement.setMeanExclusiveValue(currentRunValueLocation, value);
									tmpGlobalMappingElement.setMeanExclusivePercentValue(currentRunValueLocation, percentValue);
								}
								else if(checkForInclusiveWithTOrM(inputString))
								{	
									//Grab the mapping ID.
									mappingID = getMappingID(inputString);
									//Grab the value.
									value = getValue(inputString);
									percentValue = getPercentValue(inputString);
									
									//Grab the correct global mapping element.
									tmpGlobalMappingElement = globalMapping.getGlobalMappingElement(mappingID, 0);
									
									//Now set the values correctly.
									if((this.getMaxMeanInclusiveValue(currentRunValueLocation)) < value)
									{
										this.setMaxMeanInclusiveValue(currentRunValueLocation, value);
									}
									
									if((this.getMaxMeanInclusivePercentValue(currentRunValueLocation)) < percentValue)
									{
										this.setMaxMeanInclusivePercentValue(currentRunValueLocation, percentValue);
									}
									
									tmpGlobalMappingElement.setMeanInclusiveValue(currentRunValueLocation, value);
									tmpGlobalMappingElement.setMeanInclusivePercentValue(currentRunValueLocation, percentValue);
									
									//Set the total stat string.
									//The next string in the file should be the correct one.  Assume it.
									//If the file format changes, we are all screwed anyway.
									inputString = br.readLine();
									//Set the total stat string.
									
									double numberOfCalls = getNumberOfCalls(inputString);
									double numberOfSubRoutines = getNumberOfSubRoutines(inputString);
									double userSecPerCall = getUserSecPerCall(inputString);
									
									//Now set the values correctly.
									if((this.getMaxMeanNumberOfCalls()) < numberOfCalls)
									{
										this.setMaxMeanNumberOfCalls(numberOfCalls);
									}
									
									if((this.getMaxMeanNumberOfSubRoutines()) < numberOfSubRoutines)
									{
										this.setMaxMeanNumberOfSubRoutines(numberOfSubRoutines);
									}
									
									if((this.getMaxMeanUserSecPerCall(currentRunValueLocation)) < userSecPerCall)
									{
										this.setMaxMeanUserSecPerCall(currentRunValueLocation, userSecPerCall);
									}
										
									tmpGlobalMappingElement.setMeanNumberOfCalls(numberOfCalls);
									tmpGlobalMappingElement.setMeanNumberOfSubRoutines(numberOfSubRoutines);
									tmpGlobalMappingElement.setMeanUserSecPerCall(currentRunValueLocation, userSecPerCall);
									
									tmpGlobalMappingElement.setMeanTotalStatString(currentRunValueLocation, inputString);
									tmpGlobalMappingElement.setMeanValuesSet(true);
									//Now extract the other info from this string.
								}
							}//End - Check to See if the String begins with a m.
							//String does not begin with either an m or a t, the rest of the checks go here.
							else
							{
								///GOTTOHERE
								
								if(checkForExclusive(inputString))
								{
									if(firstRun){
										//Grab the mapping ID.
										mappingID = getMappingID(inputString);
										//Grab the value.
										value = getValue(inputString);
										
										percentValue = getPercentValue(inputString);
										
										//Update the max values if required.
										//Grab the correct global mapping element.
										tmpGlobalMappingElement = globalMapping.getGlobalMappingElement(mappingID, 0);
										
										if((tmpGlobalMappingElement.getMaxExclusiveValue(currentRunValueLocation)) < value)
											tmpGlobalMappingElement.setMaxExclusiveValue(currentRunValueLocation, value);
											
										if((tmpGlobalMappingElement.getMaxExclusivePercentValue(currentRunValueLocation)) < percentValue)
											tmpGlobalMappingElement.setMaxExclusivePercentValue(currentRunValueLocation, percentValue);
										
										//Print out the node,context,thread.
										node = getNode(inputString, false);
										context = getContext(inputString, false);
										thread = getThread(inputString, false);
										//Now the complicated part.  Setting up the node,context,thread data.
										
										
										//These first two if statements force a change if the current node or
										//current context changes from the last, but without a corresponding change
										//in the thread number.  For example, if we have the sequence:
										//0,0,0 - 1,0,0 - 2,0,0 or 0,0,0 - 0,1,0 - 1,0,0.
										if(lastNode != node)
										{
											lastContext = -1;
											lastThread = -1;
										}
										
										if(lastContext != context)
										{
											lastThread = -1;
										}
										
										if(lastThread != thread)
										{
										
											if(thread == 0)
											{
												//Create a new thread ... and set it to be the current thread.
												currentGlobalThread = new GlobalThread();
												//Add the correct number of global thread data elements.
												for(i=0;i<numberOfMappings;i++)
												{
													GlobalThreadDataElement tmpRef = null;
													
													//Add it to the currentGlobalThreadObject.
													currentGlobalThread.addThreadDataElement(tmpRef);
												}
												
												//Update the thread number.
												lastThread = thread;
												
												//Set the appropriate global thread data element.
												Vector tmpVector = currentGlobalThread.getThreadDataList();
												GlobalThreadDataElement tmpGTDE = null;
												
												tmpGTDE = (GlobalThreadDataElement) tmpVector.elementAt(mappingID);
												
												if(tmpGTDE == null)
												{
													tmpGTDE = new GlobalThreadDataElement(this, false);
													tmpGTDE.setMappingID(mappingID);
													currentGlobalThread.addThreadDataElement(tmpGTDE, mappingID);
												}
												tmpGTDE.setMappingExists();
												tmpGTDE.setExclusiveValue(currentRunValueLocation, value);
												tmpGTDE.setExclusivePercentValue(currentRunValueLocation, percentValue);
												//Now check the max values on this thread.
												if((currentGlobalThread.getMaxExclusiveValue(currentRunValueLocation)) < value)
													currentGlobalThread.setMaxExclusiveValue(currentRunValueLocation, value);
												if((currentGlobalThread.getMaxExclusivePercentValue(currentRunValueLocation)) < value)
													currentGlobalThread.setMaxExclusivePercentValue(currentRunValueLocation, percentValue);
												
												//Check to see if the context is zero.
												if(context == 0)
												{
													//Create a new context ... and set it to be the current context.
													currentGlobalContext = new GlobalContext("Context Name Not Set!");
													//Add the current thread
													currentGlobalContext.addThread(currentGlobalThread);
													
													//Create a new server ... and set it to be the current server.
													currentGlobalServer = new GlobalServer("Server Name Not Set!");
													//Add the current context.
													currentGlobalServer.addContext(currentGlobalContext);
													//Add the current server.
													StaticServerList.addElement(currentGlobalServer);
													
													//Update last context and last node.
													lastContext = context;
													lastNode = node;
												}
												else
												{
													//Context number is not zero.  Create a new context ... and set it to be current.
													currentGlobalContext = new GlobalContext("Context Name Not Set!");
													//Add the current thread
													currentGlobalContext.addThread(currentGlobalThread);
													
													//Add the current context.
													currentGlobalServer.addContext(currentGlobalContext);
													
													//Update last context and last node.
													lastContext = context;
												}
													
												
												
											}
											else
											{
												//Thread number is not zero.  Create a new thread ... and set it to be the current thread.
												currentGlobalThread = new GlobalThread();
												//Add the correct number of global thread data elements.
												for(i=0;i<numberOfMappings;i++)
												{
													GlobalThreadDataElement tmpRef = null;
													
													//Add it to the currentGlobalThreadObject.
													currentGlobalThread.addThreadDataElement(tmpRef);
												}
												
												//Update the thread number.
												lastThread = thread;
												
												//Not thread changes.  Just set the appropriate global thread data element.
												Vector tmpVector = currentGlobalThread.getThreadDataList();
												GlobalThreadDataElement tmpGTDE = null;
												tmpGTDE = (GlobalThreadDataElement) tmpVector.elementAt(mappingID);
												
												
												if(tmpGTDE == null)
												{
													tmpGTDE = new GlobalThreadDataElement(this, false);
													tmpGTDE.setMappingID(mappingID);
													currentGlobalThread.addThreadDataElement(tmpGTDE, mappingID);
												}
												
												tmpGTDE.setMappingExists();
												tmpGTDE.setExclusiveValue(currentRunValueLocation, value);
												tmpGTDE.setExclusivePercentValue(currentRunValueLocation, percentValue);
												//Now check the max values on this thread.
												if((currentGlobalThread.getMaxExclusiveValue(currentRunValueLocation)) < value)
													currentGlobalThread.setMaxExclusiveValue(currentRunValueLocation, value);
												if((currentGlobalThread.getMaxExclusivePercentValue(currentRunValueLocation)) < value)
													currentGlobalThread.setMaxExclusivePercentValue(currentRunValueLocation, percentValue);
												
												//Add the current thread
												currentGlobalContext.addThread(currentGlobalThread);
											}
										}
										else
										{
											//Not thread changes.  Just set the appropriate global thread data element.
											Vector tmpVector = currentGlobalThread.getThreadDataList();
											GlobalThreadDataElement tmpGTDE = null;
											tmpGTDE = (GlobalThreadDataElement) tmpVector.elementAt(mappingID);
										
												
											if(tmpGTDE == null)
											{
												tmpGTDE = new GlobalThreadDataElement(this, false);
												tmpGTDE.setMappingID(mappingID);
												currentGlobalThread.addThreadDataElement(tmpGTDE, mappingID);
											}
											
											tmpGTDE.setMappingExists();
											tmpGTDE.setExclusiveValue(currentRunValueLocation, value);
											tmpGTDE.setExclusivePercentValue(currentRunValueLocation, percentValue);
											//Now check the max values on this thread.
											if((currentGlobalThread.getMaxExclusiveValue(currentRunValueLocation)) < value)
												currentGlobalThread.setMaxExclusiveValue(currentRunValueLocation, value);
											if((currentGlobalThread.getMaxExclusivePercentValue(currentRunValueLocation)) < percentValue)
												currentGlobalThread.setMaxExclusivePercentValue(currentRunValueLocation, percentValue);
										}
									}
									else{
										//Grab the mapping ID.
										mappingID = getMappingID(inputString);
										//Grab the value.
										value = getValue(inputString);
										percentValue = getPercentValue(inputString);
										//Update the max values if required.
										//Grab the correct global mapping element.
										
										tmpGlobalMappingElement = globalMapping.getGlobalMappingElement(mappingID, 0);
										
										if((tmpGlobalMappingElement.getMaxExclusiveValue(currentRunValueLocation)) < value)
											tmpGlobalMappingElement.setMaxExclusiveValue(currentRunValueLocation, value);
											
										if((tmpGlobalMappingElement.getMaxExclusivePercentValue(currentRunValueLocation)) < percentValue)
											tmpGlobalMappingElement.setMaxExclusivePercentValue(currentRunValueLocation, percentValue);
										
										//Print out the node,context,thread.
										node = getNode(inputString, false);
										context = getContext(inputString, false);
										thread = getThread(inputString, false);
										
										
										//Find the correct global thread data element.
										GlobalServer tmpGS = (GlobalServer) StaticServerList.elementAt(node);
										Vector tmpGlobalContextList = tmpGS.getContextList();
										GlobalContext tmpGC = (GlobalContext) tmpGlobalContextList.elementAt(context);
										Vector tmpGlobalThreadList = tmpGC.getThreadList();
										GlobalThread tmpGT = (GlobalThread) tmpGlobalThreadList.elementAt(thread);
										Vector tmpGlobalThreadDataElementList = tmpGT.getThreadDataList();
										
										
										GlobalThreadDataElement tmpGTDE = (GlobalThreadDataElement) tmpGlobalThreadDataElementList.elementAt(mappingID);
									
										tmpGTDE.setExclusiveValue(currentRunValueLocation, value);
										tmpGTDE.setExclusivePercentValue(currentRunValueLocation, percentValue);
										
										//Now check the max values on this thread.
										if((tmpGT.getMaxExclusiveValue(currentRunValueLocation)) < value)
											tmpGT.setMaxExclusiveValue(currentRunValueLocation, value);
										if((tmpGT.getMaxExclusivePercentValue(currentRunValueLocation)) < percentValue)
											tmpGT.setMaxExclusivePercentValue(currentRunValueLocation, percentValue);
											
									}
								}
								else if(checkForInclusive(inputString))
								{
									//Grab the mapping ID.
									mappingID = getMappingID(inputString);
									//Grab the value.
									value = getValue(inputString);
									percentValue = getPercentValue(inputString);
									
									
									//Update the max values if required.
									//Grab the correct global mapping element.
									tmpGlobalMappingElement = globalMapping.getGlobalMappingElement(mappingID, 0);
									
									if((tmpGlobalMappingElement.getMaxInclusiveValue(currentRunValueLocation)) < value)
										tmpGlobalMappingElement.setMaxInclusiveValue(currentRunValueLocation, value);
										
									if((tmpGlobalMappingElement.getMaxInclusivePercentValue(currentRunValueLocation)) < percentValue)
										tmpGlobalMappingElement.setMaxInclusivePercentValue(currentRunValueLocation, percentValue);
									
									//Print out the node,context,thread.
									node = getNode(inputString, false);
									context = getContext(inputString, false);
									thread = getThread(inputString, false);
									
									//Find the correct global thread data element.
									GlobalServer tmpGS = (GlobalServer) StaticServerList.elementAt(node);
									Vector tmpGlobalContextList = tmpGS.getContextList();
									GlobalContext tmpGC = (GlobalContext) tmpGlobalContextList.elementAt(context);
									Vector tmpGlobalThreadList = tmpGC.getThreadList();
									GlobalThread tmpGT = (GlobalThread) tmpGlobalThreadList.elementAt(thread);
									Vector tmpGlobalThreadDataElementList = tmpGT.getThreadDataList();
									
									GlobalThreadDataElement tmpGTDE = (GlobalThreadDataElement) tmpGlobalThreadDataElementList.elementAt(mappingID);
									//Now set the inclusive value!
									
									if(tmpGTDE == null)
									{
										System.out.println("Don't think I ever get here.  Check the logic to make sure.");
										tmpGTDE = new GlobalThreadDataElement(this, false);
										tmpGTDE.setMappingID(mappingID);
										currentGlobalThread.addThreadDataElement(tmpGTDE, mappingID);
									}
									
									tmpGTDE.setInclusiveValue(currentRunValueLocation, value);
									tmpGTDE.setInclusivePercentValue(currentRunValueLocation, percentValue);
									//Now check the max values on this thread.
									if((tmpGT.getMaxInclusiveValue(currentRunValueLocation)) < value)
										tmpGT.setMaxInclusiveValue(currentRunValueLocation, value);
									if((tmpGT.getMaxInclusivePercentValue(currentRunValueLocation)) < percentValue)
										tmpGT.setMaxInclusivePercentValue(currentRunValueLocation, percentValue);
									
									
									//Get the number of calls and number of sub routines, and then set the total stat string.
									//The next string in the file should be the correct one.  Assume it.
									//If the file format changes, we are all screwed anyway.
									inputString = br.readLine();
									
									int numberOfCalls = (int) getNumberOfCalls(inputString);
									int numberOfSubRoutines = (int) getNumberOfSubRoutines(inputString);
									double userSecPerCall = (double) getUserSecPerCall(inputString);
									//Update max values.
									if((tmpGlobalMappingElement.getMaxNumberOfCalls()) < numberOfCalls)
										tmpGlobalMappingElement.setMaxNumberOfCalls(numberOfCalls);
									if((tmpGlobalMappingElement.getMaxNumberOfSubRoutines()) < numberOfSubRoutines)
										tmpGlobalMappingElement.setMaxNumberOfSubRoutines(numberOfSubRoutines);
									if((tmpGlobalMappingElement.getMaxUserSecPerCall(currentRunValueLocation)) < userSecPerCall)
										tmpGlobalMappingElement.setMaxUserSecPerCall(currentRunValueLocation, userSecPerCall);
										
									if((tmpGT.getMaxNumberOfCalls()) < numberOfCalls)
										tmpGT.setMaxNumberOfCalls(numberOfCalls);
									if((tmpGT.getMaxNumberOfSubRoutines()) < numberOfSubRoutines)
										tmpGT.setMaxNumberOfSubRoutines(numberOfSubRoutines);
									if((tmpGT.getMaxUserSecPerCall(currentRunValueLocation)) < userSecPerCall)
										tmpGT.setMaxUserSecPerCall(currentRunValueLocation, userSecPerCall);

									tmpGTDE.setNumberOfCalls(numberOfCalls);
									tmpGTDE.setNumberOfSubRoutines(numberOfSubRoutines);
									tmpGTDE.setUserSecPerCall(currentRunValueLocation, userSecPerCall);
									//Set the total stat string.
									tmpGTDE.setTStatString(currentRunValueLocation, inputString);
									//Now extract the other info from this string.
									
								}
								else if(checkForUserEvents(inputString))
								{
									//The first time a user event string is encountered, get the number of user events and 
									//initialize the global mapping for mapping position 2.
									if(!(this.userEventsPresent())){
										//Get the number of user events.
										numberOfUserEvents = getNumberOfUserEvents(inputString);
										initializeGlobalMapping(numberOfUserEvents, 2);
										if(jRacy.debugIsOn){
											System.out.println("The number of user events defined is: " + numberOfUserEvents);
											System.out.println("Initializing mapping selection 2 (The loaction of the user event mapping) for " +
																numberOfUserEvents + " mappings.");
										}
									} 
									
									//The first line will be the user event heading ... get it.
									inputString = br.readLine();
									userEventHeading = inputString;
									
									positionOfUserEventName = inputString.indexOf("Event Name");
									
									//Find the correct global thread data element.
									GlobalServer tmpGSUE = null;
									Vector tmpGlobalContextListUE = null;;
									GlobalContext tmpGCUE = null;;
									Vector tmpGlobalThreadListUE = null;;
									GlobalThread tmpGTUE = null;;
									Vector tmpGlobalThreadDataElementListUE = null;
									
									//Now that we know how many user events to expect, we can grab that number of lines.
									for(int j=0; j<numberOfUserEvents; j++)
									{
										inputString = br.readLine();
										
										//Initialize the user list for this thread.
										if(j == 0)
										{
											//Note that this works correctly because we process the user events in a different manner.
											//ALL the user events for each THREAD NODE are processed in the above for-loop.  Therefore,
											//the below for-loop is only run once on each THREAD NODE.  If you do not believe it, turn on
											//debugging.
											if(jRacy.debugIsOn)
												System.out.println("Creating the list for node,context,thread: " +node+","+context+","+thread);
											
											//Get the node,context,thread.
											node = getNode(inputString, true);
											context = getContext(inputString, true);
											thread = getThread(inputString, true);
											
											//Find the correct global thread data element.
											tmpGSUE = (GlobalServer) StaticServerList.elementAt(node);
											tmpGlobalContextListUE = tmpGSUE.getContextList();
											tmpGCUE = (GlobalContext) tmpGlobalContextListUE.elementAt(context);
											tmpGlobalThreadListUE = tmpGCUE.getThreadList();
											tmpGTUE = (GlobalThread) tmpGlobalThreadListUE.elementAt(thread);
											
											
											if(firstRun){
												for(int k=0; k<numberOfUserEvents; k++)
												{
													tmpGTUE.addUserThreadDataElement(new GlobalThreadDataElement(this, true));
												}
											}
											
											tmpGlobalThreadDataElementListUE = tmpGTUE.getUserThreadDataList();
										}
										
										
										//Extract all the information out of the string that I need.
										
										//Grab the mapping ID.
										userEventID = getUserEventID(inputString);
										
										//Only need to set the name in the global mapping once.
										if(!(this.userEventsPresent())){
											//Grab the mapping name.
											userEventNameString = getUserEventName(inputString);
											if(!(globalMapping.setMappingNameAt(userEventNameString, userEventID, 2)))
												System.out.println("There was an error adding mapping to the global mapping");
											if(jRacy.debugIsOn){
												System.out.println("The user event ID: " + userEventID);
												System.out.println("The user event name is: " + userEventNameString);
											}
										}
										
										int userEventNumberValue = getUENValue(inputString);
										double userEventMinValue = getUEMinValue(inputString);
										double userEventMaxValue = getUEMaxValue(inputString);
										double userEventMeanValue = getUEMeanValue(inputString);
										//Update the max values if required.
										//Grab the correct global mapping element.
										tmpGlobalMappingElement = globalMapping.getGlobalMappingElement(userEventID, 2);
										if((tmpGlobalMappingElement.getMaxUserEventNumberValue(currentRunValueLocation)) < userEventNumberValue)
											tmpGlobalMappingElement.setMaxUserEventNumberValue(currentRunValueLocation, userEventNumberValue);
										if((tmpGlobalMappingElement.getMaxUserEventMinValue(currentRunValueLocation)) < userEventMinValue)
											tmpGlobalMappingElement.setMaxUserEventMinValue(currentRunValueLocation, userEventMinValue);
										if((tmpGlobalMappingElement.getMaxUserEventMaxValue(currentRunValueLocation)) < userEventMaxValue)
											tmpGlobalMappingElement.setMaxUserEventMaxValue(currentRunValueLocation, userEventMaxValue);
										if((tmpGlobalMappingElement.getMaxUserEventMeanValue(currentRunValueLocation)) < userEventMeanValue)
											tmpGlobalMappingElement.setMaxUserEventMeanValue(currentRunValueLocation, userEventMeanValue);
										
										GlobalThreadDataElement tmpGTDEUE = (GlobalThreadDataElement) tmpGlobalThreadDataElementListUE.elementAt(userEventID);
										//Ok, now set the instance data elements.
										tmpGTDEUE.setUserEventID(userEventID);
										tmpGTDEUE.setUserEventNumberValue(currentRunValueLocation, userEventNumberValue);
										tmpGTDEUE.setUserEventMinValue(currentRunValueLocation, userEventMinValue);
										tmpGTDEUE.setUserEventMaxValue(currentRunValueLocation, userEventMaxValue);
										tmpGTDEUE.setUserEventMeanValue(currentRunValueLocation, userEventMeanValue);
										//Ok, now get the next string as that is the stat string for this event.
										inputString = br.readLine();
										tmpGTDEUE.setUserEventStatString(currentRunValueLocation, inputString);
									}
									
									//Now set the userEvents flag.
									this.setUserEventsPresent(true);
								}
							//End - String does not begin with either an m or a t, the rest of the checks go here.
							}
						}
						else
						{
							heading = inputString;
							positionOfName = inputString.indexOf("name");
						}
					
					}
					else
					{
						
						//This is the second line of the file.  It's first token will
						//be the number of mappings present.  Get it.
						tokenString = genericTokenizer.nextToken();
						
						
						if(firstRun){
						//Set the number of mappings.
						numberOfMappings = Integer.parseInt(tokenString);
						}
						else{
							if(numberOfMappings != Integer.parseInt(tokenString)){
								System.out.println("***********************");
								System.out.println("The number of mappings does not match!!!");
								System.out.println("");
								System.out.println("To add to an existing run, you must be choosing from");
								System.out.println("a list of multiple metrics from that same run!!!");
								System.out.println("***********************");
								
								return;
							}
						}
						//Now initialize the global mapping with the correct number of mappings for mapping position 0.
						initializeGlobalMapping(Integer.parseInt(tokenString), 0);
						
						//Set the counter name.
						counterName = getCounterName(inputString);
						
						//Ok, we are adding a counter name.  Since nothing much has happened yet, it is a
						//good place to initialize a few things.
						
						//Need to call addDefaultToVectors() on all objects that require it.
						this.addDefaultToVectors();
						
						//Only need to call addDefaultToVectors() if not the first run.
						if(!firstRun){
							for(Enumeration e1 = (this.globalMapping.getMapping(0)).elements(); e1.hasMoreElements() ;){
								
								GlobalMappingElement tmpGME = (GlobalMappingElement) e1.nextElement();
								tmpGME.addDefaultToVectors();
							}
							
							for(Enumeration e2 = (this.globalMapping.getMapping(2)).elements(); e2.hasMoreElements() ;){
								
								GlobalMappingElement tmpGME = (GlobalMappingElement) e2.nextElement();
								tmpGME.addDefaultToVectors();
							}
							
							
							GlobalServer tmpGlobalServer;
							GlobalContext tmpGlobalContext;
							GlobalThread tmpGlobalThread;
							GlobalThreadDataElement tmpGTDElement;
							
							Vector tmpContextList;
							Vector tmpThreadList;
							Vector tmpThreadDataList;
							
							//Get a reference to the global data.
							Vector tmpVector = this.getStaticServerList();
							
							for(Enumeration e3 = tmpVector.elements(); e3.hasMoreElements() ;)
							{
								tmpGlobalServer = (GlobalServer) e3.nextElement();
								
								//Enter the context loop for this server.
								tmpContextList = tmpGlobalServer.getContextList();
									
								for(Enumeration e4 = tmpContextList.elements(); e4.hasMoreElements() ;)
								{
									tmpGlobalContext = (GlobalContext) e4.nextElement();
										
									//Enter the thread loop for this context.
									tmpThreadList = tmpGlobalContext.getThreadList();
									for(Enumeration e5 = tmpThreadList.elements(); e5.hasMoreElements() ;)
									{
										tmpGlobalThread = (GlobalThread) e5.nextElement();
										tmpGlobalThread.addDefaultToVectors();
										
										tmpThreadDataList = tmpGlobalThread.getThreadDataList();
										for(Enumeration e6 = tmpThreadDataList.elements(); e6.hasMoreElements() ;)
										{
											tmpGTDElement = (GlobalThreadDataElement) e6.nextElement();
											
											//Only want to add an element if this mapping existed on this thread.
											//Check for this.
											if(tmpGTDElement != null)
											{
												tmpGTDElement.addDefaultToVectors();
											}
										}
										
										tmpThreadDataList = tmpGlobalThread.getUserThreadDataList();
										for(Enumeration e7 = tmpThreadDataList.elements(); e7.hasMoreElements() ;)
										{
											tmpGTDElement = (GlobalThreadDataElement) e7.nextElement();
											
											//Only want to add an element if this mapping existed on this thread.
											//Check for this.
											if(tmpGTDElement != null)
											{
												tmpGTDElement.addDefaultToVectorsUE();
											}
										}
									}
								}
							}
						}
						
						//Now set the counter name.
						
						if(counterName == null)
							counterName = new String("Default");
							
						this.addRunValueName(counterName);
						this.setCurRunValLoc(counterName);
						
						System.out.println("The number of mappings in the system is: " + tokenString);
					}
						
				}
				//Increment the loop counter.
			bSDCounter++;
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD01");
		}
	}
	
	public void applyOperation(String tmpString1, String tmpString2, String inOperation){
	
		int opA = this.getExperimentRunValueNamePosition(tmpString1);
		int opB = this.getExperimentRunValueNamePosition(tmpString2);
		String tmpString3 = null;
		
		
		int operation = -1;
		
		if(inOperation.equals("Add")){
			operation = 0;
			tmpString3 = tmpString1 + " + " + tmpString2;
			}
		else if(inOperation.equals("Subtract")){
			operation = 1;
			tmpString3 = tmpString1 + " - " + tmpString2;
			}
		else if(inOperation.equals("Multiply")){
			operation = 2;
			tmpString3 = tmpString1 + " * " + tmpString2;
			}
		else if(inOperation.equals("Divide")){
			operation = 3;
			tmpString3 = tmpString1 + " / " + tmpString2;
			}
		else{
			System.out.println("Wrong operation type");
		}
		
		//Pop up the dialog if there is already an experiment with this name.
		if(this.isExperimentRunValueNamePresent(tmpString3)){
			JOptionPane.showMessageDialog(null, "This metric has alread been computed!", "Warning!"
															  ,JOptionPane.ERROR_MESSAGE);
			return;
		}
			
		this.addRunValueName(tmpString3);
		this.setCurRunValLoc(tmpString3);
	
		for(Enumeration e1 = (this.globalMapping.getMapping(0)).elements(); e1.hasMoreElements() ;){
									
			GlobalMappingElement tmpGME = (GlobalMappingElement) e1.nextElement();
			tmpGME.addDefaultToVectors();
			tmpGME.setTotalExclusiveValue(0);
			tmpGME.setTotalInclusiveValue(0);
			tmpGME.setCounter(0);
		}
		
		for(Enumeration e2 = (this.globalMapping.getMapping(2)).elements(); e2.hasMoreElements() ;){
			
			GlobalMappingElement tmpGME = (GlobalMappingElement) e2.nextElement();
			tmpGME.addDefaultToVectors();
		}
		
		
		GlobalServer tmpGlobalServer;
		GlobalContext tmpGlobalContext;
		GlobalThread tmpGlobalThread;
		GlobalThreadDataElement tmpGlobalThreadDataElement;
		
		Vector tmpContextList;
		Vector tmpThreadList;
		Vector tmpThreadDataList;
		
		//Get a reference to the global data.
		Vector tmpVector = this.getStaticServerList();
		
		for(Enumeration e3 = tmpVector.elements(); e3.hasMoreElements() ;)
		{
			tmpGlobalServer = (GlobalServer) e3.nextElement();
			
			//Enter the context loop for this server.
			tmpContextList = tmpGlobalServer.getContextList();
				
			for(Enumeration e4 = tmpContextList.elements(); e4.hasMoreElements() ;)
			{
				tmpGlobalContext = (GlobalContext) e4.nextElement();
					
				//Enter the thread loop for this context.
				tmpThreadList = tmpGlobalContext.getThreadList();
				for(Enumeration e5 = tmpThreadList.elements(); e5.hasMoreElements() ;)
				{
					tmpGlobalThread = (GlobalThread) e5.nextElement();
					tmpGlobalThread.addDefaultToVectors();
					
					tmpThreadDataList = tmpGlobalThread.getThreadDataList();
					for(Enumeration e6 = tmpThreadDataList.elements(); e6.hasMoreElements() ;)
					{
						tmpGlobalThreadDataElement = (GlobalThreadDataElement) e6.nextElement();
						
						//Only want to add an element if this mapping existed on this thread.
						//Check for this.
						if(tmpGlobalThreadDataElement != null)
						{
							int mappingID = tmpGlobalThreadDataElement.getMappingID();
							GlobalMappingElement tmpGlobalMappingElement = globalMapping.getGlobalMappingElement(mappingID, 0);
							tmpGlobalMappingElement.incrementCounter();
							
							tmpGlobalThreadDataElement.addDefaultToVectors();
							
							double tmpDouble1 = 0;
							double tmpDouble2 = 0;
							
							double tmpDouble = 0;
							
							
							switch(operation){
							case(0):
									tmpDouble1 = tmpGlobalThreadDataElement.getExclusiveValue(opA);
									tmpDouble2 = tmpGlobalThreadDataElement.getExclusiveValue(opB);
									
									tmpDouble = tmpDouble1+tmpDouble2;
									tmpGlobalThreadDataElement.setExclusiveValue(currentRunValueLocation, tmpDouble);
									if((tmpGlobalThread.getMaxExclusiveValue(currentRunValueLocation)) < tmpDouble)
										tmpGlobalThread.setMaxExclusiveValue(currentRunValueLocation, tmpDouble);
										
									//Now do the global mapping element exclusive stuff.
									if((tmpGlobalMappingElement.getMaxExclusiveValue(currentRunValueLocation)) < tmpDouble)
											tmpGlobalMappingElement.setMaxExclusiveValue(currentRunValueLocation, tmpDouble);
									tmpGlobalMappingElement.incrementTotalExclusiveValue(tmpDouble);
									
									
									
									tmpDouble1 = tmpGlobalThreadDataElement.getInclusiveValue(opA);
									tmpDouble2 = tmpGlobalThreadDataElement.getInclusiveValue(opB);
									
									tmpDouble = tmpDouble1+tmpDouble2;
									tmpGlobalThreadDataElement.setInclusiveValue(currentRunValueLocation, tmpDouble);
									if((tmpGlobalThread.getMaxInclusiveValue(currentRunValueLocation)) < tmpDouble)
										tmpGlobalThread.setMaxInclusiveValue(currentRunValueLocation, tmpDouble);
										
									//Now do the global mapping element inclusive stuff.
									if((tmpGlobalMappingElement.getMaxInclusiveValue(currentRunValueLocation)) < tmpDouble)
											tmpGlobalMappingElement.setMaxInclusiveValue(currentRunValueLocation, tmpDouble);
									tmpGlobalMappingElement.incrementTotalInclusiveValue(tmpDouble);
						
									break;
							case(1):
									tmpDouble1 = tmpGlobalThreadDataElement.getExclusiveValue(opA);
									tmpDouble2 = tmpGlobalThreadDataElement.getExclusiveValue(opB);
									
									if(tmpDouble1 > tmpDouble2){
										tmpDouble = tmpDouble1 - tmpDouble2;
										tmpGlobalThreadDataElement.setExclusiveValue(currentRunValueLocation, tmpDouble);
										if((tmpGlobalThread.getMaxExclusiveValue(currentRunValueLocation)) < tmpDouble)
											tmpGlobalThread.setMaxExclusiveValue(currentRunValueLocation, tmpDouble);
											
										//Now do the global mapping element exclusive stuff.
										if((tmpGlobalMappingElement.getMaxExclusiveValue(currentRunValueLocation)) < tmpDouble)
											tmpGlobalMappingElement.setMaxExclusiveValue(currentRunValueLocation, tmpDouble);
										tmpGlobalMappingElement.incrementTotalExclusiveValue(tmpDouble);
									}
									
									tmpDouble1 = tmpGlobalThreadDataElement.getInclusiveValue(opA);
									tmpDouble2 = tmpGlobalThreadDataElement.getInclusiveValue(opB);
									
									if(tmpDouble1 > tmpDouble2){
										tmpDouble = tmpDouble1 - tmpDouble2;
										tmpGlobalThreadDataElement.setInclusiveValue(currentRunValueLocation, tmpDouble);
										if((tmpGlobalThread.getMaxInclusiveValue(currentRunValueLocation)) < tmpDouble)
											tmpGlobalThread.setMaxInclusiveValue(currentRunValueLocation, tmpDouble);
											
										//Now do the global mapping element inclusive stuff.
										if((tmpGlobalMappingElement.getMaxInclusiveValue(currentRunValueLocation)) < tmpDouble)
												tmpGlobalMappingElement.setMaxInclusiveValue(currentRunValueLocation, tmpDouble);
										tmpGlobalMappingElement.incrementTotalInclusiveValue(tmpDouble);
									}
									
									break;
							case(2):
									tmpDouble1 = tmpGlobalThreadDataElement.getExclusiveValue(opA);
									tmpDouble2 = tmpGlobalThreadDataElement.getExclusiveValue(opB);
									
									tmpDouble = tmpDouble1*tmpDouble2;
									tmpGlobalThreadDataElement.setExclusiveValue(currentRunValueLocation, tmpDouble);
									if((tmpGlobalThread.getMaxExclusiveValue(currentRunValueLocation)) < tmpDouble)
										tmpGlobalThread.setMaxExclusiveValue(currentRunValueLocation, tmpDouble);
										
									//Now do the global mapping element exclusive stuff.
									if((tmpGlobalMappingElement.getMaxExclusiveValue(currentRunValueLocation)) < tmpDouble)
										tmpGlobalMappingElement.setMaxExclusiveValue(currentRunValueLocation, tmpDouble);
									tmpGlobalMappingElement.incrementTotalExclusiveValue(tmpDouble);
									
									tmpDouble1 = tmpGlobalThreadDataElement.getInclusiveValue(opA);
									tmpDouble2 = tmpGlobalThreadDataElement.getInclusiveValue(opB);
									
									tmpDouble = tmpDouble1*tmpDouble2;
									tmpGlobalThreadDataElement.setInclusiveValue(currentRunValueLocation, tmpDouble);
									if((tmpGlobalThread.getMaxInclusiveValue(currentRunValueLocation)) < tmpDouble)
										tmpGlobalThread.setMaxInclusiveValue(currentRunValueLocation, tmpDouble);
									
									//Now do the global mapping element inclusive stuff.
									if((tmpGlobalMappingElement.getMaxInclusiveValue(currentRunValueLocation)) < tmpDouble)
											tmpGlobalMappingElement.setMaxInclusiveValue(currentRunValueLocation, tmpDouble);
									tmpGlobalMappingElement.incrementTotalInclusiveValue(tmpDouble);
									
									
									break;
							case(3):
									tmpDouble1 = tmpGlobalThreadDataElement.getExclusiveValue(opA);
									tmpDouble2 = tmpGlobalThreadDataElement.getExclusiveValue(opB);
									
									if(tmpDouble2 != 0){
										tmpDouble = tmpDouble1/tmpDouble2;
										
										tmpGlobalThreadDataElement.setExclusiveValue(currentRunValueLocation, tmpDouble);
										if((tmpGlobalThread.getMaxExclusiveValue(currentRunValueLocation)) < tmpDouble)
											tmpGlobalThread.setMaxExclusiveValue(currentRunValueLocation, tmpDouble);
										
										//Now do the global mapping element exclusive stuff.
										if((tmpGlobalMappingElement.getMaxExclusiveValue(currentRunValueLocation)) < tmpDouble)
											tmpGlobalMappingElement.setMaxExclusiveValue(currentRunValueLocation, tmpDouble);
										tmpGlobalMappingElement.incrementTotalExclusiveValue(tmpDouble);
									}
									
									tmpDouble1 = tmpGlobalThreadDataElement.getInclusiveValue(opA);
									tmpDouble2 = tmpGlobalThreadDataElement.getInclusiveValue(opB);
									
									if(tmpDouble2 != 0){
										tmpDouble = tmpDouble1/tmpDouble2;
										tmpGlobalThreadDataElement.setInclusiveValue(currentRunValueLocation, tmpDouble);
										if((tmpGlobalThread.getMaxInclusiveValue(currentRunValueLocation)) < tmpDouble)
											tmpGlobalThread.setMaxInclusiveValue(currentRunValueLocation, tmpDouble);
										
										//Now do the global mapping element inclusive stuff.
										if((tmpGlobalMappingElement.getMaxInclusiveValue(currentRunValueLocation)) < tmpDouble)
											tmpGlobalMappingElement.setMaxInclusiveValue(currentRunValueLocation, tmpDouble);
										tmpGlobalMappingElement.incrementTotalInclusiveValue(tmpDouble);
									}
							
									break;
							default:
									break;
							}
							
							//Try setting a total stat string.
						}
					}
					
					tmpThreadDataList = tmpGlobalThread.getUserThreadDataList();
					for(Enumeration e7 = tmpThreadDataList.elements(); e7.hasMoreElements() ;)
					{
						tmpGlobalThreadDataElement = (GlobalThreadDataElement) e7.nextElement();
						
						//Only want to add an element if this mapping existed on this thread.
						//Check for this.
						if(tmpGlobalThreadDataElement != null)
						{
							tmpGlobalThreadDataElement.addDefaultToVectorsUE();
							//Since for now the user events will remain the same.  Just select one and copy it.
							tmpGlobalThreadDataElement.setUserEventNumberValue(currentRunValueLocation, tmpGlobalThreadDataElement.getUserEventNumberValue(0));
							tmpGlobalThreadDataElement.setUserEventMinValue(currentRunValueLocation, tmpGlobalThreadDataElement.getUserEventMinValue(0));
							tmpGlobalThreadDataElement.setUserEventMaxValue(currentRunValueLocation, tmpGlobalThreadDataElement.getUserEventMaxValue(0));
							tmpGlobalThreadDataElement.setUserEventMeanValue(currentRunValueLocation, tmpGlobalThreadDataElement.getUserEventMeanValue(0));
						}
					}
				}
			}
		}
		
		//Ok,  now compute some mean values.
		for(Enumeration e10 = (this.globalMapping.getMapping(0)).elements(); e10.hasMoreElements() ;){
									
			GlobalMappingElement tmpGME = (GlobalMappingElement) e10.nextElement();
			
			double tmpDouble = (tmpGME.getTotalExclusiveValue())/(tmpGME.getCounter());
			tmpGME.setMeanExclusiveValue(this.getCurRunValLoc(), tmpDouble);
			if((this.getMaxMeanExclusiveValue(currentRunValueLocation) < tmpDouble))
				this.setMaxMeanExclusiveValue(currentRunValueLocation, tmpDouble);
			
			tmpDouble = (tmpGME.getTotalInclusiveValue())/(tmpGME.getCounter());
			tmpGME.setMeanInclusiveValue(this.getCurRunValLoc(), tmpDouble);
			if((this.getMaxMeanInclusiveValue(currentRunValueLocation) < tmpDouble))
				this.setMaxMeanInclusiveValue(currentRunValueLocation, tmpDouble);
		}
		
		for(Enumeration e11 = (this.globalMapping.getMapping(2)).elements(); e11.hasMoreElements() ;){
									
			GlobalMappingElement tmpGME = (GlobalMappingElement) e11.nextElement();
			
			tmpGME.setMaxUserEventNumberValue(currentRunValueLocation, tmpGME.getMaxUserEventNumberValue(0));
			tmpGME.setMaxUserEventMinValue(currentRunValueLocation, tmpGME.getMaxUserEventMinValue(0));
			tmpGME.setMaxUserEventMaxValue(currentRunValueLocation, tmpGME.getMaxUserEventMaxValue(0));
			tmpGME.setMaxUserEventMeanValue(currentRunValueLocation, tmpGME.getMaxUserEventMeanValue(0));
		}
	}
		
	
	//******************************
	//Helper functions for buildStatic data.
	//******************************
	boolean checkForBeginningT(String inString)
	{
		try{
			StringTokenizer checkForBeginningTTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			String tmpString;
			
			tmpString = checkForBeginningTTokenizer.nextToken();
				
			if(tmpString.equals("t"))
				return true;
			else
				return false;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD02");
		}
		
		return false;
	}
	
	boolean checkForBeginningM(String inString)
	{
		
		try{
			StringTokenizer checkForBeginningTTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			String tmpString;
			
			tmpString = checkForBeginningTTokenizer.nextToken();
				
			if(tmpString.equals("m"))
				return true;
			else
				return false;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD03");
		}
		
		return false;
	}
	
	boolean checkForExclusive(String inString)
	{
		
		try{
			//In this function I need to be careful.  If the mapping name contains "excl", I
			//might interpret this line as being the exclusive line when in fact it is not.
			
			//Check for the right string.
			StringTokenizer checkTokenizer = new StringTokenizer(inString," ");
			String tmpString2 = checkTokenizer.nextToken();
			if((tmpString2.indexOf(",")) != -1)
			{
				//Ok, so at least we have the correct string.
				//Now, we want to grab the substring that occurs AFTER the SECOND '"'.
				//At present, pprof does not seem to allow an '"' in the mapping name.  So
				//, I can be assured that I will not find more than two before the "excl" or "incl".
				StringTokenizer checkQuotesTokenizer = new StringTokenizer(inString,"\"");
				
				//Need to get the third token.  Could do it in a loop, just as quick this way.
				String tmpString = checkQuotesTokenizer.nextToken();
				tmpString = checkQuotesTokenizer.nextToken();
				tmpString = checkQuotesTokenizer.nextToken();
				
				//Ok, now, the string in tmpString should include at least "excl" or "incl", and
				//also, the first token should be either "excl" or "incl".
				StringTokenizer checkForExclusiveTokenizer = new StringTokenizer(tmpString, " \t\n\r");
				tmpString = checkForExclusiveTokenizer.nextToken();
					
				//At last, do the check.	
				if(tmpString.equals("excl"))
				{
					return true;
				}
			}
			
			//If here, it means that we are not looking at the correct string or that we did not
			//find a match.  Therefore, return false.
			return false;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD04");
		}
		
		return false;
	}
	
	boolean checkForExclusiveWithTOrM(String inString)
	{
		
		try{
			//In this function I need to be careful.  If the mapping name contains "excl", I
			//might interpret this line as being the exclusive line when in fact it is not.
			
			//Ok, so at least we have the correct string.
			//Now, we want to grab the substring that occurs AFTER the SECOND '"'.
			//At present, pprof does not seem to allow an '"' in the mapping name.  So
			//, I can be assured that I will not find more than two before the "excl" or "incl".
			StringTokenizer checkQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Need to get the third token.  Could do it in a loop, just as quick this way.
			String tmpString = checkQuotesTokenizer.nextToken();
			tmpString = checkQuotesTokenizer.nextToken();
			tmpString = checkQuotesTokenizer.nextToken();
			
			//Ok, now, the string in tmpString should include at least "excl" or "incl", and
			//also, the first token should be either "excl" or "incl".
			StringTokenizer checkForExclusiveTokenizer = new StringTokenizer(tmpString, " \t\n\r");
			tmpString = checkForExclusiveTokenizer.nextToken();
				
			//At last, do the check.	
			if(tmpString.equals("excl"))
			{
				return true;
			}
			
			//If here, it means that we are not looking at the correct string or that we did not
			//find a match.  Therefore, return false.
			return false;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD05");
		}
		
		return false;
	}
	
	boolean checkForInclusive(String inString)
	{
		
		try{
			//In this function I need to be careful.  If the mapping name contains "incl", I
			//might interpret this line as being the inclusive line when in fact it is not.
			
			
			//Check for the right string.
			StringTokenizer checkTokenizer = new StringTokenizer(inString," ");
			String tmpString2 = checkTokenizer.nextToken();
			if((tmpString2.indexOf(",")) != -1)
			{
			
				//Now, we want to grab the substring that occurs AFTER the SECOND '"'.
				//At present, pprof does not seem to allow an '"' in the mapping name.  So
				//, I can be assured that I will not find more than two before the "excl" or "incl".
				StringTokenizer checkQuotesTokenizer = new StringTokenizer(inString,"\"");
				
				//Need to get the third token.  Could do it in a loop, just as quick this way.
				String tmpString = checkQuotesTokenizer.nextToken();
				tmpString = checkQuotesTokenizer.nextToken();
				tmpString = checkQuotesTokenizer.nextToken();
				
				//Ok, now, the string in tmpString should include at least "excl" or "incl", and
				//also, the first token should be either "excl" or "incl".
				StringTokenizer checkForInclusiveTokenizer = new StringTokenizer(tmpString, " \t\n\r");
				tmpString = checkForInclusiveTokenizer.nextToken();
					
				//At last, do the check.	
				if(tmpString.equals("incl"))
				{
					return true;
				}
			}
			//If here, it means that we are not looking at the correct string or that we did not
			//find a match.  Therefore, return false.
			return false;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD06");
		}
		
		return false;
	}
	
	boolean checkForInclusiveWithTOrM(String inString)
	{
		
		try{
			//In this function I need to be careful.  If the mapping name contains "incl", I
			//might interpret this line as being the inclusive line when in fact it is not.

			//Ok, so at least we have the correct string.
			//Now, we want to grab the substring that occurs AFTER the SECOND '"'.
			//At present, pprof does not seem to allow an '"' in the mapping name.  So
			//, I can be assured that I will not find more than two before the "excl" or "incl".
			StringTokenizer checkQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Need to get the third token.  Could do it in a loop, just as quick this way.
			String tmpString = checkQuotesTokenizer.nextToken();
			tmpString = checkQuotesTokenizer.nextToken();
			tmpString = checkQuotesTokenizer.nextToken();
			
			//Ok, now, the string in tmpString should include at least "excl" or "incl", and
			//also, the first token should be either "excl" or "incl".
			StringTokenizer checkForInclusiveTokenizer = new StringTokenizer(tmpString, " \t\n\r");
			tmpString = checkForInclusiveTokenizer.nextToken();
				
			//At last, do the check.	
			if(tmpString.equals("incl"))
			{
				return true;
			}
			
			//If here, it means that we are not looking at the correct string or that we did not
			//find a match.  Therefore, return false.
			return false;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD07");
		}
		
		return false;
	}
	
	String getMappingName(String inString)
	{
		try{
			String tmpString;
			
			StringTokenizer getMappingNameTokenizer = new StringTokenizer(inString, "\"");
			
			//Since we know that the mapping name is the only one in the quotes, just ignore the
			//first token, and then grab the next.
			
			//Grab the first token.
			tmpString = getMappingNameTokenizer.nextToken();
			
			//Grab the second token.
			tmpString = getMappingNameTokenizer.nextToken();
			
			//Now return the second string.
			return tmpString;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD08");
		}
		
		return null;
	}
	
	int getMappingID(String inString)
	{
		try{
			String tmpString;
			
			StringTokenizer getMappingIDTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			//The mapping id will be the second token on its line.
			
			//Grab the first token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the second token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			
			//Now return the id.
			//Integer tmpInteger = new Integer(tmpString);
			//int tmpInt = tmpInteger.intValue();
			return Integer.parseInt(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD09");
		}
		
		return -1;
	}
	
	double getNumberOfCalls(String inString)
	{
		try{
			String tmpString;
			
			StringTokenizer getMappingIDTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			//The number of calls will be the fourth token on its line.
			
			//Grab the first token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the second token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the third token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the forth token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Now return the number of calls.
			return Double.parseDouble(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD10");
		}
		
		return -1;
	}
	
	double getNumberOfSubRoutines(String inString)
	{
		try{
			String tmpString;
			
			StringTokenizer getMappingIDTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			//The number of calls will be the fifth token on its line.
			
			//Grab the first token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the second token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the third token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the forth token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the fifth token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Now return the number of subroutines.
			return Double.parseDouble(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD11");
		}
		
		return -1;
	}
	
	double getUserSecPerCall(String inString)
	{
		try{
			String tmpString;
			
			StringTokenizer getMappingIDTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			//The number of calls will be the sixth token on its line.
			
			//Grab the first token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the second token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the third token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the forth token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the fifth token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Grab the sixth token.
			tmpString = getMappingIDTokenizer.nextToken();
			
			//Now return the number of subroutines.
			return Double.parseDouble(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD11");
		}
		
		return -1;
	}
	
	String getGroupNames(String inString)
	{
		
		try{	
				String tmpString = null;
				
				StringTokenizer getMappingNameTokenizer = new StringTokenizer(inString, "\"");
				
				//Grab the first token.
				tmpString = getMappingNameTokenizer.nextToken();
				//Grab the second token.
				tmpString = getMappingNameTokenizer.nextToken();
				//Grab the third token.
				tmpString = getMappingNameTokenizer.nextToken();
				
				//Just do the group check once.
				if(!groupNamesCheck)
				{
					//If present, "GROUP=" will be in this token.
					int tmpInt = tmpString.indexOf("GROUP=");
					if(tmpInt > 0)
					{
						groupNamesPresent = true;
					}
					
					groupNamesCheck = true;
					
				}
				
				if(groupNamesPresent)
				{
					//We can grab the group name.
					
					//Grab the forth token.
					tmpString = getMappingNameTokenizer.nextToken();
					return tmpString;
				}
				
				//If here, this profile file does not track the group names.
				return null;

			}
			catch(Exception e)
			{
				jRacy.systemError(null, "SSD12");
			}
		
		return null;
	}
	
	double getValue(String inString)
	{
		try{
			String tmpString;
			
			//First strip away the portion of the string not needed.
			StringTokenizer valueQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
			tmpString = valueQuotesTokenizer.nextToken();
			tmpString = valueQuotesTokenizer.nextToken();
			tmpString = valueQuotesTokenizer.nextToken();
			
			//Ok, now concentrate on the third token.  The token in question should be the second.
			StringTokenizer valueTokenizer = new StringTokenizer(tmpString, " \t\n\r");
			tmpString = valueTokenizer.nextToken();
			tmpString = valueTokenizer.nextToken();
			
			return (double)Double.parseDouble(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD13");
		}
		
		return -1;
	}
	
	double getPercentValue(String inString)
	{
		try{
			String tmpString;
			
			//First strip away the portion of the string not needed.
			StringTokenizer percentValueQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
			tmpString = percentValueQuotesTokenizer.nextToken();
			tmpString = percentValueQuotesTokenizer.nextToken();
			tmpString = percentValueQuotesTokenizer.nextToken();
			
			//Ok, now concentrate on the third token.  The token in question should be the third.
			StringTokenizer percentValueTokenizer = new StringTokenizer(tmpString, " \t\n\r");
			tmpString = percentValueTokenizer.nextToken();
			tmpString = percentValueTokenizer.nextToken();
			tmpString = percentValueTokenizer.nextToken();
			
			//Now return the value obtained.
			return Double.parseDouble(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD14");
		}
		
		return -1;
	}
	
	boolean checkForUserEvents(String inString)
	{
		try{
			String tmpString;
			
			StringTokenizer checkForUserEventsTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			//Looking for the second token ... no danger of conflict here.
			
			//Grab the first token.
			tmpString = checkForUserEventsTokenizer.nextToken();
			
			//Grab the second token.
			tmpString = checkForUserEventsTokenizer.nextToken();
			
			//No do the check.
			if(tmpString.equals("userevents"))
				return true;
			else
				return false;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD15");
		}
		
		return false;	
	}
	
	int getNumberOfUserEvents(String inString)
	{
		try{
			StringTokenizer getNumberOfUserEventsTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			String tmpString;
			
			//It will be the first token.
			tmpString = getNumberOfUserEventsTokenizer.nextToken();
			
			//Now return the number of user events number.
			return Integer.parseInt(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD16");
		}
		
		return -1;
	}
										
	String getUserEventName(String inString)
	{
		try{
			String tmpString;
			
			StringTokenizer getUserEventNameTokenizer = new StringTokenizer(inString, "\"");
			
			//Since we know that the user event name is the only one in the quotes, just ignore the
			//first token, and then grab the next.
			
			//Grab the first token.
			tmpString = getUserEventNameTokenizer.nextToken();
			
			//Grab the second token.
			tmpString = getUserEventNameTokenizer.nextToken();
			
			//Now return the second string.
			return tmpString;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD17");
		}
		
		return null;
	}
	
	int getUserEventID(String inString)
	{
		try{
			String tmpString;
			
			StringTokenizer getUserEventIDTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			//The mapping id will be the third token on its line.
			
			//Grab the first token.
			tmpString = getUserEventIDTokenizer.nextToken();
			
			//Grab the second token.
			tmpString = getUserEventIDTokenizer.nextToken();
			
			//Grab the second token.
			tmpString = getUserEventIDTokenizer.nextToken();
			
			return Integer.parseInt(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD18");
		}
		
		return -1;
	}
	
	int getUENValue(String inString)
	{
		
		try{
			String tmpString;
			
			//First strip away the portion of the string not needed.
			StringTokenizer uEQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
			tmpString = uEQuotesTokenizer.nextToken();
			tmpString = uEQuotesTokenizer.nextToken();
			tmpString = uEQuotesTokenizer.nextToken();
			
			//Ok, now concentrate on the third token.  The token in question should be the first.
			StringTokenizer uETokenizer = new StringTokenizer(tmpString, " \t\n\r");
			tmpString = uETokenizer.nextToken();
			
			//Now return the value obtained.
			return (int) Double.parseDouble(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD19");
		}
		
		return -1;
	}
	
	double getUEMinValue(String inString)
	{
		try{
			String tmpString;
			
			//First strip away the portion of the string not needed.
			StringTokenizer uEQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
			tmpString = uEQuotesTokenizer.nextToken();
			tmpString = uEQuotesTokenizer.nextToken();
			tmpString = uEQuotesTokenizer.nextToken();
			
			//Ok, now concentrate on the third token.  The token in question should be the third.
			StringTokenizer uETokenizer = new StringTokenizer(tmpString, " \t\n\r");
			tmpString = uETokenizer.nextToken();
			tmpString = uETokenizer.nextToken();
			tmpString = uETokenizer.nextToken();
			
			//Now return the value obtained.
			return Double.parseDouble(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD20");
		}
		
		return -1;
	}
	
	double getUEMaxValue(String inString)
	{
		try{
			String tmpString;
			
			//First strip away the portion of the string not needed.
			StringTokenizer uEQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
			tmpString = uEQuotesTokenizer.nextToken();
			tmpString = uEQuotesTokenizer.nextToken();
			tmpString = uEQuotesTokenizer.nextToken();
			
			//Ok, now concentrate on the third token.  The token in question should be the second.
			StringTokenizer uETokenizer = new StringTokenizer(tmpString, " \t\n\r");
			tmpString = uETokenizer.nextToken();
			tmpString = uETokenizer.nextToken();
			
			//Now return the value obtained.
			return Double.parseDouble(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD21");
		}
		
		return -1;
	}
	
	double getUEMeanValue(String inString)
	{
		try{
			String tmpString;
			
			//First strip away the portion of the string not needed.
			StringTokenizer uEQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
			tmpString = uEQuotesTokenizer.nextToken();
			tmpString = uEQuotesTokenizer.nextToken();
			tmpString = uEQuotesTokenizer.nextToken();
			
			//Ok, now concentrate on the third token.  The token in question should be the forth.
			StringTokenizer uETokenizer = new StringTokenizer(tmpString, " \t\n\r");
			tmpString = uETokenizer.nextToken();
			tmpString = uETokenizer.nextToken();
			tmpString = uETokenizer.nextToken();
			tmpString = uETokenizer.nextToken();
			
			//Now return the value obtained.
			return Double.parseDouble(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD22");
		}
		
		return -1;
	}
	
	int getNode(String inString, boolean UEvent)
	{
		try{
			StringTokenizer getNodeTokenizer = new StringTokenizer(inString, ", \t\n\r");
			
			String tmpString;
			
			if(UEvent)
			{
				//Need to strip off the first token.
				tmpString = getNodeTokenizer.nextToken();
			}
			
			//Get the first token.
			tmpString = getNodeTokenizer.nextToken();
			
			//Now return the node number.
			Integer tmpInteger = new Integer(tmpString);
			int tmpInt = tmpInteger.intValue();
			return tmpInt;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD23");
		}
		
		return -1;
		
	}
	
	int getContext(String inString, boolean UEvent)
	{
		try{
			StringTokenizer getContextTokenizer = new StringTokenizer(inString, ", \t\n\r");
			
			String tmpString;
			
			if(UEvent)
			{
				//Need to strip off the first token.
				tmpString = getContextTokenizer.nextToken();
			}
			
			//Get the first token.
			tmpString = getContextTokenizer.nextToken();
			
			//Get the second.
			tmpString = getContextTokenizer.nextToken();
			
			//Now return the context number.
			Integer tmpInteger = new Integer(tmpString);
			int tmpInt = tmpInteger.intValue();
			return tmpInt;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD24");
		}
		
		return -1;
	}
	
	int getThread(String inString, boolean UEvent)
	{		
		try{
			StringTokenizer getThreadTokenizer = new StringTokenizer(inString, ", \t\n\r");
			
			String tmpString;
			
			if(UEvent)
			{
				//Need to strip off the first token.
				tmpString = getThreadTokenizer.nextToken();
			}
			
			//Get the first token.
			tmpString = getThreadTokenizer.nextToken();
			
			//Get the second.
			tmpString = getThreadTokenizer.nextToken();
			
			//Get the third token.
			tmpString = getThreadTokenizer.nextToken();
			
			//Now return the context number.
			Integer tmpInteger = new Integer(tmpString);
			int tmpInt = tmpInteger.intValue();
			return tmpInt;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD25");
		}
		
		return -1;
	}
	
	String getCounterName(String inString)
	{
		try{
			String tmpString = null;
			int tmpInt = inString.indexOf("_MULTI_");
			
			if(tmpInt > 0)
			{
				//We are reading data from a multiple counter run.
				//Grab the counter name.
				tmpString = inString.substring(tmpInt+7);
				return tmpString;
			}
			
			//We are not reading data from a multiple counter run.
			return tmpString;	
			
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SSD26");
		}
		
		return null;
	}
	
	//******************************
	//End - Helper functions for buildStatic data.
	//******************************
	
	
	
	//******************************
	//Useful functions to help the drawing windows.
	//
	//For the most part, these functions just return data
	//items that are easier to calculate whilst building the global
	//lists
	//******************************
	
	public int getPositionOfName()
	{
		return positionOfName;
	}
	
	public int getPositionOfUserEventName()
	{
		return positionOfUserEventName;
	}
	
	public String getHeading()
	{
		return heading;
	}
	
	public String getUserEventHeading()
	{
		return userEventHeading;
	}
	
	public void setMaxMeanInclusiveValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		maxMeanInclusiveValueList.add(dataValueLocation, tmpDouble);}
	
	public double getMaxMeanInclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) maxMeanInclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxMeanExclusiveValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		maxMeanExclusiveValueList.add(dataValueLocation, tmpDouble);}
	
	public double getMaxMeanExclusiveValue(int dataValueLocation){
		Double tmpDouble = (Double) maxMeanExclusiveValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxMeanInclusivePercentValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		maxMeanInclusivePercentValueList.add(dataValueLocation, tmpDouble);}
	
	public double getMaxMeanInclusivePercentValue(int dataValueLocation){
		Double tmpDouble = (Double) maxMeanInclusivePercentValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxMeanExclusivePercentValue(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		maxMeanExclusivePercentValueList.add(dataValueLocation, tmpDouble);}
	
	public double getMaxMeanExclusivePercentValue(int dataValueLocation){
		Double tmpDouble = (Double) maxMeanExclusivePercentValueList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public void setMaxMeanNumberOfCalls(double inDouble){
		maxMeanNumberOfCalls = inDouble;}
	
	public double getMaxMeanNumberOfCalls(){
		return maxMeanNumberOfCalls;}
	
	public void setMaxMeanNumberOfSubRoutines(double inDouble){
		maxMeanNumberOfSubRoutines = inDouble;}
	
	public double getMaxMeanNumberOfSubRoutines(){
		return maxMeanNumberOfSubRoutines;}
	
	public void setMaxMeanUserSecPerCall(int dataValueLocation, double inDouble){
		Double tmpDouble = new Double(inDouble);
		maxMeanUserSecPerCallList.add(dataValueLocation, tmpDouble);}
	
	public double getMaxMeanUserSecPerCall(int dataValueLocation){
		Double tmpDouble = (Double) maxMeanUserSecPerCallList.elementAt(dataValueLocation);
		return tmpDouble.doubleValue();}
	
	public boolean groupNamesPresent(){
		return groupNamesPresent;
	}
	
	private void setUserEventsPresent(boolean inBoolean){
		userEventsPresent = inBoolean;
	}
	
	public boolean userEventsPresent(){
		return userEventsPresent;
	}
	//******************************
	//End - Useful functions to help the drawing windows.
	//******************************
	
	
	//******************************
	//Instance data.
	//******************************
	private SystemEvents systemEvents = null;
	private StaticMainWindow sMW = null;
	private ColorChooser clrChooser = new ColorChooser(this, null);
	private Preferences  preferences = new  Preferences(this, null);		
	
	private String profilePathName = null;
	private String profilePathNameReverse = null;
	private String runName = null;
	private Vector runValueNameList = new Vector(); 
	private int currentRunValueLocation = 0;
	private int currentRunValueWriteLocation = 0;		
	
	private GlobalMapping globalMapping;
	private Vector StaticServerList;
	private int positionOfName;
	private int positionOfUserEventName;
	private String counterName;
	private String heading;
	private String userEventHeading;
	private boolean isUserEventHeadingSet;
	boolean groupNamesCheck = false;
	boolean groupNamesPresent = false;
	boolean userEventsPresent = false;
	int bSDCounter;
	
	boolean test = true;
	
	private int numberOfMappings = -1;
	private int numberOfUserEvents = -1; 
	
	//Max mean values.
	private Vector maxMeanInclusiveValueList = new Vector();
	private Vector maxMeanExclusiveValueList = new Vector();
	private Vector maxMeanInclusivePercentValueList = new Vector();
	private Vector maxMeanExclusivePercentValueList = new Vector();
	private double maxMeanNumberOfCalls = 0;
	private double maxMeanNumberOfSubRoutines = 0;
	private Vector maxMeanUserSecPerCallList = new Vector();
	
	
	//******************************
	//End - Instance data.
	//******************************

}