/* 
   TauPprofOutputReader.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.io.*;
import java.util.*;

public class TauPprofOutputReader{
    public static void readData(Trial trial, File file){

	try{
	    System.out.println("Processing data file, please wait ......");
	    long time = System.currentTimeMillis();

	    //######
	    //Frequently used items.
	    //######
	    boolean firstRead = trial.firstRead();
	    int currentValueLocation = trial.getCurValLoc();
	    GlobalMapping globalMapping = trial.getGlobalMapping();
	    GlobalMappingElement globalMappingElement;

	    int nodeID = -1;
	    int contextID = -1;
	    int threadID = -1;
	    int lastNodeID = -1;
	    int lastContextID = -1;
	    int lastThreadID = -1;

	    Node currentNode = null;
	    Context currentContext = null;
	    Thread currentThread = null;

	    String inputString = null;
	    String s1 = null;
	    String s2 = null;

	    String tokenString;
	    String mappingNameString = null;
	    String groupNamesString = null;
	    StringTokenizer genericTokenizer;

	    int mappingID = -1;
	    double value = -1;
	    double percentValue = -1;

	    //A loop counter.
	    int bSDCounter = 0;
	    //######
	    //End - Frequently used items.
	    //######


	    FileInputStream fileIn = new FileInputStream(inFile);
	    //ProgressMonitorInputStream progressIn = new ProgressMonitorInputStream(null, "Processing ...", fileIn);
	    //InputStreamReader inReader = new InputStreamReader(progressIn);
	    InputStreamReader inReader = new InputStreamReader(fileIn);
	    BufferedReader br = new BufferedReader(inReader);
            

	    //Process the file.
      
	    //####################################
	    //First Line
	    //####################################
	    //This line is not required. Check to make sure that it is there however.
	    inputString = br.readLine();
	    if(inputString == null)
		return;
	    bSDCounter++;
	    //####################################
	    //End - First Line
	    //####################################
      
	    //####################################
	    //Second  Line
	    //####################################
	    //This is an important line.
	    inputString = br.readLine();
	    genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
	    //It's first token will be the number of mappings present.  Get it.
	    tokenString = genericTokenizer.nextToken();
      
	    if(firstRead){
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
	    if(firstRead)
		initializeGlobalMapping(Integer.parseInt(tokenString), 0);
      
	    //Set the counter name.
	    counterName = getCounterName(inputString);
      
	    //Ok, we are adding a counter name.  Since nothing much has happened yet, it is a
	    //good place to initialize a few things.
      
	    //Need to call addDefaultToVectors() on all objects that require it.
	    trial.addDefaultToVectors();

	    //Only need to call addDefaultToVectors() if not the first run.
	    if(!firstRead){
		if(ParaProf.debugIsOn)
		    System.out.println("Increasing the storage for the new counter.");
		
		for(Enumeration e1 = (globalMapping.getMapping(0)).elements(); e1.hasMoreElements() ;){
		    GlobalMappingElement tmpGME = (GlobalMappingElement) e1.nextElement();
		    tmpGME.incrementStorage();
		}
	  
		for(Enumeration e2 = (globalMapping.getMapping(2)).elements(); e2.hasMoreElements() ;){
		    GlobalMappingElement tmpGME = (GlobalMappingElement) e2.nextElement();
		    tmpGME.incrementStorage();
		}
	  
		for(Enumeration e3 = trial.getNodes().elements(); e3.hasMoreElements() ;){
		    Node node = (Node) e3.nextElement();
		    for(Enumeration e4 = node.getContexts().elements(); e4.hasMoreElements() ;){
			Context context = (Context) e4.nextElement();
			for(Enumeration e5 = context.getThreads().elements(); e5.hasMoreElements() ;){
			    Thread thread = (Thread) e5.nextElement();
			    thread.incrementStorage();
			    for(Enumeration e6 = thread.getFunctionList().elements(); e6.hasMoreElements() ;){
				GlobalThreadDataElement ref = (GlobalThreadDataElement) e6.nextElement();
				//Only want to add an element if this mapping existed on this thread.
				//Check for this.
				if(ref != null)
				    ref.incrementStorage();
			    }
			}
		    }
		}
	  
		if(ParaProf.debugIsOn)
		    System.out.println("Done increasing the storage for the new counter.");
	    }
      
	    //Now set the counter name.
      
	    if(counterName == null)
		counterName = new String("Wallclock Time");

	    System.out.println("Counter name is: " + counterName);
      
	    Value newValue = addValue();
	    newValue.setValueName(counterName);
	    setCurValLoc(newValue.getValueID());
	    System.out.println("The number of mappings in the system is: " + tokenString);
      
	    bSDCounter++;
	    //####################################
	    //End - Second  Line
	    //####################################
      
	    //####################################
	    //Third Line
	    //####################################
	    //Do not need the third line.
	    inputString = br.readLine();
	    if(inputString == null)
		return;
	    bSDCounter++;
	    //####################################
	    //End - Third Line
	    //####################################
      
	    while((inputString = br.readLine()) != null){
		genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
          
		//Check to See if the String begins with a t.
		if((inputString.charAt(0)) == 't'){
		    mappingID = getMappingID(inputString);
		    value = getValue(inputString);
		    if(checkForExcInc(inputString, true, false)){
			mappingNameString = getMappingName(inputString);
			mappingID = getMappingID(inputString);
			
			if(firstRead){ 
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
					if((tmpInt != -1) && (ParaProf.debugIsOn))
					    System.out.println("Adding " + tmpString + " group with id: " + tmpInt + " to mapping: " + mappingNameString);
				    }    
				}    
			    }
			}
                  
			if(firstRead){
			    //Now that we have the mapping name and id, fill in the global mapping element
			    //for this mapping.  I am assuming here that pprof's output lists only the
			    //global ids.
			    if(!(globalMapping.setMappingNameAt(mappingNameString, mappingID, 0)))
				System.out.println("There was an error adding mapping to the global mapping");
			}
			globalMapping.setTotalExclusiveValueAt(currentValueLocation, value, mappingID, 0);
		    }
		    else{
			globalMapping.setTotalInclusiveValueAt(currentValueLocation, value, mappingID, 0);
		    }
		} //End - Check to See if the String begins with a t.
		//Check to See if the String begins with a mt.
		else if((inputString.charAt(0)) == 'm'){
		    mappingID = getMappingID(inputString);
		    value = getValue(inputString);
		    percentValue = getPercentValue(inputString);
		    //Grab the correct global mapping element.
		    globalMappingElement = globalMapping.getGlobalMappingElement(mappingID, 0);
                
		    if(checkForExcInc(inputString, true, false)){
			//Now set the values correctly.
			if((getMaxMeanExclusiveValue(currentValueLocation)) < value){
			    setMaxMeanExclusiveValue(currentValueLocation, value);}
			if((getMaxMeanExclusivePercentValue(currentValueLocation)) < percentValue){
			    setMaxMeanExclusivePercentValue(currentValueLocation, percentValue);}
			
			globalMappingElement.setMeanExclusiveValue(currentValueLocation, value);
			globalMappingElement.setMeanExclusivePercentValue(currentValueLocation, percentValue);
		    }
		    else{
			//Now set the values correctly.
			if((getMaxMeanInclusiveValue(currentValueLocation)) < value){
			    setMaxMeanInclusiveValue(currentValueLocation, value);}
			if((getMaxMeanInclusivePercentValue(currentValueLocation)) < percentValue){
			    setMaxMeanInclusivePercentValue(currentValueLocation, percentValue);}
                  
			globalMappingElement.setMeanInclusiveValue(currentValueLocation, value);
			globalMappingElement.setMeanInclusivePercentValue(currentValueLocation, percentValue);
                  
			//Set number of calls/subroutines/usersec per call.
			inputString = br.readLine();
			TauPprofOutputReader.setNumberOfCSUMean(trial, currentValueLocation, inputString, globalMappingElement);
			globalMappingElement.setMeanValuesSet(true);
		    }
		}//End - Check to See if the String begins with a m.
		//String does not begin with either an m or a t, the rest of the checks go here.
		else{
		    if(checkForExcInc(inputString, true, true)){ 
			
			//Stuff common to a non-first run and a first run.
			//Grab the mapping ID.
			mappingID = getMappingID(inputString);
			//Grab the value.
			value = getValue(inputString);
			percentValue = getPercentValue(inputString);
			
			//Update the max values if required.
			//Grab the correct global mapping element.
			globalMappingElement = globalMapping.getGlobalMappingElement(mappingID, 0);
			if((globalMappingElement.getMaxExclusiveValue(currentValueLocation)) < value)
			    globalMappingElement.setMaxExclusiveValue(currentValueLocation, value);
			if((globalMappingElement.getMaxExclusivePercentValue(currentValueLocation)) < percentValue)
			    globalMappingElement.setMaxExclusivePercentValue(currentValueLocation, percentValue);
			//Get the node,context,thread.
			nodeID = getNCT(0,inputString, false);
			contextID = getNCT(1,inputString, false);
			threadID = getNCT(2,inputString, false);
			
			if(firstRead){
			    //Now the complicated part.  Setting up the node,context,thread data.
			    //These first two if statements force a change if the current node or
			    //current context changes from the last, but without a corresponding change
			    //in the thread number.  For example, if we have the sequence:
			    //0,0,0 - 1,0,0 - 2,0,0 or 0,0,0 - 0,1,0 - 1,0,0.
			    if(lastNodeID != nodeID){
				lastContextID = -1;
				lastThreadID = -1;
			    }
			    if(lastContextID != contextID){
				lastThreadID = -1;
			    }
			    if(lastThreadID != threadID){
				if(threadID == 0){
				    //Create a new thread ... and set it to be the current thread.
				    currentThread = new Thread(nodeID, contextID, threadID);
				    //Add the correct number of global thread data elements.
				    currentThread.initializeFunctionList(numberOfMappings);
				    //Update the thread number.
				    lastThreadID = threadID;
				    
				    //Set the appropriate global thread data element.
				    Vector tmpVector = currentThread.getFunctionList();
				    GlobalThreadDataElement tmpGTDE = null;
				    
				    tmpGTDE = (GlobalThreadDataElement) tmpVector.elementAt(mappingID);
				    
				    if(tmpGTDE == null){
					tmpGTDE = new GlobalThreadDataElement(trial, false);
					tmpGTDE.setMappingID(mappingID);
					currentThread.addFunction(tmpGTDE, mappingID);
				    }
				    tmpGTDE.setMappingExists();
				    tmpGTDE.setExclusiveValue(currentValueLocation, value);
				    tmpGTDE.setExclusivePercentValue(currentValueLocation, percentValue);
				    //Now check the max values on this thread.
				    if((currentThread.getMaxExclusiveValue(currentValueLocation)) < value)
					currentThread.setMaxExclusiveValue(currentValueLocation, value);
				    if((currentThread.getMaxExclusivePercentValue(currentValueLocation)) < value)
					currentThread.setMaxExclusivePercentValue(currentValueLocation, percentValue);
				    
				    //Check to see if the context is zero.
				    if(contextID == 0){
					//Create a new context ... and set it to be the current context.
					currentContext = new Context(nodeID, contextID);
					//Add the current thread
					currentContext.addThread(currentThread);
					
					//Create a new server ... and set it to be the current server.
					currentNode = new Node(nodeID);
					//Add the current context.
					currentNode.addContext(currentContext);
					//Add the current server.
					nodes.addElement(currentNode);
					
					//Update last context and last node.
					lastContextID = contextID;
					lastNodeID = nodeID;
				    }
				    else{
					//Context number is not zero.  Create a new context ... and set it to be current.
					currentContext = new Context(nodeID, contextID);
					//Add the current thread
					currentContext.addThread(currentThread);
					
					//Add the current context.
					currentNode.addContext(currentContext);
					
					//Update last context and last node.
					lastContextID = contextID;
				    }
				}
				else{
				    //Thread number is not zero.  Create a new thread ... and set it to be the current thread.
				    currentThread = new Thread(nodeID, contextID, threadID);
				    //Add the correct number of global thread data elements.
				    currentThread.initializeFunctionList(numberOfMappings);
				    //Update the thread number.
				    lastThreadID = threadID;
				    
				    //Not thread changes.  Just set the appropriate global thread data element.
				    Vector tmpVector = currentThread.getFunctionList();
				    GlobalThreadDataElement tmpGTDE = null;
				    tmpGTDE = (GlobalThreadDataElement) tmpVector.elementAt(mappingID);
				    				    
				    if(tmpGTDE == null){
					tmpGTDE = new GlobalThreadDataElement(trial, false);
					tmpGTDE.setMappingID(mappingID);
					currentThread.addFunction(tmpGTDE, mappingID);
				    }
				    
				    tmpGTDE.setMappingExists();
				    tmpGTDE.setExclusiveValue(currentValueLocation, value);
				    tmpGTDE.setExclusivePercentValue(currentValueLocation, percentValue);
				    //Now check the max values on this thread.
				    if((currentThread.getMaxExclusiveValue(currentValueLocation)) < value)
					currentThread.setMaxExclusiveValue(currentValueLocation, value);
				    if((currentThread.getMaxExclusivePercentValue(currentValueLocation)) < value)
					currentThread.setMaxExclusivePercentValue(currentValueLocation, percentValue);
				    
				    //Add the current thread
				    currentContext.addThread(currentThread);
				}
			    }
			    else{
				//Not thread changes.  Just set the appropriate global thread data element.
				Vector tmpVector = currentThread.getFunctionList();
				GlobalThreadDataElement tmpGTDE = null;
				tmpGTDE = (GlobalThreadDataElement) tmpVector.elementAt(mappingID);
				
				
				if(tmpGTDE == null){
				    tmpGTDE = new GlobalThreadDataElement(trial, false);
				    tmpGTDE.setMappingID(mappingID);
				    currentThread.addFunction(tmpGTDE, mappingID);
				}
				
				tmpGTDE.setMappingExists();
				tmpGTDE.setExclusiveValue(currentValueLocation, value);
				tmpGTDE.setExclusivePercentValue(currentValueLocation, percentValue);
				//Now check the max values on this thread.
				if((currentThread.getMaxExclusiveValue(currentValueLocation)) < value)
				    currentThread.setMaxExclusiveValue(currentValueLocation, value);
				if((currentThread.getMaxExclusivePercentValue(currentValueLocation)) < percentValue)
				    currentThread.setMaxExclusivePercentValue(currentValueLocation, percentValue);
			    }
			}
			else{
			    Thread thread = trial.getThread(nodeID,contextID,threadID);
			    GlobalThreadDataElement tmpGTDE = thread.getFunction(mappingID);
			    tmpGTDE.setExclusiveValue(currentValueLocation, value);
			    tmpGTDE.setExclusivePercentValue(currentValueLocation, percentValue);
			    if((thread.getMaxExclusiveValue(currentValueLocation)) < value)
				thread.setMaxExclusiveValue(currentValueLocation, value);
			    if((thread.getMaxExclusivePercentValue(currentValueLocation)) < percentValue)
				thread.setMaxExclusivePercentValue(currentValueLocation, percentValue);
			}
		    }
		    else if(checkForExcInc(inputString, false, true)){
			//Grab the mapping ID.
			mappingID = getMappingID(inputString);
			//Grab the value.
			value = getValue(inputString);
			percentValue = getPercentValue(inputString);
			
			//Update the max values if required.
			//Grab the correct global mapping element.
			globalMappingElement = globalMapping.getGlobalMappingElement(mappingID, 0);
			
			if((globalMappingElement.getMaxInclusiveValue(currentValueLocation)) < value)
			    globalMappingElement.setMaxInclusiveValue(currentValueLocation, value);
			
			if((globalMappingElement.getMaxInclusivePercentValue(currentValueLocation)) < percentValue)
			    globalMappingElement.setMaxInclusivePercentValue(currentValueLocation, percentValue);
			
			//Print out the node,context,thread.
			nodeID = getNCT(0,inputString, false);
			contextID = getNCT(1,inputString, false);
			threadID = getNCT(2,inputString, false);
			Thread thread = trial.getThread(nodeID,contextID,threadID);
			GlobalThreadDataElement tmpGTDE = thread.getFunction(mappingID);
			
			tmpGTDE.setInclusiveValue(currentValueLocation, value);
			tmpGTDE.setInclusivePercentValue(currentValueLocation, percentValue);
			if((thread.getMaxInclusiveValue(currentValueLocation)) < value)
			    thread.setMaxInclusiveValue(currentValueLocation, value);
			if((thread.getMaxInclusivePercentValue(currentValueLocation)) < percentValue)
			    thread.setMaxInclusivePercentValue(currentValueLocation, percentValue);
			
			//Get the number of calls and number of sub routines
			inputString = br.readLine();
			TauPprofOutputReader.setNumberOfCSU(trial, inputString, globalMappingElement, thread, tmpGTDE);
		    }
		    else if(noue(inputString)){
			//Just ignore the string if this is not the first check.
			//Assuming is that user events do not change for each counter value.
			if(firstRead){
			    //The first time a user event string is encountered, get the number of user events and 
			    //initialize the global mapping for mapping position 2.
			    if(!(userEventsPresent())){
				//Get the number of user events.
				numberOfUserEvents = getNumberOfUserEvents(inputString);
				initializeGlobalMapping(numberOfUserEvents, 2);
				if(ParaProf.debugIsOn){
				    System.out.println("The number of user events defined is: " + numberOfUserEvents);
				    System.out.println("Initializing mapping selection 2 (The loaction of the user event mapping) for " +
						       numberOfUserEvents + " mappings.");
				}
			    } 
			    
			    //The first line will be the user event heading ... skip it.
			    br.readLine();
			    //Now that we know how many user events to expect, we can grab that number of lines.
			    for(int j=0; j<numberOfUserEvents; j++){
				s1 = br.readLine();
				s2 = br.readLine();
				UserEventData ued = getData(s1,s2, userEventsPresent);

				//Initialize the user list for this thread.
				if(j == 0){
				    //Note that this works correctly because we process the user events in a different manner.
				    //ALL the user events for each THREAD NODE are processed in the above for-loop.  Therefore,
				    //the below for-loop is only run once on each THREAD NODE.

				    Thread thread = trial.getThread(nodeID,contextID,threadID);
				    if(firstRead){
					thread.initializeUsereventList(numberOfUserEvents);
				    }
				}
				//Only need to set the name in the global mapping once.
				if(!(userEventsPresent())){
				    if(!(globalMapping.setMappingNameAt(ued.name, ued.id, 2)))
					System.out.println("There was an error adding mapping to the global mapping");
				}
				if(ued.noc != 0){
				    //Update the max values if required.
				    //Grab the correct global mapping element.
				    globalMappingElement = globalMapping.getGlobalMappingElement(ued.id, 2);
				    if((globalMappingElement.getMaxUserEventNumberValue()) < ued.noc)
					globalMappingElement.setMaxUserEventNumberValue(ued.noc);
				    if((globalMappingElement.getMaxUserEventMinValue()) < ued.min)
					globalMappingElement.setMaxUserEventMinValue(ued.min);
				    if((globalMappingElement.getMaxUserEventMaxValue()) < ued.max)
					globalMappingElement.setMaxUserEventMaxValue(ued.max);
				    if((globalMappingElement.getMaxUserEventMeanValue()) < ued.mean)
					globalMappingElement.setMaxUserEventMeanValue(ued.mean);
				    
				    GlobalThreadDataElement tmpGTDEUE = new GlobalThreadDataElement(trial, true);
				    tmpGTDEUE.setUserEventID(ued.id);
				    tmpGTDEUE.setUserEventNumberValue(ued.noc);
				    tmpGTDEUE.setUserEventMinValue(ued.min);
				    tmpGTDEUE.setUserEventMaxValue(ued.max);
				    tmpGTDEUE.setUserEventMeanValue(ued.mean);
				    (trial.getThread(nodeID,contextID,threadID)).addUserevent(tmpGTDEUE, ued.id);
				}
			    }
			    //Now set the userEvents flag.
			    setUserEventsPresent(true);
			}
		    }
		}
		//Increment the loop counter.
		bSDCounter++;
	    }
	    
	    //Close the file.
	    br.close();

	    if(ParaProf.debugIsOn){
		System.out.println("The total number of threads is: " + trial.getTotalNumberOfThreads());
		System.out.println("The number of mappings is: " + trial.getNumberOfMappings());
		System.out.println("The number of user events is: " + trial.getNumberOfUserEvents());
	    }

	    System.out.println("Processing callpath data ...");
	    if(CallPathUtilFuncs.isAvailable(getGlobalMapping().getMappingIterator(0))){
		setCallPathDataPresent(true);
		CallPathUtilFuncs.buildRelations(getGlobalMapping());
	    }
	    else{
		System.out.println("No callpath data found.");
	    }
	    System.out.println("Done - Processing callpath data!");
      
	    time = (System.currentTimeMillis()) - time;
	    System.out.println("Done processing data file, please wait ......");
	    System.out.println("Time to process file (in milliseconds): " + time);
	}
        catch(Exception e){
	    ParaProf.systemError(e, null, "SSD01");
	}
    }
    
    //####################################
    //Helper functions.
    //####################################

    private static boolean noue(String s){
	int stringPosition = 0;
	char tmpChar = s.charAt(stringPosition);
	while(tmpChar!='\u0020'){
	    stringPosition++;
	    tmpChar = s.charAt(stringPosition);
	}
	stringPosition++;
	tmpChar = s.charAt(stringPosition);
	if(tmpChar=='u')
	    return true;
	else
	    return false;
    }

    private static UserEventData getData(String s1, String s2, boolean doneNames){
	UserEventData ued = new UserEventData();
	try{
	    char[] tmpArray = s1.toCharArray();
	    char[] result = new char[tmpArray.length];

	    char lastCharCheck = ',';
	    int stringPosition = 10;
	    
	    int start = 0;
	    int end = 9;
	    int resultPosition = 0;

	    for(int i=start;i<9;i++){
		if(i==2)
		    lastCharCheck = '\u0020';
		else if(i==4){
		    //Want to skip processing the name if we
		    //do not need it.
		    if(doneNames){
			stringPosition++;
			lastCharCheck = '"';
			while(tmpArray[stringPosition]!=lastCharCheck){
			    stringPosition++;
			}
			stringPosition++;

			//Set things to look as if we are in i=5 iteration.
			i=5;
			lastCharCheck = '\u0020';
			stringPosition++;
		    }
		    else{
			lastCharCheck = '"';
			stringPosition++;
		    }
		}
		else if(i==5){
		    lastCharCheck = '\u0020';
		    stringPosition++;
		}
		while(tmpArray[stringPosition]!=lastCharCheck){
		    result[resultPosition]=tmpArray[stringPosition];
		    stringPosition++;
		    resultPosition++;
		}
		
		switch(i){
		case 0:
		    ued.node = Integer.parseInt(new String(result,0,resultPosition));
		    break;
		case 1:
		    ued.context = Integer.parseInt(new String(result,0,resultPosition));
		    break;
		case 2:
		    ued.threadID = Integer.parseInt(new String(result,0,resultPosition));
		    break;
		case 3:
		    ued.id = Integer.parseInt(new String(result,0,resultPosition));
		    break;
		case 4:
		    if(!userEventsPresent)
			ued.name = new String(result,0,resultPosition);  
		    break;
		case 5:
		    ued.noc = (int) Double.parseDouble(new String(result,0,resultPosition));
		    break;
		case 6:
		    ued.max = Double.parseDouble(new String(result,0,resultPosition));
		    break;
		case 7:
		    ued.min = Double.parseDouble(new String(result,0,resultPosition));
		    break;
		case 8:
		    ued.mean = Double.parseDouble(new String(result,0,resultPosition));
		    break;
		default:
		    throw new UnexpectedStateException(String.valueOf(i));
		}
		resultPosition=0;
		stringPosition++;
	    }
	    //One more item to pick up if userevent string.
	    int length = tmpArray.length;
	    while(stringPosition < length){
		result[resultPosition]=tmpArray[stringPosition];
		stringPosition++;
		resultPosition++;
	    }
	    ued.std = Double.parseDouble(new String(result,0,resultPosition));
	}
	catch(Exception e){
	    System.out.println("An error occured!");
	    e.printStackTrace();
	}
	return ued;
    }

    private static String getMappingName(String inString){
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
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD08");
	}
	
	return null;
    }
    
    private static int getMappingID(String inString)
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
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD09");
	}
	return -1;
    }
    
    private static boolean checkForExcInc(String inString, boolean exclusive, boolean checkString){
	boolean result = false;
    
	try{
	    //In this function I need to be careful.  If the mapping name contains "excl", I
	    //might interpret this line as being the exclusive line when in fact it is not.
      
	    if(checkString){
		StringTokenizer checkTokenizer = new StringTokenizer(inString," ");
		String tmpString2 = checkTokenizer.nextToken();
		if((tmpString2.indexOf(",")) == -1)
		    return result;
	    }
      
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
	    if(exclusive){
		if(tmpString.equals("excl"))
		    result = true;
	    }
	    else{
		if(tmpString.equals("incl"))
		    result = true;
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD04");}
	return result;
    }
  
    private static void setNumberOfCSU(Trial trial, String inString, GlobalMappingElement inGME,
				Thread thread, GlobalThreadDataElement inGTDE){ 
	//Set the number of calls/subroutines/usersec per call.
	//The number of calls will be the fourth token on its line.
	//The number of subroutines will be the fifth token on its line.
	//The usersec per call will be the sixth token on its line.
	try{
	    int currentValueLocation = trial.getCurValLoc();

	    String tmpString = null;
	    double tmpDouble = -1;
	    int tmpInt = -1;  //Parse as a double, but cast to this int just in case pprof.dat records as a double.
	    StringTokenizer getMappingIDTokenizer = new StringTokenizer(inString, " \t\n\r");
	    tmpString = getMappingIDTokenizer.nextToken();
	    tmpString = getMappingIDTokenizer.nextToken();
	    tmpString = getMappingIDTokenizer.nextToken();
	    //Set number of calls.
	    tmpString = getMappingIDTokenizer.nextToken();
	    tmpDouble = Double.parseDouble(tmpString);
	    tmpInt = (int) tmpDouble;
	    if((inGME.getMaxNumberOfCalls()) < tmpInt)
		inGME.setMaxNumberOfCalls(tmpInt);
	    if((thread.getMaxNumberOfCalls()) < tmpInt)
		thread.setMaxNumberOfCalls(tmpInt);
	    inGTDE.setNumberOfCalls(tmpInt);
	    //Set number of subroutines.
	    tmpString = getMappingIDTokenizer.nextToken();
	    tmpDouble = Double.parseDouble(tmpString);
	    tmpInt = (int) tmpDouble;
	    if((inGME.getMaxNumberOfSubRoutines()) < tmpInt)
		inGME.setMaxNumberOfSubRoutines(tmpInt);
	    if((thread.getMaxNumberOfSubRoutines()) < tmpInt)
		thread.setMaxNumberOfSubRoutines(tmpInt);
	    inGTDE.setNumberOfSubRoutines(tmpInt);
	    //Set usersec per call.
	    tmpString = getMappingIDTokenizer.nextToken();
	    tmpDouble = Double.parseDouble(tmpString);
	    if((inGME.getMaxUserSecPerCall(currentValueLocation)) < tmpDouble)
		inGME.setMaxUserSecPerCall(currentValueLocation, tmpDouble);
	    if((thread.getMaxUserSecPerCall(currentValueLocation)) < tmpDouble)
		thread.setMaxUserSecPerCall(currentValueLocation, tmpDouble);
	    inGTDE.setUserSecPerCall(currentValueLocation, tmpDouble);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD10");
	}
    }
    
    private static void setNumberOfCSUMean(Trial trial, int currentValueLocation, String inString, GlobalMappingElement inGME){
	//Set the number of calls/subroutines/usersec per call for mean.
	//The number of calls will be the fourth token on its line.
	//The number of subroutines will be the fifth token on its line.
	//The usersec per call will be the sixth token on its line.
	try{
	    String tmpString = null;
	    double tmpDouble = -1;
	    StringTokenizer getMappingIDTokenizer = new StringTokenizer(inString, " \t\n\r");
	    tmpString = getMappingIDTokenizer.nextToken();
	    tmpString = getMappingIDTokenizer.nextToken();
	    tmpString = getMappingIDTokenizer.nextToken();
	    //Set number of calls.
	    tmpString = getMappingIDTokenizer.nextToken();
	    tmpDouble = Double.parseDouble(tmpString);
	    if((trial.getMaxMeanNumberOfCalls()) < tmpDouble)
		trial.setMaxMeanNumberOfCalls(tmpDouble);
	    inGME.setMeanNumberOfCalls(tmpDouble);
	    //Set number of subroutines.
	    tmpString = getMappingIDTokenizer.nextToken();
	    tmpDouble = Double.parseDouble(tmpString);
	    if((trial.getMaxMeanNumberOfSubRoutines()) < tmpDouble)
		trial.setMaxMeanNumberOfSubRoutines(tmpDouble);
	    inGME.setMeanNumberOfSubRoutines(tmpDouble);
	    //Set usersec per call.
	    tmpString = getMappingIDTokenizer.nextToken();
	    tmpDouble = Double.parseDouble(tmpString);
	    if((trial.getMaxMeanUserSecPerCall(currentValueLocation)) < tmpDouble)
		trial.setMaxMeanUserSecPerCall(currentValueLocation, tmpDouble);
	    inGME.setMeanUserSecPerCall(currentValueLocation, tmpDouble);
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "SSD10");
	    }
    }
  
    private static String getGroupNames(String inString){
    
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
		ParaProf.systemError(e, null, "SSD12");
	    }
    
	return null;
    }
  
    private static double getValue(String inString){
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
		ParaProf.systemError(e, null, "SSD13");
	    }
    
	return -1;
    }
  
    private static double getPercentValue(String inString){
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
		ParaProf.systemError(e, null, "SSD14");
	    }
    
	return -1;
    }
  
    private static int getNumberOfUserEvents(String inString){
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
		ParaProf.systemError(e, null, "SSD16");
	    }
    
	return -1;
    }
  
    private static int getNCT(int selector, String inString, boolean UEvent){
	//I am assuming an quick implimentation of charAt and append for this function.
	int nCT = -1;
	char lastCharCheck = '\u0020';
    
	try{
	    char tmpChar = '\u0020';
	    StringBuffer tmpBuffer = new StringBuffer();
	    int stringPosition = 0;
	    if(UEvent)
		stringPosition = 10;
      
	    if(selector != 2)
		lastCharCheck = ',';
        
	    for(int i=0;i<selector;i++){
		//Skip over ','.
		while(tmpChar!=','){
		    tmpChar = inString.charAt(stringPosition);
		    stringPosition++;
		}
		//Reset tmpChar.
		tmpChar = '\u0020';
		//Skip over the second ','.
	    }
          
	    tmpChar = inString.charAt(stringPosition);
	    while(tmpChar!=lastCharCheck){
		tmpBuffer.append(tmpChar);
		stringPosition++;
		tmpChar = inString.charAt(stringPosition);
	    }
        
	    //System.out.println("nCT string is: " + tmpBuffer.toString());
	    //System.out.println("String length is: " + tmpBuffer.toString().length());
	    nCT = Integer.parseInt(tmpBuffer.toString());
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD23");
	}
    
	return nCT;
    }
  
    private static String getCounterName(String inString){
	try{
	    String tmpString = null;
	    int tmpInt = inString.indexOf("_MULTI_");
      
	    if(tmpInt > 0){
		//We are reading data from a multiple counter run.
		//Grab the counter name.
		tmpString = inString.substring(tmpInt+7);
		return tmpString;
	    }
      	    //We are not reading data from a multiple counter run.
	    return tmpString; 
      	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD26");
	}
    
	return null;
    }

    //####################################
    //End - Helper functions.
    //####################################
}
