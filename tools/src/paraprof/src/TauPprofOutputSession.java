/* 
   TauPprofOutputSession.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;



import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

import java.io.*;
import java.util.*;
import dms.dss.*;

public class TauPprofOutputSession extends DataSession{


    /**
     * Terminate the DataSession object.
     *
     */
    public void terminate(){}   // formerly "close"

    /**
     * Returns a ListIterator of Application objects.
     *
     * @return	DataSessionIterator object of all Applications.
     * @see	DataSessionIterator
     * @see	Application
     */
    public ListIterator getApplicationList(){
	return null;}

    /**
     * Returns a ListIterator of Experiment objects
     *
     * @return	DataSessionIterator object of all Experiments.  If there is an Application saved in the DataSession, then only the Experiments for that Application are returned.
     * @see	DataSessionIterator
     * @see	Experiment
     * @see	DataSession#setApplication
     */
    public ListIterator getExperimentList(){
	return null;}

    /**
     * Returns a ListIterator of Trial objects
     *
     * @return	DataSessionIterator object of all Trials.  If there is an Application and/or Experiment saved in the DataSession, then only the Trials for that Application and/or Experiment are returned.
     * @see	DataSessionIterator
     * @see	Trial
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     */
    public ListIterator getTrialList(){
	return null;}


    /**
     * Set the Application for this DataSession.  The DataSession object will maintain a reference to the Application referenced by the id.  To clear this reference, call setApplication(Application) with a null reference.
     *
     * @param	id unique id of the Application object to be saved.
     * @see	Application
     * @see	DataSession#setApplication(Application)
     */
    public Application setApplication(int id){
	return null;}
    
    /**
     * Set the Application for this DataSession.  The DataSession object will maintain a reference to the Application referenced by the name and/or version.  To clear this reference, call setApplication(Application) with a null reference.
     *
     * @param	name name of the Application object to be saved.
     * @param	version version of the Application object to be saved.
     * @see	Application
     * @see	DataSession#setApplication(Application)
     */
    public Application setApplication(String name, String version){
	return null;}

    /**
     * Set the Experiment for this DataSession.  The DataSession object will maintain a reference to the Experiment referenced by the id.  To clear this reference, call setExperiment(Experiment) with a null reference.
     *
     * @param	id unique id of the Experiment object to be saved.
     * @see	Experiment
     * @see	DataSession#setExperiment(Experiment)
     */
    public Experiment setExperiment(int id){
	return null;}
    
    /**
     * Set the Trial for this DataSession.  The DataSession object will maintain a reference to the Trial referenced by the id.  To clear this reference, call setTrial(Trial) with a null reference.
     *
     * @param	id unique id of the Trial object to be saved.
     * @see	Trial
     * @see	DataSession#setTrial(Trial)
     */
    public Trial setTrial(int id){
	return null;}

    /**
     * Returns a ListIterator of Function objects.
     *
     * @return	DataSessionIterator object of all Functions.  If there is an Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) saved in the DataSession, then only the Functions for that Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) are returned.
     * @see	DataSessionIterator
     * @see	Function
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     */
    public ListIterator getFunctions(){
	return null;}

    /**
     * Set the Function for this DataSession.  The DataSession object will maintain a reference to the Function referenced by the id.  To clear this reference, call setFunction(Function) with a null reference.
     *
     * @param	id unique id of the Function object to be saved.
     * @see	Function
     * @see	DataSession#setFunction(Function)
     */
    public Function setFunction(int id){
	return null;}

    /**
     * Returns the Function identified by the unique function id.
     *
     * @param	functionID unique id of the Function object.
     * @return	Function object identified by the unique function id.
     * @see	Function
     */
    public Function getFunction(int functionID){
	return null;}
	
    /**
     * Returns a ListIterator of UserEvent objects.
     *
     * @return	DataSessionIterator object of all UserEvents.  If there is an Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) saved in the DataSession, then only the UserEvents for that Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) are returned.
     * @see	DataSessionIterator
     * @see	UserEvent
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     */
    public ListIterator getUserEvents(){
	return null;}

    /**
     * Set the UserEvent for this DataSession.  The DataSession object will maintain a reference to the UserEvent referenced by the id.  To clear this reference, call setUserEvent(UserEvent) with a null reference.
     *
     * @param	id unique id of the UserEvent object to be saved.
     * @see	UserEvent
     * @see	DataSession#setUserEvent(UserEvent)
     */
    public UserEvent setUserEvent(int id){
	return null;}

    /**
     * Returns the UserEvent identified by the unique user event id.
     *
     * @param	userEventID unique id of the UserEvent object to be saved.
     * @return	UserEvent object identified by the unique user event id.
     * @see	UserEvent
     */
    public UserEvent getUserEvent(int userEventID){
	return null;}
	
    /**
     * Returns the FunctionData for this DataSession
     *
     * @return	DataSessionIterator of FunctionData objects.  If there is an Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or Function(s) saved in the DataSession, then only the Functions for that Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or Function(s) are returned.
     * @see	DataSessionIterator
     * @see	FunctionDataObject
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     * @see	DataSession#setFunction
     */
    public ListIterator getFunctionData(){
	return null;}

    /**
     * Returns the UserEventData for this DataSession
     *
     * @return	DataSessionIterator of UserEventData objects.  If there is an Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or Function(s) saved in the DataSession, then only the UserEvents for that Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or Function(s) are returned.
     * @see	DataSessionIterator
     * @see	UserEventDataObject
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     * @see	DataSession#setUserEvent
     */
    public ListIterator getUserEventData(){
	return null;}

    /**
     * Initialize the DataSession object.
     *
     * @param	obj	an implementation-specific object required to initialize the DataSession
     */
    public void initialize(Object obj){
	File file = null;

	try{
	    if(obj instanceof File)
		file = (File) obj;

	    System.out.println("Processing data file, please wait ......");
	    long time = System.currentTimeMillis();

	    //######
	    //Frequently used items.
	    //######
	    int metric = 0;
	    GlobalMappingElement globalMappingElement = null;
	    GlobalThreadDataElement globalThreadDataElement = null;

	    Node node = null;
	    Context context = null;
	    Thread thread = null;
	    int nodeID = -1;
	    int contextID = -1;
	    int threadID = -1;

	    String inputString = null;
	    String s1 = null;
	    String s2 = null;

	    String tokenString;
	    String groupNamesString = null;
	    StringTokenizer genericTokenizer;

	    FunctionDataLine1 line1 = null;
	    int mappingID = -1;

	    //A loop counter.
	    int bSDCounter = 0;

	    int numberOfUserEvents = 0;
	    //######
	    //End - Frequently used items.
	    //######


	    FileInputStream fileIn = new FileInputStream(file);
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
		this.setNumberOfMappings(Integer.parseInt(tokenString));
	    }
	    else{
		if((this.getNumberOfMappings()) != Integer.parseInt(tokenString)){
		    System.out.println("***********************");
		    System.out.println("The number of mappings does not match!!!");
		    System.out.println("");
		    System.out.println("To add to an existing run, you must be choosing from");
		    System.out.println("a list of multiple metrics from that same run!!!");
		    System.out.println("***********************");
	      
		    return;
		}
	    }

	    //Set the counter name.
	    String counterName = getCounterName(inputString);
      
	    //Ok, we are adding a counter name.  Since nothing much has happened yet, it is a
	    //good place to initialize a few things.
      
	    //Need to call increaseVectorStorage() on all objects that require it.
	    this.increaseVectorStorage();

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
	  
		for(Enumeration e3 = nct.getNodes().elements(); e3.hasMoreElements() ;){
		    node = (Node) e3.nextElement();
		    for(Enumeration e4 = node.getContexts().elements(); e4.hasMoreElements() ;){
			context = (Context) e4.nextElement();
			for(Enumeration e5 = context.getThreads().elements(); e5.hasMoreElements() ;){
			    thread = (Thread) e5.nextElement();
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
      
	    Metric metricRef = this.addMetric();
	    metricRef.setName(counterName);
	    metric = metricRef.getID();
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

		int lineType = -1;
		/*
		  (0) t-exclusive
		  (1) t-inclusive
		  (2) m-exclusive
		  (3) m-inclusive
		  (4) exclusive
		  (5) inclusive
		  (6) userevent
		*/
	    
		//Determine the lineType.
		if((inputString.charAt(0)) == 't'){
		    if(checkForExcInc(inputString, true, false))
			lineType = 0;
		    else
			lineType = 1;
		}
		else if((inputString.charAt(0)) == 'm'){
		    if(checkForExcInc(inputString, true, false))
			lineType = 2;
		    else
		       lineType = 3;
		}
		else if(checkForExcInc(inputString, true, true))
		    lineType = 4;
		else if(checkForExcInc(inputString, false, true)) 
		    lineType = 5;
		else if(noue(inputString))
		    lineType = 6;
	    
		//Common things to grab
		if((lineType!=6) && (lineType!=-1)){
		    line1 = this.getFunctionDataLine1(inputString);
		    mappingID = globalMapping.addGlobalMapping(line1.name, 0);
		    globalMappingElement = globalMapping.getGlobalMappingElement(mappingID, 0);
		}

		switch(lineType){
		case 0:
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
					System.out.println("Adding " + tmpString + " group with id: " + tmpInt + " to mapping: " + line1.name);
				}    
			    }    
			}
		    }
		    globalMapping.setTotalExclusiveValueAt(metric, line1.value, mappingID, 0);
		    break;
		case 1:
		    globalMapping.setTotalInclusiveValueAt(metric, line1.value, mappingID, 0);
		    break;
		case 2:
		    //Now set the values correctly.
		    if((this.getMaxMeanExclusiveValue(metric)) < line1.value){
			this.setMaxMeanExclusiveValue(metric, line1.value);}
		    if((this.getMaxMeanExclusivePercentValue(metric)) < line1.percentValue){
			this.setMaxMeanExclusivePercentValue(metric, line1.percentValue);}
		    
		    globalMappingElement.setMeanExclusiveValue(metric, line1.value);
		    globalMappingElement.setMeanExclusivePercentValue(metric, line1.percentValue);
		    break;
		case 3:
		    //Now set the values correctly.
		    if((this.getMaxMeanInclusiveValue(metric)) < line1.value){
			this.setMaxMeanInclusiveValue(metric, line1.value);}
		    if((this.getMaxMeanInclusivePercentValue(metric)) < line1.percentValue){
			this.setMaxMeanInclusivePercentValue(metric, line1.percentValue);}
		    
		    globalMappingElement.setMeanInclusiveValue(metric, line1.value);
		    globalMappingElement.setMeanInclusivePercentValue(metric, line1.percentValue);
		    
		    //Set number of calls/subroutines/usersec per call.
		    inputString = br.readLine();

		    FunctionDataLine2 meanLine2 = this.getFunctionDataLine2(inputString);

		    //Set the values.
		    globalMappingElement.setMeanNumberOfCalls(meanLine2.noc);
		    globalMappingElement.setMeanNumberOfSubRoutines(meanLine2.nsbr);
		    globalMappingElement.setMeanUserSecPerCall(metric, meanLine2.useccall);

		    //Set the max values.
		    if((this.getMaxMeanNumberOfCalls()) < meanLine2.noc)
			this.setMaxMeanNumberOfCalls(meanLine2.noc);

		    if((this.getMaxMeanNumberOfSubRoutines()) < meanLine2.nsbr)
			this.setMaxMeanNumberOfSubRoutines(meanLine2.nsbr);

		    if((this.getMaxMeanUserSecPerCall(metric)) < meanLine2.useccall)
			this.setMaxMeanUserSecPerCall(metric, meanLine2.useccall);

		    globalMappingElement.setMeanValuesSet(true);
		    break;
		case 4:
		    if((globalMappingElement.getMaxExclusiveValue(metric)) < line1.value)
			globalMappingElement.setMaxExclusiveValue(metric, line1.value);
		    if((globalMappingElement.getMaxExclusivePercentValue(metric)) < line1.percentValue)
			globalMappingElement.setMaxExclusivePercentValue(metric, line1.percentValue);

		    //Get the node,context,thread.
		    int[] array = this.getNCT(inputString);
		    nodeID = array[0];
		    contextID = array[1];
		    threadID = array[2];

		    node = nct.getNode(nodeID);
		    if(node==null)
			node = nct.addNode(nodeID);
		    context = node.getContext(contextID);
		    if(context==null)
			context = node.addContext(contextID);
		    thread = context.getThread(threadID);
		    if(thread==null){
			thread = context.addThread(threadID);
			thread.initializeFunctionList(this.getNumberOfMappings());
		    }

		    Vector vector = thread.getFunctionList();
		    globalThreadDataElement = null;
				
		    globalThreadDataElement = (GlobalThreadDataElement) vector.elementAt(mappingID);
				
		    if(globalThreadDataElement == null){
			globalThreadDataElement = new GlobalThreadDataElement(globalMapping.getGlobalMappingElement(mappingID, 0), false);
			thread.addFunction(globalThreadDataElement, mappingID);
		    }
		    globalThreadDataElement.setMappingExists();
		    globalThreadDataElement.setExclusiveValue(metric, line1.value);
		    globalThreadDataElement.setExclusivePercentValue(metric, line1.percentValue);
		    //Now check the max values on this thread.
		    if((thread.getMaxExclusiveValue(metric)) < line1.value)
			thread.setMaxExclusiveValue(metric, line1.value);
		    if((thread.getMaxExclusivePercentValue(metric)) < line1.percentValue)
			thread.setMaxExclusivePercentValue(metric, line1.percentValue);
		    break;
		case 5:
		    if((globalMappingElement.getMaxInclusiveValue(metric)) < line1.value)
			globalMappingElement.setMaxInclusiveValue(metric, line1.value);
		    
		    if((globalMappingElement.getMaxInclusivePercentValue(metric)) < line1.percentValue)
			globalMappingElement.setMaxInclusivePercentValue(metric, line1.percentValue);
		    
		    thread = nct.getThread(nodeID,contextID,threadID);
		    globalThreadDataElement = thread.getFunction(mappingID);
		    
		    globalThreadDataElement.setInclusiveValue(metric, line1.value);
		    globalThreadDataElement.setInclusivePercentValue(metric, line1.percentValue);
		    if((thread.getMaxInclusiveValue(metric)) < line1.value)
			thread.setMaxInclusiveValue(metric, line1.value);
		    if((thread.getMaxInclusivePercentValue(metric)) < line1.percentValue)
			thread.setMaxInclusivePercentValue(metric, line1.percentValue);
		    
		    //Get the number of calls and number of sub routines
		    inputString = br.readLine();
		    FunctionDataLine2 line2 = this.getFunctionDataLine2(inputString);

		    //Set the values.
		    globalThreadDataElement.setNumberOfCalls(line2.noc);
		    globalThreadDataElement.setNumberOfSubRoutines(line2.nsbr);
		    globalThreadDataElement.setUserSecPerCall(metric, line2.useccall);

		    //Set the max values.
		    if(globalMappingElement.getMaxNumberOfCalls() < line2.noc)
			globalMappingElement.setMaxNumberOfCalls(line2.noc);
		    if(thread.getMaxNumberOfCalls() < line2.noc)
			thread.setMaxNumberOfCalls(line2.noc);

		    if(globalMappingElement.getMaxNumberOfSubRoutines() < line2.nsbr)
			globalMappingElement.setMaxNumberOfSubRoutines(line2.nsbr);
		    if(thread.getMaxNumberOfSubRoutines() < line2.nsbr)
			thread.setMaxNumberOfSubRoutines(line2.nsbr);
		    
		    if(globalMappingElement.getMaxUserSecPerCall(metric) < line2.useccall)
			globalMappingElement.setMaxUserSecPerCall(metric, line2.useccall);
		    if(thread.getMaxUserSecPerCall(metric) < line2.useccall)
			thread.setMaxUserSecPerCall(metric, line2.useccall);
		    break;
		case 6:
		    //Just ignore the string if this is not the first check.
		    //Assuming is that user events do not change for each counter value.
		    if(firstRead){
			if(!(userEventsPresent())){
			    //Get the number of user events.
			    numberOfUserEvents = getNumberOfUserEvents(inputString);
			    this.setNumberOfUserEvents(numberOfUserEvents);
			} 
			
			//The first line will be the user event heading ... skip it.
			br.readLine();
			//Now that we know how many user events to expect, we can grab that number of lines.
			for(int j=0; j<numberOfUserEvents; j++){
			    s1 = br.readLine();
			    s2 = br.readLine();
			    UserEventData ued = getUserEventData(s1);
			    System.out.println("noc:"+ued.noc+"min:"+ued.min+"max:"+ued.max+"mean:"+ued.mean);
			    
			    //Initialize the user list for this thread.
			    if(j == 0){
				//Note that this works correctly because we process the user events in a different manner.
				//ALL the user events for each THREAD NODE are processed in the above for-loop.  Therefore,
				//the below for-loop is only run once on each THREAD NODE.
				
				if(firstRead){
				    (nct.getThread(nodeID,contextID,threadID)).initializeUsereventList(numberOfUserEvents);
				}
			    }

			    int userEventID = globalMapping.addGlobalMapping(ued.name, 2);
			    if(ued.noc != 0){
				//Update the max values if required.
				//Grab the correct global mapping element.
				globalMappingElement = globalMapping.getGlobalMappingElement(userEventID, 2);
				if((globalMappingElement.getMaxUserEventNumberValue()) < ued.noc)
				    globalMappingElement.setMaxUserEventNumberValue(ued.noc);
				if((globalMappingElement.getMaxUserEventMinValue()) < ued.min)
				    globalMappingElement.setMaxUserEventMinValue(ued.min);
				if((globalMappingElement.getMaxUserEventMaxValue()) < ued.max)
				    globalMappingElement.setMaxUserEventMaxValue(ued.max);
				if((globalMappingElement.getMaxUserEventMeanValue()) < ued.mean)
				    globalMappingElement.setMaxUserEventMeanValue(ued.mean);
				
				GlobalThreadDataElement tmpGTDEUE = new GlobalThreadDataElement(globalMapping.getGlobalMappingElement(userEventID, 2), true);
				tmpGTDEUE.setUserEventNumberValue(ued.noc);
				tmpGTDEUE.setUserEventMinValue(ued.min);
				tmpGTDEUE.setUserEventMaxValue(ued.max);
				tmpGTDEUE.setUserEventMeanValue(ued.mean);
				(nct.getThread(nodeID,contextID,threadID)).addUserevent(tmpGTDEUE, userEventID);
			    }
			}
			//Now set the userEvents flag.
			setUserEventsPresent(true);
		    }
		    break;
		default:
		    if(ParaProf.debugIsOn){
			System.out.println("Skipping line: " + bSDCounter);
		    }
		    break;
		}
		   
		//Increment the loop counter.
		bSDCounter++;
	    }
	    
	    //Close the file.
	    br.close();
	    
	    if(ParaProf.debugIsOn){
		System.out.println("The total number of threads is: " + nct.getTotalNumberOfThreads());
		System.out.println("The number of mappings is: " + this.getNumberOfMappings());
		System.out.println("The number of user events is: " + this.getNumberOfUserEvents());
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
    
    public NCT getNCT(){
	return nct;}

    public Vector getMetrics(){
	return metrics;}

    public GlobalMapping getGlobalMapping(){
	return globalMapping;}

    public Vector getMaxMeanExclusiveList(){
	return maxMeanExclusiveValueList;}

    public Vector getMaxMeanInclusiveList(){
	return maxMeanInclusiveValueList;}

    public Vector getMaxMeanInclusivePercentList(){
	return maxMeanInclusivePercentValueList;}

    public Vector getMaxMeanExclusivePercentList(){
	return maxMeanExclusivePercentValueList;}
  
    public Vector getMaxMeanUserSecPerCallList(){
	return maxMeanUserSecPerCallList;}

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

    public int getNumberOfMappings(){
	return numberOfMappings;}

    public int getNumberOfUserEvents(){
	return numberOfUserEvents;}
    
    public boolean groupNamesPresent(){
	return groupNamesPresent;}

    public boolean userEventsPresent(){
	return userEventsPresent;}
  
    public boolean callPathDataPresent(){
	return callPathDataPresent;}
    
    //####################################
    //Private Section.
    //####################################

    //######
    //Pprof.dat string processing methods.
    //######
    private boolean noue(String s){
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

    private UserEventData getUserEventData(String string){
	try{
	    UserEventData ued = new UserEventData();
	    StringTokenizer st1 = new StringTokenizer(string, "\"");
	    st1.nextToken();
	    ued.name = st1.nextToken();
	    
	    String stttt = st1.nextToken();
	    StringTokenizer st2 = new StringTokenizer(stttt, " \t\n\r");

	    System.out.println("new string:"+stttt+":");

	    ued.noc = (int) Double.parseDouble(st2.nextToken());
	    ued.max = Double.parseDouble(st2.nextToken());
	    ued.min = Double.parseDouble(st2.nextToken());
	    ued.mean = Double.parseDouble(st2.nextToken());
	    ued.std = Double.parseDouble(st2.nextToken());
	    return ued;
	}
	catch(Exception e){
	    System.out.println("An error occured!");
	    e.printStackTrace();
	}
	return null;
    }

    private boolean checkForExcInc(String inString, boolean exclusive, boolean checkString){
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

    private FunctionDataLine1 getFunctionDataLine1(String string){
	try{
	    FunctionDataLine1 line1 = new FunctionDataLine1();
	    StringTokenizer st1 = new StringTokenizer(string, "\"");
	    st1.nextToken();
	    line1.name = st1.nextToken();
	    
	    StringTokenizer st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");
	    st2.nextToken();
	    line1.value = Double.parseDouble(st2.nextToken());
	    line1.percentValue = Double.parseDouble(st2.nextToken());
	    
	    return line1;
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD08");
	}
	return null;
    }

    private FunctionDataLine2 getFunctionDataLine2(String string){ 
	//The number of calls will be the fourth token on its line.
	//The number of subroutines will be the fifth token on its line.
	//The usersec per call will be the sixth token on its line.
	try{
	    FunctionDataLine2 line2 = new FunctionDataLine2();
	    StringTokenizer getMappingIDTokenizer = new StringTokenizer(string, " \t\n\r");
	    getMappingIDTokenizer.nextToken();
	    getMappingIDTokenizer.nextToken();
	    getMappingIDTokenizer.nextToken();
	    
	    line2.noc = (int) Double.parseDouble(getMappingIDTokenizer.nextToken());
	    line2.nsbr = (int) Double.parseDouble(getMappingIDTokenizer.nextToken());
	    line2.useccall = Double.parseDouble(getMappingIDTokenizer.nextToken());
	    return line2;
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD10");
	}
	return null;
    }

    private String getGroupNames(String string){
	try{  
	    StringTokenizer getMappingNameTokenizer = new StringTokenizer(string, "\"");
 	    getMappingNameTokenizer.nextToken();
	    getMappingNameTokenizer.nextToken();
	    String str = getMappingNameTokenizer.nextToken();
        
	    //Just do the group check once.
	    if(!groupNamesCheck){
		//If present, "GROUP=" will be in this token.
		int tmpInt = str.indexOf("GROUP=");
		if(tmpInt > 0){
		    groupNamesPresent = true;
		}
		groupNamesCheck = true;
	    }
	    
	    if(groupNamesPresent){
		 str = getMappingNameTokenizer.nextToken();
		    return str;
	    }
	    //If here, this profile file does not track the group names.
	    return null;
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD12");
	}
	return null;
    }
  
    private int getNumberOfUserEvents(String string){
	try{
	    StringTokenizer st = new StringTokenizer(string, " \t\n\r");
	    return Integer.parseInt(st.nextToken());
  	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD16");
	}
	return -1;
    }

    private int[] getNCT(String string){
	int[] nct = new int[3];
	StringTokenizer st = new StringTokenizer(string, " ,\t\n\r");
	nct[0] = Integer.parseInt(st.nextToken());
	nct[1] = Integer.parseInt(st.nextToken());
	nct[2] = Integer.parseInt(st.nextToken());
	return nct;
    }
  
    private String getCounterName(String inString){
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
    //######
    //End - Pprof.dat string processing methods.
    //######

    private void increaseVectorStorage(){
	maxMeanInclusiveValueList.add(new Double(0));
	maxMeanExclusiveValueList.add(new Double(0));
	maxMeanInclusivePercentValueList.add(new Double(0));
	maxMeanExclusivePercentValueList.add(new Double(0));
	maxMeanUserSecPerCallList.add(new Double(0));
    }

    private Metric addMetric(){
	Metric newMetric = new Metric();
	newMetric.setID((metrics.size()));
	metrics.add(newMetric);
	return newMetric;
    }

    private void setMaxMeanInclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanInclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    private void setMaxMeanExclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanExclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    private void setMaxMeanInclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanInclusivePercentValueList.add(dataValueLocation, tmpDouble);}
  
    private void setMaxMeanExclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanExclusivePercentValueList.add(dataValueLocation, tmpDouble);}

    private void setMaxMeanNumberOfCalls(double inDouble){
	maxMeanNumberOfCalls = inDouble;}
  
    private void setMaxMeanNumberOfSubRoutines(double inDouble){
	maxMeanNumberOfSubRoutines = inDouble;}

    private void setMaxMeanUserSecPerCall(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanUserSecPerCallList.add(dataValueLocation, tmpDouble);}
  
    private void setNumberOfMappings(int numberOfMappings){
	this.numberOfMappings = numberOfMappings;}

    private void setNumberOfUserEvents(int numberOfUserEvents){
	this.numberOfUserEvents = numberOfUserEvents;}

    private void setGroupNamesPresent(boolean groupNamesPresent){
	this.groupNamesPresent = groupNamesPresent;}
  
    private void setUserEventsPresent(boolean userEventsPresent){
	this.userEventsPresent = userEventsPresent;}

    private void setCallPathDataPresent(boolean callPathDataPresent){
	this.callPathDataPresent = callPathDataPresent;}
    //####################################
    //End - Private Section.
    //####################################

    //####################################
    //Instance data.
    //####################################
    boolean firstRead = true;
    boolean groupNamesCheck = false;
    
    private int numberOfMappings = 0;
    private int numberOfUserEvents = 0;
    private int totalNumberOfContexts = -1;
    private int totalNumberOfThreads = -1;
    private boolean groupNamesPresent = false;
    private boolean userEventsPresent = false;
    private boolean callPathDataPresent = false;

    private GlobalMapping globalMapping = new GlobalMapping();
    private NCT nct = new NCT();
    private Vector metrics = new Vector();

    private Vector maxMeanInclusiveValueList = new Vector();
    private Vector maxMeanExclusiveValueList = new Vector();
    private Vector maxMeanInclusivePercentValueList = new Vector();
    private Vector maxMeanExclusivePercentValueList = new Vector();
    private double maxMeanNumberOfCalls = 0;
    private double maxMeanNumberOfSubRoutines = 0;
    private Vector maxMeanUserSecPerCallList = new Vector();
    
    //####################################
    //End - Instance data.
    //####################################
}

class FunctionDataLine1{
    public double value = -1.0;
    public double percentValue = -1.0;
    public String name = null;
}

class FunctionDataLine2{
    public int noc = -1;
    public int nsbr = -1;
    public double useccall = -1.0;
}

class UserEventData{
    public String name = null;
    public int noc = -1;
    public double max = -1.0;
    public double min = -1.0;
    public double mean = -1.0;
    public double std = -1.0;
}

class UnexpectedStateException extends Exception{
    public UnexpectedStateException(){}
    public UnexpectedStateException(String err){
	super("UnexpectedStateException - message: " + err);
    }
}
