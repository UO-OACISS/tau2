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
	try{
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
	    
	    int mappingID = -1;
	    
	    //A loop counter.
	    int bSDCounter = 0;
	    
	    int numberOfUserEvents = 0;

	    Vector v = null;
	    File[] files = null;
	    //######
	    //End - Frequently used items.
	    //######
	    v = (Vector) obj;
	    for(Enumeration e = v.elements(); e.hasMoreElements() ;){
		files = (File[]) e.nextElement();
		System.out.println("Processing data file, please wait ......");
		long time = System.currentTimeMillis();

		FileInputStream fileIn = new FileInputStream(files[0]);
		InputStreamReader inReader = new InputStreamReader(fileIn);
		BufferedReader br = new BufferedReader(inReader);
      
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
		    counterName = new String("Time");

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
			this.getFunctionDataLine1(inputString);
			mappingID = globalMapping.addGlobalMapping(functionDataLine1.s0, 0);
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
					    System.out.println("Adding " + tmpString + " group with id: " + tmpInt + " to mapping: " + functionDataLine1.s0);
				    }    
				}    
			    }
			}
			globalMapping.setTotalExclusiveValueAt(metric, functionDataLine1.d0, mappingID, 0);
			break;
		    case 1:
			globalMapping.setTotalInclusiveValueAt(metric, functionDataLine1.d0, mappingID, 0);
			break;
		    case 2:
			//Now set the values correctly.
			if((this.getMaxMeanExclusiveValue(metric)) < functionDataLine1.d0){
			    this.setMaxMeanExclusiveValue(metric, functionDataLine1.d0);}
			if((this.getMaxMeanExclusivePercentValue(metric)) < functionDataLine1.d1){
			    this.setMaxMeanExclusivePercentValue(metric, functionDataLine1.d1);}
		    
			globalMappingElement.setMeanExclusiveValue(metric, functionDataLine1.d0);
			globalMappingElement.setMeanExclusivePercentValue(metric, functionDataLine1.d1);
			break;
		    case 3:
			//Now set the values correctly.
			if((this.getMaxMeanInclusiveValue(metric)) < functionDataLine1.d0){
			    this.setMaxMeanInclusiveValue(metric, functionDataLine1.d0);}
			if((this.getMaxMeanInclusivePercentValue(metric)) < functionDataLine1.d1){
			    this.setMaxMeanInclusivePercentValue(metric, functionDataLine1.d1);}
		    
			globalMappingElement.setMeanInclusiveValue(metric, functionDataLine1.d0);
			globalMappingElement.setMeanInclusivePercentValue(metric, functionDataLine1.d1);
		    
			//Set number of calls/subroutines/usersec per call.
			inputString = br.readLine();

			this.getFunctionDataLine2(inputString);

			//Set the values.
			globalMappingElement.setMeanNumberOfCalls(functionDataLine2.i0);
			globalMappingElement.setMeanNumberOfSubRoutines(functionDataLine2.i1);
			globalMappingElement.setMeanUserSecPerCall(metric, functionDataLine2.d0);

			//Set the max values.
			if((this.getMaxMeanNumberOfCalls()) < functionDataLine2.i0)
			    this.setMaxMeanNumberOfCalls(functionDataLine2.i0);

			if((this.getMaxMeanNumberOfSubRoutines()) < functionDataLine2.i1)
			    this.setMaxMeanNumberOfSubRoutines(functionDataLine2.i1);

			if((this.getMaxMeanUserSecPerCall(metric)) < functionDataLine2.d0)
			    this.setMaxMeanUserSecPerCall(metric, functionDataLine2.d0);

			globalMappingElement.setMeanValuesSet(true);
			break;
		    case 4:
			if((globalMappingElement.getMaxExclusiveValue(metric)) < functionDataLine1.d0)
			    globalMappingElement.setMaxExclusiveValue(metric, functionDataLine1.d0);
			if((globalMappingElement.getMaxExclusivePercentValue(metric)) < functionDataLine1.d1)
			    globalMappingElement.setMaxExclusivePercentValue(metric, functionDataLine1.d1);

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
			globalThreadDataElement.setExclusiveValue(metric, functionDataLine1.d0);
			globalThreadDataElement.setExclusivePercentValue(metric, functionDataLine1.d1);
			//Now check the max values on this thread.
			if((thread.getMaxExclusiveValue(metric)) < functionDataLine1.d0)
			    thread.setMaxExclusiveValue(metric, functionDataLine1.d0);
			if((thread.getMaxExclusivePercentValue(metric)) < functionDataLine1.d1)
			    thread.setMaxExclusivePercentValue(metric, functionDataLine1.d1);
			break;
		    case 5:
			if((globalMappingElement.getMaxInclusiveValue(metric)) < functionDataLine1.d0)
			    globalMappingElement.setMaxInclusiveValue(metric, functionDataLine1.d0);
		    
			if((globalMappingElement.getMaxInclusivePercentValue(metric)) < functionDataLine1.d1)
			    globalMappingElement.setMaxInclusivePercentValue(metric, functionDataLine1.d1);
		    
			thread = nct.getThread(nodeID,contextID,threadID);
			globalThreadDataElement = thread.getFunction(mappingID);
		    
			globalThreadDataElement.setInclusiveValue(metric, functionDataLine1.d0);
			globalThreadDataElement.setInclusivePercentValue(metric, functionDataLine1.d1);
			if((thread.getMaxInclusiveValue(metric)) < functionDataLine1.d0)
			    thread.setMaxInclusiveValue(metric, functionDataLine1.d0);
			if((thread.getMaxInclusivePercentValue(metric)) < functionDataLine1.d1)
			    thread.setMaxInclusivePercentValue(metric, functionDataLine1.d1);
		    
			//Get the number of calls and number of sub routines
			inputString = br.readLine();
			this.getFunctionDataLine2(inputString);

			//Set the values.
			globalThreadDataElement.setNumberOfCalls(functionDataLine2.i0);
			globalThreadDataElement.setNumberOfSubRoutines(functionDataLine2.i1);
			globalThreadDataElement.setUserSecPerCall(metric, functionDataLine2.d0);

			//Set the max values.
			if(globalMappingElement.getMaxNumberOfCalls() < functionDataLine2.i0)
			    globalMappingElement.setMaxNumberOfCalls(functionDataLine2.i0);
			if(thread.getMaxNumberOfCalls() < functionDataLine2.i0)
			    thread.setMaxNumberOfCalls(functionDataLine2.i0);

			if(globalMappingElement.getMaxNumberOfSubRoutines() < functionDataLine2.i1)
			    globalMappingElement.setMaxNumberOfSubRoutines(functionDataLine2.i1);
			if(thread.getMaxNumberOfSubRoutines() < functionDataLine2.i1)
			    thread.setMaxNumberOfSubRoutines(functionDataLine2.i1);
		    
			if(globalMappingElement.getMaxUserSecPerCall(metric) < functionDataLine2.d0)
			    globalMappingElement.setMaxUserSecPerCall(metric, functionDataLine2.d0);
			if(thread.getMaxUserSecPerCall(metric) < functionDataLine2.d0)
			    thread.setMaxUserSecPerCall(metric, functionDataLine2.d0);
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
				getUserEventData(s1);
				System.out.println("noc:"+usereventDataLine.i0+"min:"+usereventDataLine.d1+"max:"+usereventDataLine.d0+"mean:"+usereventDataLine.d2);
			    
				//Initialize the user list for this thread.
				if(j == 0){
				    //Note that this works correctly because we process the user events in a different manner.
				    //ALL the user events for each THREAD NODE are processed in the above for-loop.  Therefore,
				    //the below for-loop is only run once on each THREAD NODE.
				
				    if(firstRead){
					(nct.getThread(nodeID,contextID,threadID)).initializeUsereventList(numberOfUserEvents);
				    }
				}

				int userEventID = globalMapping.addGlobalMapping(usereventDataLine.s0, 2);
				if(usereventDataLine.i0 != 0){
				    //Update the max values if required.
				    //Grab the correct global mapping element.
				    globalMappingElement = globalMapping.getGlobalMappingElement(userEventID, 2);
				    if((globalMappingElement.getMaxUserEventNumberValue()) < usereventDataLine.i0)
					globalMappingElement.setMaxUserEventNumberValue(usereventDataLine.i0);
				    if((globalMappingElement.getMaxUserEventMinValue()) < usereventDataLine.d1)
					globalMappingElement.setMaxUserEventMinValue(usereventDataLine.d1);
				    if((globalMappingElement.getMaxUserEventMaxValue()) < usereventDataLine.d0)
					globalMappingElement.setMaxUserEventMaxValue(usereventDataLine.d0);
				    if((globalMappingElement.getMaxUserEventMeanValue()) < usereventDataLine.d2)
					globalMappingElement.setMaxUserEventMeanValue(usereventDataLine.d2);
				
				    GlobalThreadDataElement tmpGTDEUE = new GlobalThreadDataElement(globalMapping.getGlobalMappingElement(userEventID, 2), true);
				    tmpGTDEUE.setUserEventNumberValue(usereventDataLine.i0);
				    tmpGTDEUE.setUserEventMinValue(usereventDataLine.d1);
				    tmpGTDEUE.setUserEventMaxValue(usereventDataLine.d0);
				    tmpGTDEUE.setUserEventMeanValue(usereventDataLine.d2);
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

		//Set firstRead to false.
		firstRead = false;
	    
		time = (System.currentTimeMillis()) - time;
		System.out.println("Done processing data file, please wait ......");
		System.out.println("Time to process file (in milliseconds): " + time);
	    }
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

    private void getFunctionDataLine1(String string){
	try{
	    StringTokenizer st1 = new StringTokenizer(string, "\"");
	    st1.nextToken();
	    functionDataLine1.s0 = st1.nextToken(); //Name
	    
	    StringTokenizer st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");
	    st2.nextToken();
	    functionDataLine1.d0 = Double.parseDouble(st2.nextToken()); //Value
	    functionDataLine1.d1 = Double.parseDouble(st2.nextToken()); //Percent value
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD08");
	}
    }

    private void getFunctionDataLine2(String string){ 
	try{
	    StringTokenizer getMappingIDTokenizer = new StringTokenizer(string, " \t\n\r");
	    getMappingIDTokenizer.nextToken();
	    getMappingIDTokenizer.nextToken();
	    getMappingIDTokenizer.nextToken();
	    
	    functionDataLine2.i0 = (int) Double.parseDouble(getMappingIDTokenizer.nextToken()); //Number of calls
	    functionDataLine2.i1 = (int) Double.parseDouble(getMappingIDTokenizer.nextToken()); //Number of subroutines
	    functionDataLine2.d0 = Double.parseDouble(getMappingIDTokenizer.nextToken()); //User seconds per call
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD10");
	}
    }

    private void getUserEventData(String string){
	try{
	    StringTokenizer st1 = new StringTokenizer(string, "\"");
	    st1.nextToken();
	    usereventDataLine.s0 = st1.nextToken();

	    StringTokenizer st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");
	    usereventDataLine.i0 = (int) Double.parseDouble(st2.nextToken()); //Number of calls.
	    usereventDataLine.d0 = Double.parseDouble(st2.nextToken()); //Max
	    usereventDataLine.d1 = Double.parseDouble(st2.nextToken()); //Min
	    usereventDataLine.d2 = Double.parseDouble(st2.nextToken()); //Mean
	    usereventDataLine.d3 = Double.parseDouble(st2.nextToken()); //Standard Deviation.
	}
	catch(Exception e){
	    System.out.println("An error occured!");
	    e.printStackTrace();
	}
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
    private boolean firstRead = true;
    private boolean groupNamesCheck = false;
    private LineData functionDataLine1 = new LineData();
    private LineData functionDataLine2 = new LineData();
    private LineData  usereventDataLine = new LineData();
    
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
