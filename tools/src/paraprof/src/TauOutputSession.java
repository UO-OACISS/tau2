/* 
   TauOutputSession.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

/*
  To do: 
  1) Add some sanity checks to make sure that multiple metrics really do belong together.
  For example, wrap the creation of nodes, contexts, threads, global mapping elements, and
  the like so that they do not occur after the first metric has been loaded.  This will
  not of course ensure 100% that the data is consistent, but it will at least prevent the
  worst cases.
*/

package paraprof;



import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

import java.io.*;
import java.util.*;
import dms.dss.*;

public class TauOutputSession extends ParaProfDataSession{

    public TauOutputSession(){
	super();
    }

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

		//Need to call increaseVectorStorage() on all objects that require it.
		this.increaseVectorStorage();
		    
		//Only need to call addDefaultToVectors() if not the first run.
		if(!(this.firstMetric())){
		    if(ParaProf.debugIsOn)
			System.out.println("Increasing the storage for the new counter.");
		    
		    for(Enumeration e1 = (this.getGlobalMapping().getMapping(0)).elements(); e1.hasMoreElements() ;){
			GlobalMappingElement tmpGME = (GlobalMappingElement) e1.nextElement();
			tmpGME.incrementStorage();
		    }
		    
		    for(Enumeration e2 = (this.getGlobalMapping().getMapping(2)).elements(); e2.hasMoreElements() ;){
			GlobalMappingElement tmpGME = (GlobalMappingElement) e2.nextElement();
			tmpGME.incrementStorage();
		    }
		    
		    for(Enumeration e3 = this.getNCT().getNodes().elements(); e3.hasMoreElements() ;){
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

		Metric metricRef = this.addMetric();
		metric = metricRef.getID();

		files = (File[]) e.nextElement();
		for(int i=0;i<files.length;i++){
		    System.out.println("Processing file: " + files[i].getName());

		    FileInputStream fileIn = new FileInputStream(files[i]);
		    InputStreamReader inReader = new InputStreamReader(fileIn);
		    BufferedReader br = new BufferedReader(inReader);

		    int[] nct = this.getNCT(files[i].getName());
		    nodeID = nct[0];
		    contextID = nct[1];
		    threadID = nct[2];
		    
		    node = this.getNCT().getNode(nodeID);
		    if(node==null)
			node = this.getNCT().addNode(nodeID);
		    context = node.getContext(contextID);
		    if(context==null)
			context = node.addContext(contextID);
		    thread = context.getThread(threadID);
		    if(thread==null){
			thread = context.addThread(threadID);
			thread.initializeFunctionList(this.getNumberOfMappings());
		    }
		    System.out.println("n,c,t: " + nct[0] + "," + nct[1] + "," + nct[2]);

		    //####################################
		    //First  Line
		    //####################################
		    inputString = br.readLine();

		    if(i==0){
			//Set the counter name.
			String counterName = getCounterName(inputString);
			//Now set the counter name.
			if(counterName == null)
			    counterName = new String("Time");
			System.out.println("Counter name is: " + counterName);
		    
			metricRef.setName(counterName);
		    }
		    
		    //####################################
		    //End - First Line
		    //####################################
		    
		    //####################################
		    //Second Line
		    //####################################
		    //This line is not required. Check to make sure that it is there however.
		    inputString = br.readLine();
		    if(inputString == null)
			return;
		    //####################################
		    //End - Second Line
		    //####################################

		    while(((inputString = br.readLine()) != null) && ((inputString.indexOf('"'))==0)){
			System.out.println(inputString);
			this.getFunctionDataLine(inputString);
			String groupNames = this.getGroupNames(inputString);
			//Calculate usec/call
			double usecCall = functionDataLine.d0/functionDataLine.i0;
			System.out.println("Name:"+functionDataLine.s0);
			System.out.println("Calls:"+functionDataLine.i0);
			System.out.println("Subrs:"+functionDataLine.i1);
			System.out.println("Excl:"+functionDataLine.d0);
			System.out.println("Incl:"+functionDataLine.d1);
			System.out.println("ProfileCalls:"+functionDataLine.d2);
			System.out.println("groupNames:"+groupNames);
			if(functionDataLine.i0 !=0){
			    mappingID = this.getGlobalMapping().addGlobalMapping(functionDataLine.s0, 0);
			    globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 0);
			    globalMappingElement.incrementCounter();
			    globalThreadDataElement = thread.getFunction(mappingID);
			    
			    if(globalThreadDataElement == null){
				globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(mappingID, 0), false);
				thread.addFunction(globalThreadDataElement, mappingID);
			    }
			    
			    globalThreadDataElement.setMappingExists();
			    globalThreadDataElement.setExclusiveValue(metric, functionDataLine.d0);
			    globalThreadDataElement.setInclusiveValue(metric, functionDataLine.d1);
			    globalThreadDataElement.setNumberOfCalls(functionDataLine.i0);
			    globalThreadDataElement.setNumberOfSubRoutines(functionDataLine.i1);
			    globalThreadDataElement.setUserSecPerCall(metric, usecCall);
			    
			    globalMappingElement.incrementTotalExclusiveValue(functionDataLine.d0);
			    globalMappingElement.incrementTotalInclusiveValue(functionDataLine.d1);
			    
			    //Set the max values.
			    if((globalMappingElement.getMaxExclusiveValue(metric)) < functionDataLine.d0)
				globalMappingElement.setMaxExclusiveValue(metric, functionDataLine.d0);
			    if((thread.getMaxExclusiveValue(metric)) < functionDataLine.d0)
				thread.setMaxExclusiveValue(metric, functionDataLine.d0);
			    
			    if((globalMappingElement.getMaxInclusiveValue(metric)) < functionDataLine.d1)
				globalMappingElement.setMaxInclusiveValue(metric, functionDataLine.d1);
			    if((thread.getMaxInclusiveValue(metric)) < functionDataLine.d1)
				thread.setMaxInclusiveValue(metric, functionDataLine.d1);
			    
			    if(globalMappingElement.getMaxNumberOfCalls() < functionDataLine.i0)
				globalMappingElement.setMaxNumberOfCalls(functionDataLine.i0);
			    if(thread.getMaxNumberOfCalls() < functionDataLine.i0)
				thread.setMaxNumberOfCalls(functionDataLine.i0);
			    
			    if(globalMappingElement.getMaxNumberOfSubRoutines() < functionDataLine.i1)
				globalMappingElement.setMaxNumberOfSubRoutines(functionDataLine.i1);
			    if(thread.getMaxNumberOfSubRoutines() < functionDataLine.i1)
				thread.setMaxNumberOfSubRoutines(functionDataLine.i1);
			    
			    if(globalMappingElement.getMaxUserSecPerCall(metric) < usecCall)
				globalMappingElement.setMaxUserSecPerCall(metric, usecCall);
			    if(thread.getMaxUserSecPerCall(metric) < usecCall)
				thread.setMaxUserSecPerCall(metric, usecCall);
			}
		    }
		    System.out.println("done processing functions");
		    
		    while(((inputString = br.readLine()) != null) && ((inputString.indexOf('"'))==0)){
			System.out.println(inputString);
		    }
		    System.out.println("done processing aggregates");

		    //If the above while loop was not terminated because inputString was null, then
		    //userevents are present. Skip userevent heading line, then get the userevents.
		    br.readLine();
		    
		    if(this.firstMetric()){
			while(((inputString = br.readLine()) != null) && ((inputString.indexOf('"'))==0)){
			    System.out.println(inputString);
			    this.getUserEventData(inputString);
			    System.out.println("eventname:"+usereventDataLine.s0);
			    System.out.println("numevents:"+usereventDataLine.i0);
			    System.out.println("max:"+usereventDataLine.d0);
			    System.out.println("min:"+usereventDataLine.d1);
			    System.out.println("mean:"+usereventDataLine.d2);
			    System.out.println("sumsqr:"+usereventDataLine.d3);
			}
			System.out.println("done processing userevents");
		    }
		    
		    //The thread object takes care of computing maximums and totals for a given metric.
		    thread.setThreadSummaryData(metric);

		    //######
		    //Compute percent values.
		    //######
		    ListIterator l = thread.getFunctionListIterator();
		    while(l.hasNext()){
			globalThreadDataElement = (GlobalThreadDataElement) l.next();
			double exclusiveTotal = thread.getTotalExclusiveValue(metric);
			double inclusiveMax = thread.getMaxInclusiveValue(metric);
			
			if(globalThreadDataElement != null){
			    globalMappingElement =
				this.getGlobalMapping().getGlobalMappingElement(globalThreadDataElement.getMappingID(), 0);
			    
			    double d1 = globalThreadDataElement.getExclusiveValue(metric);
			    double d2 = globalThreadDataElement.getInclusiveValue(metric);
			    
			    if(exclusiveTotal!=0){
				double result = (d1/exclusiveTotal)*100.00;
				globalThreadDataElement.setExclusivePercentValue(metric, result);
				//Now do the global mapping element exclusive stuff.
				if((globalMappingElement.getMaxExclusivePercentValue(metric)) < result)
				    globalMappingElement.setMaxExclusivePercentValue(metric, result);
			    }
			    
			    if(inclusiveMax!=0){
				double result = (d2/inclusiveMax) * 100;
				globalThreadDataElement.setInclusivePercentValue(metric, result);
				//Now do the global mapping element exclusive stuff.
				if((globalMappingElement.getMaxInclusivePercentValue(metric)) < result)
				    globalMappingElement.setMaxInclusivePercentValue(metric, result);
			    }
			}
		    }
		    //######
		    //End - Compute percent values.
		    //######
		    
		    //Call the setThreadSummaryData function again on this thread so that
		    //it can fill in all the summary data.
		    thread.setThreadSummaryData(metric);
		}
		
		ListIterator l = this.getGlobalMapping().getMappingIterator(0);
		double exclusiveTotal = 0.0;
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

		l = this.getGlobalMapping().getMappingIterator(0);
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
	}
        catch(Exception e){
	    ParaProf.systemError(e, null, "SSD01");
	}
    }
    
    //####################################
    //Private Section.
    //####################################

    //######
    //profile.*.*.* string processing methods.
    //######
     private int[] getNCT(String string){
	int[] nct = new int[3];
	StringTokenizer st = new StringTokenizer(string, ".\t\n\r");
	st.nextToken();
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

    private void getFunctionDataLine(String string){
	try{
	    StringTokenizer st1 = new StringTokenizer(string, "\"");
	    functionDataLine.s0 = st1.nextToken(); //Name
	    
	    StringTokenizer st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");
	    functionDataLine.i0 = Integer.parseInt(st2.nextToken()); //Calls
	    functionDataLine.i1 = Integer.parseInt(st2.nextToken()); //Subroutines
	    functionDataLine.d0 = Double.parseDouble(st2.nextToken()); //Exclusive
	    functionDataLine.d1 = Double.parseDouble(st2.nextToken()); //Inclusive
	    functionDataLine.d2 = Double.parseDouble(st2.nextToken()); //ProfileCalls
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD08");
	}
    }

    private String getGroupNames(String string){
	try{  
	    StringTokenizer getMappingNameTokenizer = new StringTokenizer(string, "\"");
 	    getMappingNameTokenizer.nextToken();
	    String str = getMappingNameTokenizer.nextToken();
        
	    //Just do the group check once.
	    if(!(this.groupCheck())){
		//If present, "GROUP=" will be in this token.
		int tmpInt = str.indexOf("GROUP=");
		if(tmpInt > 0){
		    this.setGroupNamesPresent(true);
		}
		this.setGroupCheck(true);
	    }
	    
	    if(groupNamesPresent()){
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

    private void getUserEventData(String string){
	try{
	    StringTokenizer st1 = new StringTokenizer(string, "\"");
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

    //######
    //End - profile.*.*.* string processing methods.
    //######

    //####################################
    //End - Private Section.
    //####################################

    //####################################
    //Instance data.
    //####################################
    private LineData functionDataLine = new LineData();
    private LineData  usereventDataLine = new LineData();
    //####################################
    //End - Instance data.
    //####################################
}
