/* 
   Name:        ParaProfDataSession.java
   Author:      Robert Bell
   Description:  
*/

package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.io.*;

public abstract class ParaProfDataSession  extends DataSession{
    public ParaProfDataSession() {
	super();
	this.setGlobalMapping(new GlobalMapping());
	this.setNCT(new NCT());
    }

    public ParaProfDataSession(boolean debug){
	super();
	this.debug = debug;
	this.setGlobalMapping(new GlobalMapping());
	this.setNCT(new NCT());
    }

    //####################################
    //Public Section.
    //####################################

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
     * @return	DataSessionIterator object of all Experiments.  If there is an Application saved 
     *          in the DataSession, then only the Experiments for that Application are returned.
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
     * Returns a ListIterator of IntervalEvent objects.
     *
     * @return	DataSessionIterator object of all IntervalEvents.  If there is an Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) saved in the DataSession, then only the IntervalEvents for that Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) are returned.
     * @see	DataSessionIterator
     * @see	IntervalEvent
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     */
    public ListIterator getIntervalEvents(){
	return null;}

    /**
     * Set the IntervalEvent for this DataSession.  The DataSession object will maintain a reference to the IntervalEvent referenced by the id.  To clear this reference, call setIntervalEvent(IntervalEvent) with a null reference.
     *
     * @param	id unique id of the IntervalEvent object to be saved.
     * @see	IntervalEvent
     * @see	DataSession#setIntervalEvent(IntervalEvent)
     */
    public IntervalEvent setIntervalEvent(int id){
	return null;}

    /**
     * Returns the IntervalEvent identified by the unique intervalEvent id.
     *
     * @param	intervalEventID unique id of the IntervalEvent object.
     * @return	IntervalEvent object identified by the unique intervalEvent id.
     * @see	IntervalEvent
     */
    public IntervalEvent getIntervalEvent(int intervalEventID){
	return null;}
	
    /**
     * Returns a ListIterator of AtomicEvent objects.
     *
     * @return	DataSessionIterator object of all AtomicEvents.  If there is an Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) saved in the DataSession, then only the AtomicEvents for that Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) are returned.
     * @see	DataSessionIterator
     * @see	AtomicEvent
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     */
    public ListIterator getAtomicEvents(){
	return null;}

    /**
     * Set the AtomicEvent for this DataSession.  The DataSession object will maintain a reference to the AtomicEvent referenced by the id.  To clear this reference, call setAtomicEvent(AtomicEvent) with a null reference.
     *
     * @param	id unique id of the AtomicEvent object to be saved.
     * @see	AtomicEvent
     * @see	DataSession#setAtomicEvent(AtomicEvent)
     */
    public AtomicEvent setAtomicEvent(int id){
	return null;}

    /**
     * Returns the AtomicEvent identified by the unique atomic event id.
     *
     * @param	atomicEventID unique id of the AtomicEvent object to be saved.
     * @return	AtomicEvent object identified by the unique atomic event id.
     * @see	AtomicEvent
     */
    public AtomicEvent getAtomicEvent(int atomicEventID){
	return null;}
	
    /**
     * Returns the IntervalEventData for this DataSession
     *
     * @return	DataSessionIterator of IntervalLocationProfile objects.  If there is an Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or IntervalEvent(s) saved in the DataSession, then only the IntervalEvents for that Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or IntervalEvent(s) are returned.
     * @see	DataSessionIterator
     * @see	IntervalLocationProfile
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     * @see	DataSession#setIntervalEvent
     */
    public ListIterator getIntervalEventData(){
	return null;}

    /**
     * Returns the AtomicEventData for this DataSession
     *
     * @return	DataSessionIterator of AtomicEventData objects.  If there is an Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or IntervalEvent(s) saved in the DataSession, then only the AtomicEvents for that Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or IntervalEvent(s) are returned.
     * @see	DataSessionIterator
     * @see	AtomicLocationProfile
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     * @see	DataSession#setAtomicEvent
     */
    public ListIterator getAtomicEventData(){
	return null;}

    /**
     * Gets the interval record detail for this intervalEvent.
     *
     * @param intervalEvent
     */
    public void getIntervalEventDetail(IntervalEvent intervalEvent){}
    
    /**
     * Gets the atomic record detail for this atomicEvent.
     *
     * @param atomicEvent
     */
    public void getAtomicEventDetail(AtomicEvent atomicEvent){}
    
    /**
     * Saves the trial.
     *
     * @return database index ID of the saved trial
     */
    public int saveTrial(){return -1;}
    
    /**
     * Saves the Trial.
     *
     * @param trial
     * @return ID of the saved trial
     */
    public int saveTrial(Trial trial){return -1;}
    
    /**
     * Saves the IntervalEvent.
     *
     * @param intervalEvent
     * @param newTrialID
     * @param newMetHash
     * @return ID of the saved intervalEvent
     */
    public int saveIntervalEvent(IntervalEvent intervalEvent, int newTrialID, Hashtable newMetHash){return -1;}
    
    /**
     * Saves the IntervalLocationProfile.
     *
     * @param intervalEventData
     * @param newIntervalEventID
     * @param newMetHash
     */
    public void saveIntervalEventData(IntervalLocationProfile intervalEventData, int newIntervalEventID, Hashtable newMetHash){}
    
    /**
     * Saves the AtomicEvent object.
     *
     * @param atomicEvent
     * @return ID of the saved atomic event
     */
    public int saveAtomicEvent(AtomicEvent atomicEvent, int newTrialID){return -1;}

    
    /**
     * Saves the atomicEventData object.
     *
     * @param atomicEventData
     */
    public void saveAtomicEventData(AtomicLocationProfile atomicEventData, int newAtomicEventID){}
    
    public boolean profileStatsPresent(){
	return profileStatsPresent;}

    public boolean profileCallsPresent(){
	return profileCallsPresent();}

    public boolean aggregatesPresent(){
	return aggregatesPresent;}
    
    public boolean groupNamesPresent(){
	return groupNamesPresent;}

    public boolean userEventsPresent(){
	return userEventsPresent;}
  
    public boolean callPathDataPresent(){
	return callPathDataPresent;}

    public void setDebug(boolean debug){
	try{
	    this.debug = debug;
	    if(debug && out==null){
		Class c = this.getClass();
		out = new PrintWriter(new FileWriter(new File(c.getName()+".out")));
	    }
	}
	catch(IOException exception){
	    exception.printStackTrace();
	}
    }
    
    public boolean debug(){
	return debug;}

    //Gets the maximum id reached for all nodes, context, and threads.
    //This takes into account that id values might not be contiguous (ie, we do not
    //simply get the maximum number seen.  For example, there might be only one profile
    //in the system for n,c,t of 0,1,234.  We do not want to just return [1,1,1] representing
    //the number of items, but the actual id values which are the largest (ie, return [0,1,234]).
    public int[] getMaxNCTNumbers(){
	if(maxNCT==null){
	    maxNCT = new int[3];
	    for(int i=0;i<3;i++){
		maxNCT[i]=0;}
	    for(Enumeration e1 = (this.getNCT().getNodes()).elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		if(node.getNodeID()>maxNCT[0])
		    maxNCT[0]=node.getNodeID();
		for(Enumeration e2 = (node.getContexts()).elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    if(context.getContextID()>maxNCT[1])
			maxNCT[1]=context.getContextID();
		    for(Enumeration e3 = (context.getThreads()).elements(); e3.hasMoreElements() ;){
			edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
			if(thread.getThreadID()>maxNCT[2])
			    maxNCT[2]=thread.getThreadID();
		    }
		}
	    }
	    
	}
	return maxNCT;
    }
    
    
    //After loading all data, this function can be called to manage the generation of all the derived data that ParaProf needs
    //for a particular mapping selection.
    public void generateDerivedData(int mappingSelection){
	for(Enumeration e = this.getNCT().getThreads().elements(); e.hasMoreElements() ;){
	    ((Thread) e.nextElement()).setThreadDataAllMetrics();
	}
	this.setMeanDataAllMetrics(mappingSelection);
    }

        
    public void setMeanData(int mappingSelection, int metric){
	// see comments in setMeandataAllMetrics for the computation

	if(this.debug()){
	    this.outputToFile("####################################");
	    this.outputToFile("Setting mean data :: public void setMeanData(int mappingSelection, int metric)");
	}


	double exclSum = 0;
	double inclSum = 0;
	double maxInclSum = 0;
	double callSum = 0;
	double subrSum = 0;


	GlobalMapping allFunctions = this.getGlobalMapping();
	ListIterator l = allFunctions.getMappingIterator(mappingSelection);


	while (l.hasNext()) { // for each function
	    GlobalMappingElement function = (GlobalMappingElement) l.next();

	    callSum = 0;
	    subrSum = 0;
	    exclSum = 0;
	    inclSum = 0;

	    int functionId = function.getMappingID();

	    // this must be stored somewhere else, but I'm going to compute it since I don't know where
	    int numThreads = 0;

	    for (Enumeration e1 = this.getNCT().getNodes().elements(); e1.hasMoreElements() ;) {
		Node node = (Node) e1.nextElement();
		for (Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;) {
		    Context context = (Context) e2.nextElement();
		    for (Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements() ;) {
			edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
			GlobalThreadDataElement globalThreadDataElement = thread.getFunction(functionId);

			if (globalThreadDataElement != null) { // if this function was called for this nct

			    exclSum += globalThreadDataElement.getExclusiveValue(metric);
			    inclSum += globalThreadDataElement.getInclusiveValue(metric);
				
			    callSum+=globalThreadDataElement.getNumberOfCalls();
			    subrSum+=globalThreadDataElement.getNumberOfSubRoutines();
			}
			numThreads++;
		    }
		}
	    }

	    // set the totals for all but percentages - need to do those later...
	    function.setTotalNumberOfCalls(callSum);
	    function.setTotalNumberOfSubRoutines(subrSum);

	    // mean is just the total / numThreads
	    function.setMeanNumberOfCalls((double)callSum / numThreads);
	    function.setMeanNumberOfSubRoutines((double)subrSum / numThreads);

	    function.setTotalExclusiveValue(metric, exclSum);
	    function.setTotalInclusiveValue(metric, inclSum);
	    function.setTotalUserSecPerCall(metric, inclSum / callSum);
	    
	    // mean data computed as above in comments
	    function.setMeanExclusiveValue(metric, exclSum / numThreads);
	    function.setMeanInclusiveValue(metric, inclSum / numThreads);
	    function.setMeanUserSecPerCall(metric, inclSum / callSum);

	    if (inclSum > maxInclSum) {
		maxInclSum = inclSum;
	    }


	    // now set the max values for mean
	    if (function.getMeanNumberOfSubRoutines() > allFunctions.getMaxMeanNumberOfSubRoutines())
		allFunctions.setMaxMeanNumberOfSubRoutines(function.getMeanNumberOfSubRoutines());

	    if (function.getMeanNumberOfCalls() > allFunctions.getMaxMeanNumberOfCalls())
		allFunctions.setMaxMeanNumberOfCalls(function.getMeanNumberOfCalls());
	    
	    if (function.getMeanExclusiveValue(metric) > allFunctions.getMaxMeanExclusiveValue(metric))
		allFunctions.setMaxMeanExclusiveValue(metric, function.getMeanExclusiveValue(metric));
	    
	    if (function.getMeanInclusiveValue(metric) > allFunctions.getMaxMeanInclusiveValue(metric))
		allFunctions.setMaxMeanInclusiveValue(metric, function.getMeanInclusiveValue(metric));
	}


	// now compute percentages since we now have max(all incls for total)
	// for each function
	l = allFunctions.getMappingIterator(mappingSelection);
	while (l.hasNext()) {
	    
	    GlobalMappingElement function = (GlobalMappingElement) l.next();
	    
	    if (maxInclSum != 0) {
		function.setTotalInclusivePercentValue(metric, function.getTotalInclusiveValue(metric) / maxInclSum * 100);
		function.setTotalExclusivePercentValue(metric, function.getTotalExclusiveValue(metric) / maxInclSum * 100);
		
		// mean is exactly the same
		function.setMeanInclusivePercentValue(metric, function.getTotalInclusiveValue(metric) / maxInclSum * 100);
		function.setMeanExclusivePercentValue(metric, function.getTotalExclusiveValue(metric) / maxInclSum * 100);
	    }
	    
	    
	    // set max mean stuff
	    if (function.getMeanExclusivePercentValue(metric) > allFunctions.getMaxMeanExclusivePercentValue(metric))
		allFunctions.setMaxMeanExclusivePercentValue(metric, function.getMeanExclusivePercentValue(metric));
	    if (function.getMeanInclusivePercentValue(metric) > allFunctions.getMaxMeanInclusivePercentValue(metric))
		allFunctions.setMaxMeanInclusivePercentValue(metric, function.getMeanInclusivePercentValue(metric));
	    function.setMeanValuesSet(true);
	}
	
	if(this.debug()){
	    this.outputToFile("Done - Setting mean data :: public void setMeanData(int mappingSelection, int metric)");
	    this.outputToFile("####################################");
	}
    }
    

    public void setMeanDataAllMetrics(int mappingSelection){

	// Given, excl, incl, call, subr for each thread, for each function

	// for a node:

	// inclpercent = incl / (max(all incls for this thread)) * 100
	// exclpercent = excl / (max(all incls for this thread)) * 100
	// inclpercall = incl / call

	// for the total:

	//   for each function:
	//     incl = sum(all threads, incl)
	//     excl = sum(all threads, excl)
	//     call = sum(all threads, call)
	//     subr = sum(all threads, subr)
    
	//     inclpercent = incl / (max(all incls for total)) * 100
	//     exclpercent = excl / (max(all incls for total)) * 100
	//     inclpercall = incl / call

	// for the mean:
	//   for each function:
	//     incl = total(incl) / numThreads
	//     excl = total(excl) / numThreads
	//     call = total(call) / numThreads
	//     subr = total(subr) / numThreads

	//     inclpercent = total(inclpercent)
	//     exclpercent = total(exclpercent)
	//     inclpercall = total(inclpercall) 


	int numberOfMetrics = this.getNumberOfMetrics();

	double[] exclSum = new double[numberOfMetrics];
	double[] inclSum = new double[numberOfMetrics];
	double[] maxInclSum = new double[numberOfMetrics];
	int callSum = 0;
	int subrSum = 0;

	for (int i=0;i<numberOfMetrics;i++) {
	    maxInclSum[i] = 0;
	}


	GlobalMapping allFunctions = this.getGlobalMapping();
	ListIterator l = allFunctions.getMappingIterator(mappingSelection);


	while (l.hasNext()) { // for each function
	    GlobalMappingElement function = (GlobalMappingElement) l.next();

	    callSum = 0;
	    subrSum = 0;
	    for (int i=0;i<numberOfMetrics;i++) {
		exclSum[i] = 0;
		inclSum[i] = 0;
	    }

	    

	    int functionId = function.getMappingID();

	    // this must be stored somewhere else, but I'm going to compute it since I don't know where
	    int numThreads = 0;

	    for (Enumeration e1 = this.getNCT().getNodes().elements(); e1.hasMoreElements() ;) {
		Node node = (Node) e1.nextElement();
		for (Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;) {
		    Context context = (Context) e2.nextElement();
		    for (Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements() ;) {
			edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
			GlobalThreadDataElement globalThreadDataElement = thread.getFunction(functionId);

			if (globalThreadDataElement != null) { // if this function was called for this nct

			    for (int i=0;i<numberOfMetrics;i++) {

				exclSum[i]+=globalThreadDataElement.getExclusiveValue(i);
				inclSum[i]+=globalThreadDataElement.getInclusiveValue(i);
				
				// the same for every metric
				if (i == 0) {
				    callSum+=globalThreadDataElement.getNumberOfCalls();
				    subrSum+=globalThreadDataElement.getNumberOfSubRoutines();
				}
			    }
			}
			numThreads++;
		    }
		}
	    }

	    // set the totals for all but percentages - need to do those later...
	    function.setTotalNumberOfCalls(callSum);
	    function.setTotalNumberOfSubRoutines(subrSum);

	    // mean is just the total / numThreads
	    function.setMeanNumberOfCalls((double)callSum / numThreads);
	    function.setMeanNumberOfSubRoutines((double)subrSum / numThreads);

	    for (int i=0;i<numberOfMetrics;i++) {
		function.setTotalExclusiveValue(i, exclSum[i]);
		function.setTotalInclusiveValue(i, inclSum[i]);
		function.setTotalUserSecPerCall(i, inclSum[i] / callSum);

		// mean data computed as above in comments
		function.setMeanExclusiveValue(i, exclSum[i] / numThreads);
		function.setMeanInclusiveValue(i, inclSum[i] / numThreads);
		function.setMeanUserSecPerCall(i, inclSum[i] / callSum);

		if (inclSum[i] > maxInclSum[i]) {
		    maxInclSum[i] = inclSum[i];
		}
	    }


	    // now set the max values for mean
	    if (function.getMeanNumberOfSubRoutines() > allFunctions.getMaxMeanNumberOfSubRoutines())
		allFunctions.setMaxMeanNumberOfSubRoutines(function.getMeanNumberOfSubRoutines());

	    if (function.getMeanNumberOfCalls() > allFunctions.getMaxMeanNumberOfCalls())
		allFunctions.setMaxMeanNumberOfCalls(function.getMeanNumberOfCalls());
	    
	    for (int i=0;i<numberOfMetrics;i++) {
		if (function.getMeanExclusiveValue(i) > allFunctions.getMaxMeanExclusiveValue(i))
		    allFunctions.setMaxMeanExclusiveValue(i,function.getMeanExclusiveValue(i));
		
		if (function.getMeanInclusiveValue(i) > allFunctions.getMaxMeanInclusiveValue(i))
		    allFunctions.setMaxMeanInclusiveValue(i,function.getMeanInclusiveValue(i));
	    }
	}


	// now compute percentages since we now have max(all incls for total)
	// for each function
	l = allFunctions.getMappingIterator(mappingSelection);
	while (l.hasNext()) {
	    
	    GlobalMappingElement function = (GlobalMappingElement) l.next();
	    
	    for (int i=0;i<numberOfMetrics;i++) {
		if (maxInclSum[i] != 0) {
		    function.setTotalInclusivePercentValue(i, function.getTotalInclusiveValue(i) / maxInclSum[i] * 100);
		    function.setTotalExclusivePercentValue(i, function.getTotalExclusiveValue(i) / maxInclSum[i] * 100);

		    // mean is exactly the same
		    function.setMeanInclusivePercentValue(i, function.getTotalInclusiveValue(i) / maxInclSum[i] * 100);
		    function.setMeanExclusivePercentValue(i, function.getTotalExclusiveValue(i) / maxInclSum[i] * 100);
		}


		// set max mean stuff
		if (function.getMeanExclusivePercentValue(i) > allFunctions.getMaxMeanExclusivePercentValue(i))
		    allFunctions.setMaxMeanExclusivePercentValue(i, function.getMeanExclusivePercentValue(i));
		if (function.getMeanInclusivePercentValue(i) > allFunctions.getMaxMeanInclusivePercentValue(i))
		    allFunctions.setMaxMeanInclusivePercentValue(i, function.getMeanInclusivePercentValue(i));
	    }
	    function.setMeanValuesSet(true);
	}
    }
    //######
    //End - Set mean values functions.
    //######

    //####################################
    //End - Public Section.
    //####################################

    //####################################
    //Protected Section.
    //####################################
    protected Metric addMetric(String name){
	Metric metric = new Metric();
	metric.setName(name);
	addMetric(metric);
	return metric;
    }

    protected void setProfileStatsPresent(boolean profileStatsPresent){
	this.profileStatsPresent = profileStatsPresent;}

    protected void setProfileCallsPresent(boolean profileCallsPresent){
	this.profileCallsPresent = profileCallsPresent;}

    protected void setAggregatesPresent(boolean aggregatesPresent){
	this.aggregatesPresent = aggregatesPresent;}

    protected void setGroupNamesPresent(boolean groupNamesPresent){
	this.groupNamesPresent = groupNamesPresent;}
  
    protected void setUserEventsPresent(boolean userEventsPresent){
	this.userEventsPresent = userEventsPresent;}

    protected void setCallPathDataPresent(boolean callPathDataPresent){
	this.callPathDataPresent = callPathDataPresent;}

    protected void setFirstMetric(boolean firstMetric){
	this.firstMetric = firstMetric;}

    protected boolean firstMetric(){
	return firstMetric;}

    protected void setGroupCheck(boolean groupCheck){
	this.groupCheck = groupCheck;}

    protected boolean groupCheck(){
	return groupCheck;}

    protected void outputToFile(String s){
	if(out!=null)
	    out.println(s);
    }

    protected void flushDebugFileBuffer(){
	if(out!=null)
	    out.flush();
    }

    protected void closeDebugFile(){
	if(out!=null)
	    out.close();
    }
    //####################################
    //End - Protected Section.
    //####################################

    //####################################
    //Instance data.
    //####################################

    //######
    //Private Section.
    //######
    protected Object initializeObject = null;

    private boolean firstMetric = true;
    private boolean groupCheck = false;
    
    private int totalNumberOfContexts = -1;
    private int totalNumberOfThreads = -1;
    private boolean profileStatsPresent = false;
    private boolean profileCallsPresent = false;
    private boolean aggregatesPresent = false;
    private boolean groupNamesPresent = false;
    private boolean userEventsPresent = false;
    private boolean callPathDataPresent = false;

    private int[] maxNCT = null;

    private boolean debug = false;
    //When in debugging mode, this class can print a lot of data.
    //Initialized in this.setDebug(...).
    private PrintWriter out = null;
    
    //######
    //End - Private Section.
    //######

    //####################################
    //End - Instance data.
    //####################################
}
