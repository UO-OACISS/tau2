

/* 
   ParaProfDataSession.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import java.io.*;
import dms.dss.*;

public abstract class ParaProfDataSession  extends DataSession implements Runnable{
    public ParaProfDataSession () {
	super();
    }

    //####################################
    //Public Section.
    //####################################

    /**
     * Initialize the DataSession object.
     *
     * @param	obj	an implementation-specific object required to initialize the DataSession
     */
    public void initialize(Object obj){
	initializeObject = obj;
	java.lang.Thread thread = new java.lang.Thread(this);
	thread.start();
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

    public void getFunctionDetail(Function function){};

    public GlobalMapping getGlobalMapping(){
	return globalMapping;}

    public int getNumberOfMappings(){
	return globalMapping.getNumberOfMappings(0);}

    public int getNumberOfUserEvents(){
	return globalMapping.getNumberOfMappings(2);}

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
	    Class c = this.getClass();
	    out = new PrintWriter(new FileWriter(new File(c.getName()+".out")));
	}
	catch(IOException exception){
	    exception.printStackTrace();
	}
    }
    
    public boolean debug(){
	return debug;}

    public NCT getNCT(){
	return nct;}

    //Returns the total number of threads in this trial.
    public int getTotalNumberOfThreads(){
	if(totalNumberOfThreads==-1){
	    for(Enumeration e1 = nct.getNodes().elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    totalNumberOfThreads+=(context.getNumberOfThreads());
		}
	    }
	}
	return totalNumberOfThreads;
    }

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
	    for(Enumeration e1 = (nct.getNodes()).elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		if(node.getNodeID()>maxNCT[0])
		    maxNCT[0]=node.getNodeID();
		for(Enumeration e2 = (node.getContexts()).elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    if(context.getContextID()>maxNCT[1])
			maxNCT[1]=context.getContextID();
		    for(Enumeration e3 = (context.getThreads()).elements(); e3.hasMoreElements() ;){
			Thread thread = (Thread) e3.nextElement();
			if(thread.getThreadID()>maxNCT[2])
			    maxNCT[2]=thread.getThreadID();
		    }
		}
	    }
	    
	}
	return maxNCT;
    }
    
    public void setMeanData(int mappingSelection, int metric){
	//Cycle through the list of global mapping elements.  For each one, sum up
	//the exclusive and the inclusive times respectively, and each total by the
	//number of times we encountered that mapping.
	GlobalMapping globalMapping = this.getGlobalMapping();
	ListIterator l = globalMapping.getMappingIterator(mappingSelection);
	while(l.hasNext()){
	    double exclusiveTotal = 0.0;
	    double inclusiveTotal = 0.0;
	    int numberOfCallsTotal = 0;
	    int numberOfSubroutinesTotal = 0;
	    double userSecPerCallValueTotal = 0;
	    int count = 0;
	    GlobalMappingElement globalMappingElement = (GlobalMappingElement) l.next();
	    int id = globalMappingElement.getGlobalID();
	    for(Enumeration e1 = this.getNCT().getNodes().elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    for(Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements() ;){
			Thread thread = (Thread) e3.nextElement();
			GlobalThreadDataElement globalThreadDataElement = thread.getFunction(id);
			if(globalThreadDataElement != null){
			    exclusiveTotal+=globalThreadDataElement.getExclusiveValue(metric);
			    inclusiveTotal+=globalThreadDataElement.getInclusiveValue(metric);
			    numberOfCallsTotal+=globalThreadDataElement.getNumberOfCalls();
			    numberOfSubroutinesTotal+=globalThreadDataElement.getNumberOfSubRoutines();
			    userSecPerCallValueTotal+=globalThreadDataElement.getUserSecPerCall(metric);
			}
			count++;
		    }
		}
	    }
	    if(count!=0){
		double meanExlusiveValue = exclusiveTotal/count;
		double meanInlusiveValue = inclusiveTotal/count;
		double meanNumberOfCalls = numberOfCallsTotal/count;
		double meanNumberOfSubroutines = numberOfSubroutinesTotal/count;
		double meanUserSecPerCallValue = userSecPerCallValueTotal/count;
		
		globalMappingElement.setMeanExclusiveValue(metric, meanExlusiveValue);
		if(globalMapping.getMaxMeanExclusiveValue(metric) < meanExlusiveValue)
		    globalMapping.setMaxMeanExclusiveValue(metric, meanExlusiveValue);
		
		globalMappingElement.setMeanInclusiveValue(metric, meanInlusiveValue);
		if(globalMapping.getMaxMeanInclusiveValue(metric) < meanInlusiveValue)
		    globalMapping.setMaxMeanInclusiveValue(metric, meanInlusiveValue);

		globalMappingElement.setMeanNumberOfCalls(meanNumberOfCalls);
		if(globalMapping.getMaxMeanNumberOfCalls() < meanNumberOfCalls)
		    globalMapping.setMaxMeanNumberOfCalls(meanNumberOfCalls);

		globalMappingElement.setMeanNumberOfSubRoutines(meanNumberOfSubroutines);
		if(globalMapping.getMaxMeanNumberOfSubRoutines() < meanNumberOfSubroutines)
		    globalMapping.setMaxMeanNumberOfSubRoutines(meanNumberOfSubroutines);

		globalMappingElement.setMeanUserSecPerCall(metric, meanUserSecPerCallValue);
		if(globalMapping.getMaxMeanUserSecPerCall(metric) < meanUserSecPerCallValue)
		    globalMapping.setMaxMeanUserSecPerCall(metric, meanUserSecPerCallValue);
	    }
	}

	//Now set the percent values.
	l = globalMapping.getMappingIterator(mappingSelection);
	double maxMeanInclusiveValue = globalMapping.getMaxMeanInclusiveValue(metric);
	while(l.hasNext()){
	    GlobalMappingElement globalMappingElement = (GlobalMappingElement) l.next();
	    double meanExclusivePercentValue = (globalMappingElement.getMeanExclusiveValue(metric)/maxMeanInclusiveValue)*100.0;
	    double meanInclusivePercentValue = (globalMappingElement.getMeanInclusiveValue(metric)/maxMeanInclusiveValue)*100.0;
	    globalMappingElement.setMeanExclusivePercentValue(metric,meanExclusivePercentValue);
	    globalMappingElement.setMeanInclusivePercentValue(metric,meanInclusivePercentValue);
	    	    
	    globalMapping.setMaxMeanExclusivePercentValue(metric, 100.00);
	    globalMapping.setMaxMeanInclusivePercentValue(metric, 100.00);
	    globalMappingElement.setMeanValuesSet(true);
	}

    }

    public void setMeanDataAllMetrics(int mappingSelection, int numberOfMetrics){
	//Cycle through the list of global mapping elements.  For each one, sum up
	//the exclusive and the inclusive times respectively, and each total by the
	//number of times we encountered that mapping.
	GlobalMapping globalMapping = this.getGlobalMapping();
	ListIterator l = globalMapping.getMappingIterator(mappingSelection);

	//Allocate outside loop, and reset to zero at each iteration.
	//Working on the assumption that this is slightly quicker than
	//re-allocating in each loop iteration. 
	double[] exclusiveTotal = new double[numberOfMetrics];
	double[] inclusiveTotal = new double[numberOfMetrics];
	int numberOfCallsTotal = 0;
	int numberOfSubroutinesTotal = 0;
	double[] userSecPerCallValueTotal = new double[numberOfMetrics];
	
	double[] meanExclusiveValue = new double[numberOfMetrics];
	double[] meanInclusiveValue = new double[numberOfMetrics];
	double[] meanUserSecPerCallValue = new double[numberOfMetrics];
	double[] maxMeanInclusiveValue = new double[numberOfMetrics];

	while(l.hasNext()){

	    //Reset values for this itertion.
	    numberOfCallsTotal = 0;
	    numberOfSubroutinesTotal = 0;
	    for(int i=0;i<numberOfMetrics;i++){
		exclusiveTotal[i] = 0;
		inclusiveTotal[i] = 0;
		userSecPerCallValueTotal[i] = 0;
		meanExclusiveValue[i] = 0;
		meanInclusiveValue[i] = 0;
		meanUserSecPerCallValue[i] = 0;
	    }
	    int count = 0;
	    GlobalMappingElement globalMappingElement = (GlobalMappingElement) l.next();
	    int id = globalMappingElement.getGlobalID();
	    for(Enumeration e1 = this.getNCT().getNodes().elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    for(Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements() ;){
			Thread thread = (Thread) e3.nextElement();
			GlobalThreadDataElement globalThreadDataElement = thread.getFunction(id);
			if(globalThreadDataElement != null){
			    for(int i=0;i<numberOfMetrics;i++){
				exclusiveTotal[i]+=globalThreadDataElement.getExclusiveValue(i);
				inclusiveTotal[i]+=globalThreadDataElement.getInclusiveValue(i);
				if(i==0){
				    numberOfCallsTotal+=globalThreadDataElement.getNumberOfCalls();
				    numberOfSubroutinesTotal+=globalThreadDataElement.getNumberOfSubRoutines();
				}
				userSecPerCallValueTotal[i]+=globalThreadDataElement.getUserSecPerCall(i);
			    }
			}
			count++;
		    }
		}
	    }
	    if(count!=0){
		//Since we only need to do the numberOfCalls and numberOfSubroutines for
		//the first metric, do it first (outside the loop).
		double meanNumberOfCalls = numberOfCallsTotal/count;
		double meanNumberOfSubroutines = numberOfSubroutinesTotal/count;
		
		globalMappingElement.setMeanNumberOfCalls(meanNumberOfCalls);
		if(globalMapping.getMaxMeanNumberOfCalls() < meanNumberOfCalls)
		    globalMapping.setMaxMeanNumberOfCalls(meanNumberOfCalls);

		globalMappingElement.setMeanNumberOfSubRoutines(meanNumberOfSubroutines);
		if(globalMapping.getMaxMeanNumberOfSubRoutines() < meanNumberOfSubroutines)
		    globalMapping.setMaxMeanNumberOfSubRoutines(meanNumberOfSubroutines);
		
		for(int i=0;i<numberOfMetrics;i++){
		    meanExclusiveValue[i] = exclusiveTotal[i]/count;
		    meanInclusiveValue[i] = inclusiveTotal[i]/count;
		    meanUserSecPerCallValue[i] = userSecPerCallValueTotal[i]/count;

		    globalMappingElement.setMeanExclusiveValue(i, meanExclusiveValue[i]);
		    if(globalMapping.getMaxMeanExclusiveValue(i) < meanExclusiveValue[i])
			globalMapping.setMaxMeanExclusiveValue(i, meanExclusiveValue[i]);
		    
		    globalMappingElement.setMeanInclusiveValue(i, meanInclusiveValue[i]);
		    if(globalMapping.getMaxMeanInclusiveValue(i) < meanInclusiveValue[i])
			globalMapping.setMaxMeanInclusiveValue(i, meanInclusiveValue[i]);

		    globalMappingElement.setMeanUserSecPerCall(i, meanUserSecPerCallValue[i]);
		    if(globalMapping.getMaxMeanUserSecPerCall(i) < meanUserSecPerCallValue[i])
			globalMapping.setMaxMeanUserSecPerCall(i, meanUserSecPerCallValue[i]);
		}
	    }
	}

	//Now set the percent values.
	for(int i=0;i<numberOfMetrics;i++){
	    maxMeanInclusiveValue[i] = globalMapping.getMaxMeanInclusiveValue(i);
	}
	l = globalMapping.getMappingIterator(mappingSelection);
	while(l.hasNext()){
	    GlobalMappingElement globalMappingElement = (GlobalMappingElement) l.next();
	    for(int i=0;i<numberOfMetrics;i++){
		double meanExclusivePercentValue = (globalMappingElement.getMeanExclusiveValue(i)/maxMeanInclusiveValue[i])*100.0;
		double meanInclusivePercentValue = (globalMappingElement.getMeanInclusiveValue(i)/maxMeanInclusiveValue[i])*100.0;
		globalMappingElement.setMeanExclusivePercentValue(i,meanExclusivePercentValue);
		globalMappingElement.setMeanInclusivePercentValue(i,meanInclusivePercentValue);

		globalMapping.setMaxMeanExclusivePercentValue(i, 100.00);
		globalMapping.setMaxMeanInclusivePercentValue(i, 100.00);
	    }
	    globalMappingElement.setMeanValuesSet(true);
	}
    }
    //######
    //End - Set mean values functions.
    //######

    //######
    //Methods that manage the ParaProfObservers.
    //######
    public void addObserver(ParaProfObserver observer){
	observers.add(observer);}

    public void removeObserver(ParaProfObserver observer){
	observers.remove(observer);}

    public void notifyObservers(){
	for(Enumeration e = observers.elements(); e.hasMoreElements() ;)
	    ((ParaProfObserver) e.nextElement()).update(this);
    }
    //######
    //End - Methods that manage the ParaProfObservers.
    //######

    //####################################
    //End - Public Section.
    //####################################

    //####################################
    //Protected Section.
    //####################################
    protected void setMetrics(Vector metrics){
	this.metrics = metrics;}

    protected void addMetric(String metricName){
	metrics.add(metricName);}

    protected int getNumberOfMetrics(){
	return metrics.size();}

    protected String getMetricName(int metricID){
	return (String) metrics.elementAt(metricID);}

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
	out.println(s);}
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

    private GlobalMapping globalMapping = new GlobalMapping();
    private NCT nct = new NCT();
    private int[] maxNCT = null;

    private boolean debug = false;
    //When in debugging mode, this class can print a lot of data.
    //Initialized in this.setDebug(...).
    private PrintWriter out = null;

    private Vector observers = new Vector();
    //######
    //End - Private Section.
    //######

    //####################################
    //End - Instance data.
    //####################################


}
