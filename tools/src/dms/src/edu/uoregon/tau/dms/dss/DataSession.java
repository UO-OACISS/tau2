package edu.uoregon.tau.dms.dss;

import java.util.*;

/**
 * This is the top level class for the API.
 *
 * <P>CVS $Id: DataSession.java,v 1.1 2004/05/05 17:43:30 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 */
public abstract class DataSession {
	protected Application application = null;
	protected Experiment experiment = null;
	protected Trial trial = null;
	protected Vector nodes = null;
	protected Vector contexts = null;
	protected Vector threads = null;
	protected Vector intervalEvents = null;
	protected Vector metrics = null;
	protected Vector intervalEventData = null;
	protected Vector atomicEvents = null;
	protected Vector atomicEventData = null;

	public static int PERFDMF_NONE = -1;

    //######
    //Object structure representation of data mirrored on ParaProf's usage.
    //######
    private GlobalMapping globalMapping = null;
    private NCT nct = null;
    //######
    //End - Object structure representation of data mirrored on ParaProf's usage.
    //######

	public DataSession () {
		super();
	}

/**
 * Initialize the DataSession object.
 *
 * @param	obj	an implementation-specific object required to initialize the DataSession
 */
	abstract public void initialize(Object obj) ;  // formerly "open"

/**
 * Terminate the DataSession object.
 *
 */
	abstract public void terminate() ;   // formerly "close"

/**
 * Returns a ListIterator of Application objects.
 *
 * @return	DataSessionIterator object of all Applications.
 * @see	DataSessionIterator
 * @see	Application
 */
	abstract public ListIterator getApplicationList() ;
	
/**
 * Returns the Application object.
 *
 * @return	Application object, if one is set.
 * @see	Application
 */
	public Application getApplication() {
		return this.application;
	}

/**
 * Returns a ListIterator of Experiment objects
 *
 * @return	DataSessionIterator object of all Experiments.  If there is an Application saved in the DataSession, then only the Experiments for that Application are returned.
 * @see	DataSessionIterator
 * @see	Experiment
 * @see	DataSession#setApplication
 */
	abstract public ListIterator getExperimentList() ;

/**
 * Returns the Experiment object.
 *
 * @return	Experiment object, if one is set.
 * @see	Experiment
 */
	public Experiment getExperiment() {
		return this.experiment;
	}

/**
 * Returns a ListIterator of Trial objects
 *
 * @return	DataSessionIterator object of all Trials.  If there is an Application and/or Experiment saved in the DataSession, then only the Trials for that Application and/or Experiment are returned.
 * @see	DataSessionIterator
 * @see	Trial
 * @see	DataSession#setApplication
 * @see	DataSession#setExperiment
 */
	abstract public ListIterator getTrialList() ;

/**
 * Returns the Trial object.
 *
 * @return	Trial object, if one is set.
 * @see	Trial
 */
	public Trial getTrial() {
		return this.trial;
	}

/**
 * Set the Application for this DataSession.  The DataSession object will maintain a reference to the Application object.  To clear this reference, call setApplication(Application) with a null reference.
 *
 * @param	application Application object to be saved.
 * @see	Application
 */
	public void setApplication(Application application) {
		this.application = application;
	}

/**
 * Set the Application for this DataSession.  The DataSession object will maintain a reference to the Application referenced by the id.  To clear this reference, call setApplication(Application) with a null reference.
 *
 * @param	id unique id of the Application object to be saved.
 * @see	Application
 * @see	DataSession#setApplication(Application)
 */
	abstract public Application setApplication(int id) ;

/**
 * Set the Application for this DataSession.  The DataSession object will maintain a reference to the Application referenced by the name and/or version.  To clear this reference, call setApplication(Application) with a null reference.
 *
 * @param	name name of the Application object to be saved.
 * @param	version version of the Application object to be saved.
 * @see	Application
 * @see	DataSession#setApplication(Application)
 */
	abstract public Application setApplication(String name, String version) ;

/**
 * Set the Experiment for this DataSession.  The DataSession object will maintain a reference to the Experiment object.  To clear this reference, call setExperiment(Experiment) with a null reference.
 *
 * @param	experiment Experiment object to be saved.
 * @see	Experiment
 */
	public void setExperiment(Experiment experiment) {
		this.experiment = experiment;
		/* don't know if we want to do this...
		if (this.application == null && this.experiment != null) {
			setApplication(experiment.getApplicationID());
		} */
	}

/**
 * Set the Experiment for this DataSession.  The DataSession object will maintain a reference to the Experiment referenced by the id.  To clear this reference, call setExperiment(Experiment) with a null reference.
 *
 * @param	id unique id of the Experiment object to be saved.
 * @see	Experiment
 * @see	DataSession#setExperiment(Experiment)
 */
	abstract public Experiment setExperiment(int id) ;

/**
 * Set the Trial for this DataSession.  The DataSession object will maintain a reference to the Trial object.  To clear this reference, call setTrial(Trial) with a null reference.
 *
 * @param	trial Trial object to be saved.
 * @see	Trial
 */
	public void setTrial(Trial trial) {
		this.trial = trial;
		/* don't know if we want to do this
		if (this.experiment == null && this.trial != null) {
			setExperiment(trial.getExperimentID());
		}*/
	}

/**
 * Set the Trial for this DataSession.  The DataSession object will maintain a reference to the Trial referenced by the id.  To clear this reference, call setTrial(Trial) with a null reference.
 *
 * @param	id unique id of the Trial object to be saved.
 * @see	Trial
 * @see	DataSession#setTrial(Trial)
 */
	abstract public Trial setTrial(int id) ;

/**
 * Set the node for this DataSession.  The DataSession object will maintain the value of the node identified by the id.  To clear this value, call setNode(int) with DataSession.PERFDMF_NONE.
 *
 * @param	node value of the node to be saved.
 * @see	DataSession#PERFDMF_NONE
 */
	public void setNode(int node) {
		if (node == PERFDMF_NONE) {
			this.nodes = null;
		} else {
			Integer iNode = new Integer(node);
			this.nodes = new Vector();
			this.nodes.addElement(iNode);
		}
	}

/**
 * Set a Vector of node values for this DataSession.  The DataSession object will maintain a reference to the Vector of node values.  To clear this reference, call setNode(int) with DataSession.PERFDMF_NONE.
 *
 * @param	nodes Vector of node values to be saved.
 * @see	DataSession#PERFDMF_NONE
 */
	public void setNode(Vector nodes) {
		this.nodes = nodes;
	}

/**
 * Set the context for this DataSession.  The DataSession object will maintain the value of the context identified by the id.  To clear this value, call setContext(int) with DataSession.PERFDMF_NONE.
 *
 * @param	context value of the context to be saved.
 * @see	DataSession#PERFDMF_NONE
 */
	public void setContext(int context) {
		if (context == PERFDMF_NONE) {
			this.contexts = null;
		} else {
			Integer iContext = new Integer(context);
			this.contexts = new Vector();
			this.contexts.addElement(iContext);
		}
	}

/**
 * Set a Vector of context values for this DataSession.  The DataSession object will maintain a reference to the Vector of context values.  To clear this reference, call setContext(int) with DataSession.PERFDMF_NONE.
 *
 * @param	contexts Vector of context values to be saved.
 * @see	DataSession#PERFDMF_NONE
 */
	public void setContext(Vector contexts) {
		this.contexts = contexts;
	}

/**
 * Set the thread for this DataSession.  The DataSession object will maintain the value of the thread identified by the id.  To clear this value, call setThread(int) with DataSession.PERFDMF_NONE.
 *
 * @param	thread value of the thread to be saved.
 * @see	DataSession#PERFDMF_NONE
 */
	public void setThread(int thread) {
		if (thread == PERFDMF_NONE) {
			this.threads = null;
		} else {
			Integer iThread = new Integer(thread);
			this.threads = new Vector();
			this.threads.addElement(iThread);
		}
	}

/**
 * Set a Vector of thread values for this DataSession.  The DataSession object will maintain a reference to the Vector of thread values.  To clear this reference, call setThread(int) with DataSession.PERFDMF_NONE.
 *
 * @param	threads Vector of thread values to be saved.
 * @see	DataSession#PERFDMF_NONE
 */
	public void setThread(Vector threads) {
		this.threads = threads;
	}

/**
 * Set the metric for this DataSession.  The DataSession object will maintain the value of the metric identified by the String.  To clear this value, call setMetric(Metric) with null.
 *
 * @param	metric value of the metric to be saved.
 */
	public void setMetric(Metric metric) {
		if (metric == null) {
			this.metrics = null;
		} else {
			this.metrics = new Vector();
			this.metrics.addElement(metric);
		}
	}

/**
 * Set a Vector of metrics for this DataSession.  The DataSession object will maintain a reference to the Vector of metric values.  To clear this reference, call setMetric(Metric) with null.
 *
 * @param	metrics Vector of metric values to be saved.
 */
	public void setMetrics(Vector metrics) {
		this.metrics = metrics;
	}

/**
 * Adds a metric to this data sessions metrics.  The DataSession object will maintain a reference to the Vector of metric values.  To clear this reference, call setMetric(String) with null.
 *
 *@param	metric name of metric.
 *
 * @return	Metric the newly added metric.
 */
    public int addMetric(Metric metric) {
	if (this.metrics == null) {
	    if (this.trial != null) {
		this.metrics = this.trial.getMetrics();
	    }
	}
	
	//Try getting the matric.
	if(this.metrics!=null){
	    metric.setID(this.getNumberOfMetrics());
	    metrics.add(metric);
	    return metric.getID();
	}
	else
	    return -1;
    }

/**
 * Get a Vector of metric values for this DataSession.  The DataSession object will maintain a reference to the Vector of metric values.  To clear this reference, call setMetric(String) with null.
 *
 * @return	Vector of metric values
 */
	public Vector getMetrics() {
		if (this.metrics == null) {
			if (this.trial != null) {
				this.metrics = this.trial.getMetrics();
			}
		}
		return this.metrics;
	}


/**
 * Get the metric with the given id..  The DataSession object will maintain a reference to the Vector of metric values.  To clear this reference, call setMetric(String) with null.
 *
 * @param	metricID metric id.
 *
 * @return	Metric with given id.
 */
    public Metric getMetric(int metricID) {
	if (this.metrics == null) {
	    if (this.trial != null) {
		this.metrics = this.trial.getMetrics();
	    }
	}
	
	//Try getting the matric.
	if((this.metrics!=null) && (metricID<this.metrics.size()))
	    return (Metric) this.metrics.elementAt(metricID);
	else
	    return null;
    }


/**
 * Get the metric name corresponding to the given id.  The DataSession object will maintain a reference to the Vector of metric values.  To clear this reference, call setMetric(String) with null.
 *
 * @param	metricID metric id.
 *
 * @return	The metric name as a String.
 */
	public String getMetricName(int metricID) {
		if (this.metrics == null) {
			if (this.trial != null) {
				this.metrics = this.trial.getMetrics();
			}
		}
		
		//Try getting the matric name.
		if((this.metrics!=null) && (metricID<this.metrics.size()))
		    return ((Metric)this.metrics.elementAt(metricID)).getName();
		else
		    return null;
	}

/**
 * Get the metric id corresponding to the given string.  The DataSession object will maintain a reference to the Vector of metric values.  To clear this reference, call setMetric(String) with null.
 *
 *@param	string the metric name.
 *
 * @return	The metric name as a String.
 */
    public int getMetricID(String string){
	if (this.metrics == null) {
	    if (this.trial != null) {
		this.metrics = this.trial.getMetrics();
	    }
	}
	
	//Try getting the matric id.
	if(this.metrics!=null){
	    for(Enumeration e = metrics.elements(); e.hasMoreElements() ;){
		Metric metric = (Metric) e.nextElement();
		if((metric.getName()).equals(string))
		    return metric.getID();
	    }
	}
	return -1;
    }

/**
 * Get the number of metrics.  The DataSession object will maintain a reference to the Vector of metric values.  To clear this reference, call setMetric(String) with null.
 *
 * @return	Returns the number of metrics as an int.
 */
	public int getNumberOfMetrics() {
		if (this.metrics == null) {
			if (this.trial != null) {
				this.metrics = this.trial.getMetrics();
			}
		}
		
		//Try getting the matric name.
		if(this.metrics!=null)
		    return metrics.size();
		else
		    return -1;
	}

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
	abstract public ListIterator getIntervalEvents() ;

/**
 * Set the IntervalEvent for this DataSession.  The DataSession object will maintain a reference to the IntervalEvent object.  To clear this reference, call setIntervalEvent(IntervalEvent) with a null reference.
 *
 * @param	intervalEvent IntervalEvent object to be saved.
 * @see	IntervalEvent
 * @see	DataSession#setIntervalEvent(IntervalEvent)
 */
	public void setIntervalEvent(IntervalEvent intervalEvent) {
		if (intervalEvent == null)
			this.intervalEvents = null;
		else {
			this.intervalEvents = new Vector();
			this.intervalEvents.addElement(intervalEvent);
		}
	}
	
/**
 * Set the IntervalEvent for this DataSession.  The DataSession object will maintain a reference to the IntervalEvent referenced by the id.  To clear this reference, call setIntervalEvent(IntervalEvent) with a null reference.
 *
 * @param	id unique id of the IntervalEvent object to be saved.
 * @see	IntervalEvent
 * @see	DataSession#setIntervalEvent(IntervalEvent)
 */
	abstract public IntervalEvent setIntervalEvent(int id) ;

/**
 * Set a Vector of IntervalEvent objects for this DataSession.  The DataSession object will maintain a reference to a Vector of IntervalEvent objects.  To clear this reference, call setIntervalEvent(IntervalEvent) with a null reference.
 *
 * @param	intervalEvents Vector of IntervalEvent objects to be saved.
 * @see	IntervalEvent
 * @see	DataSession#setIntervalEvent(IntervalEvent)
 */
	public void setIntervalEvent(Vector intervalEvents) {
		this.intervalEvents = intervalEvents;
	}
	
/**
 * Returns the IntervalEvent identified by the unique intervalEvent id.
 *
 * @param	intervalEventID unique id of the IntervalEvent object.
 * @return	IntervalEvent object identified by the unique intervalEvent id.
 * @see	IntervalEvent
 */
	abstract public IntervalEvent getIntervalEvent(int intervalEventID) ;
	
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
	abstract public ListIterator getAtomicEvents() ;

/**
 * Set the AtomicEvent for this DataSession.  The DataSession object will maintain a reference to the AtomicEvent object.  To clear this reference, call setAtomicEvent(AtomicEvent) with a null reference.
 *
 * @param	atomicEvent AtomicEvent object to be saved.
 * @see	AtomicEvent
 * @see	DataSession#setAtomicEvent(AtomicEvent)
 */
	public void setAtomicEvent(AtomicEvent atomicEvent) {
		if (atomicEvent == null) 
			this.atomicEvents = null;
		else {
			this.atomicEvents = new Vector();
			this.atomicEvents.addElement(atomicEvent);
		}
	}
	
/**
 * Set the AtomicEvent for this DataSession.  The DataSession object will maintain a reference to the AtomicEvent referenced by the id.  To clear this reference, call setAtomicEvent(AtomicEvent) with a null reference.
 *
 * @param	id unique id of the AtomicEvent object to be saved.
 * @see	AtomicEvent
 * @see	DataSession#setAtomicEvent(AtomicEvent)
 */
	abstract public AtomicEvent setAtomicEvent(int id) ;

/**
 * Set a Vector of AtomicEvent objects for this DataSession.  The DataSession object will maintain a reference to a Vector of AtomicEvent objects.  To clear this reference, call setAtomicEvent(AtomicEvent) with a null reference.
 *
 * @param	atomicEvents Vector of AtomicEvent objects to be saved.
 * @see	AtomicEvent
 * @see	DataSession#setAtomicEvent(AtomicEvent)
 */
	public void setAtomicEvent(Vector atomicEvents) {
		this.atomicEvents = atomicEvents;
	}
	
/**
 * Returns the AtomicEvent identified by the unique atomic event id.
 *
 * @param	atomicEventID unique id of the AtomicEvent object to be saved.
 * @return	AtomicEvent object identified by the unique atomic event id.
 * @see	AtomicEvent
 */
	abstract public AtomicEvent getAtomicEvent(int atomicEventID) ;
	
/**
 * Returns the IntervalEventData for this DataSession
 *
 * @return	DataSessionIterator of IntervalEventData objects.  If there is an Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or IntervalEvent(s) saved in the DataSession, then only the IntervalEvents for that Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or IntervalEvent(s) are returned.
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
	abstract public ListIterator getIntervalEventData();

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
	abstract public ListIterator getAtomicEventData();

/**
 * Gets the interval record detail for this intervalEvent.
 *
 * @param intervalEvent
 */
	abstract public void getIntervalEventDetail(IntervalEvent intervalEvent) ;

/**
 * Saves the trial.
 *
 * @return database index ID of the saved trial
 */
	abstract public int saveTrial() ;

/**
 * Saves the Trial.
 *
 * @param  trial
 * @return ID of the saved trial
 */
	abstract public int saveTrial(Trial trial) ;

/**
 * Saves the IntervalEvent.
 *
 * @param intervalEvent
 * @return ID of the saved intervalEvent
 */
	abstract public int saveIntervalEvent(IntervalEvent intervalEvent, int newTrialID, Hashtable newMetHash) ;

/**
 * Saves the IntervalLocationProfile.
 *
 * @param intervalEventData
 */
	abstract public void saveIntervalEventData(IntervalLocationProfile intervalEventData, int newIntervalEventID, Hashtable newMetHash) ;

/**
 * Saves the AtomicEvent object.
 *
 * @param atomicEvent
 * @return ID of the saved atomic event
 */
	abstract public int saveAtomicEvent(AtomicEvent atomicEvent, int newTrialID) ;

/**
 * Saves the atomicEventData object.
 *
 * @param atomicEventData
 */
	abstract public void saveAtomicEventData(AtomicLocationProfile atomicEventData, int newAtomicEventID) ;

    //######
    //IntervalEvents interfacing to object structure representation of data mirrored on ParaProf's usage.
    //######
/**
  * Sets this data session's NCT object.
  *
  * @param nct NCT object.
  */
    public void setNCT(NCT nct){
	this.nct = nct;}

 /**
  * Gets this data session's NCT object.
  *
  * @return An NCT object.
  */
    public NCT getNCT(){
	return nct;}

/**
  * Sets this data session's GlobalMapping object.
  *
  * @param globalMapping GlobalMapping object.
  */
    public void setGlobalMapping(GlobalMapping globalMapping){
	this.globalMapping = globalMapping;}

/**
  * Gets this data session's GlobalMapping object.
  *
  * @return A GlobalMapping object.
  */
    public GlobalMapping getGlobalMapping(){
	return globalMapping;}
    //######
    //End - IntervalEvents interfacing to object structure representation of data mirrored on ParaProf's usage.
    //######

    /**
     * Resets the DataSession object.
     *
     */
    public void reset() {
	application = null;
	experiment = null;
	trial = null;
	nodes = null;
	contexts = null;
	threads = null;
	intervalEvents = null;
	metrics = null;
	intervalEventData = null;
	atomicEvents = null;
	atomicEventData = null;
    }

/**
  * Deletes the trial
  *
  * @param trialID the ID of the trial to delete
  */

  public void deleteTrial (int trialID) {};

/**
  * Deletes the application
  *
  * @param trialID the ID of the application to delete
  */

  public void deleteApplication (int applicationID) {};

/**
  * Deletes the experiment
  *
  * @param trialID the ID of the experiment to delete
  */

  public void deleteExperiment (int experimentID) {};

/**
  * Saves the application
  *
  * @param trialID the ID of the application to save
  */

  public void saveApplication (Application application) {};

/**
  * Saves the experiment
  *
  * @param trialID the ID of the experiment to save
  */

  public void saveExperiment (Experiment experiment) {};

};
