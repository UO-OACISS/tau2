package dms.dss;

import java.util.*;

/**
 * This is the top level class for the API.
 *
 * <P>CVS $Id: DataSession.java,v 1.1 2004/03/27 01:02:54 khuck Exp $</P>
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
	protected Vector functions = null;
	protected Vector metrics = null;
	protected Vector functionData = null;
	protected Vector userEvents = null;
	protected Vector userEventData = null;

	public static int PERFDB_NONE = -1;

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
 * Set the node for this DataSession.  The DataSession object will maintain the value of the node identified by the id.  To clear this value, call setNode(int) with DataSession.PERFDB_NONE.
 *
 * @param	node value of the node to be saved.
 * @see	DataSession#PERFDB_NONE
 */
	public void setNode(int node) {
		if (node == PERFDB_NONE) {
			this.nodes = null;
		} else {
			Integer iNode = new Integer(node);
			this.nodes = new Vector();
			this.nodes.addElement(iNode);
		}
	}

/**
 * Set a Vector of node values for this DataSession.  The DataSession object will maintain a reference to the Vector of node values.  To clear this reference, call setNode(int) with DataSession.PERFDB_NONE.
 *
 * @param	nodes Vector of node values to be saved.
 * @see	DataSession#PERFDB_NONE
 */
	public void setNode(Vector nodes) {
		this.nodes = nodes;
	}

/**
 * Set the context for this DataSession.  The DataSession object will maintain the value of the context identified by the id.  To clear this value, call setContext(int) with DataSession.PERFDB_NONE.
 *
 * @param	context value of the context to be saved.
 * @see	DataSession#PERFDB_NONE
 */
	public void setContext(int context) {
		if (context == PERFDB_NONE) {
			this.contexts = null;
		} else {
			Integer iContext = new Integer(context);
			this.contexts = new Vector();
			this.contexts.addElement(iContext);
		}
	}

/**
 * Set a Vector of context values for this DataSession.  The DataSession object will maintain a reference to the Vector of context values.  To clear this reference, call setContext(int) with DataSession.PERFDB_NONE.
 *
 * @param	contexts Vector of context values to be saved.
 * @see	DataSession#PERFDB_NONE
 */
	public void setContext(Vector contexts) {
		this.contexts = contexts;
	}

/**
 * Set the thread for this DataSession.  The DataSession object will maintain the value of the thread identified by the id.  To clear this value, call setThread(int) with DataSession.PERFDB_NONE.
 *
 * @param	thread value of the thread to be saved.
 * @see	DataSession#PERFDB_NONE
 */
	public void setThread(int thread) {
		if (thread == PERFDB_NONE) {
			this.threads = null;
		} else {
			Integer iThread = new Integer(thread);
			this.threads = new Vector();
			this.threads.addElement(iThread);
		}
	}

/**
 * Set a Vector of thread values for this DataSession.  The DataSession object will maintain a reference to the Vector of thread values.  To clear this reference, call setThread(int) with DataSession.PERFDB_NONE.
 *
 * @param	threads Vector of thread values to be saved.
 * @see	DataSession#PERFDB_NONE
 */
	public void setThread(Vector threads) {
		this.threads = threads;
	}

/**
 * Set the metric for this DataSession.  The DataSession object will maintain the value of the metric identified by the String.  To clear this value, call setMetric(String) with null.
 *
 * @param	name value of the metric to be saved.
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
 * Set a Vector of metric values for this DataSession.  The DataSession object will maintain a reference to the Vector of metric values.  To clear this reference, call setMetric(String) with null.
 *
 * @param	names Vector of metric values to be saved.
 */
	public void setMetric(Vector names) {
		this.metrics = names;
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
	abstract public ListIterator getFunctions() ;

/**
 * Set the Function for this DataSession.  The DataSession object will maintain a reference to the Function object.  To clear this reference, call setFunction(Function) with a null reference.
 *
 * @param	function Function object to be saved.
 * @see	Function
 * @see	DataSession#setFunction(Function)
 */
	public void setFunction(Function function) {
		if (function == null)
			this.functions = null;
		else {
			this.functions = new Vector();
			this.functions.addElement(function);
		}
	}
	
/**
 * Set the Function for this DataSession.  The DataSession object will maintain a reference to the Function referenced by the id.  To clear this reference, call setFunction(Function) with a null reference.
 *
 * @param	id unique id of the Function object to be saved.
 * @see	Function
 * @see	DataSession#setFunction(Function)
 */
	abstract public Function setFunction(int id) ;

/**
 * Set a Vector of Function objects for this DataSession.  The DataSession object will maintain a reference to a Vector of Function objects.  To clear this reference, call setFunction(Function) with a null reference.
 *
 * @param	functions Vector of Function objects to be saved.
 * @see	Function
 * @see	DataSession#setFunction(Function)
 */
	public void setFunction(Vector functions) {
		this.functions = functions;
	}
	
/**
 * Returns the Function identified by the unique function id.
 *
 * @param	functionID unique id of the Function object.
 * @return	Function object identified by the unique function id.
 * @see	Function
 */
	abstract public Function getFunction(int functionID) ;
	
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
	abstract public ListIterator getUserEvents() ;

/**
 * Set the UserEvent for this DataSession.  The DataSession object will maintain a reference to the UserEvent object.  To clear this reference, call setUserEvent(UserEvent) with a null reference.
 *
 * @param	userEvent UserEvent object to be saved.
 * @see	UserEvent
 * @see	DataSession#setUserEvent(UserEvent)
 */
	public void setUserEvent(UserEvent userEvent) {
		if (userEvent == null) 
			this.userEvents = null;
		else {
			this.userEvents = new Vector();
			this.userEvents.addElement(userEvent);
		}
	}
	
/**
 * Set the UserEvent for this DataSession.  The DataSession object will maintain a reference to the UserEvent referenced by the id.  To clear this reference, call setUserEvent(UserEvent) with a null reference.
 *
 * @param	id unique id of the UserEvent object to be saved.
 * @see	UserEvent
 * @see	DataSession#setUserEvent(UserEvent)
 */
	abstract public UserEvent setUserEvent(int id) ;

/**
 * Set a Vector of UserEvent objects for this DataSession.  The DataSession object will maintain a reference to a Vector of UserEvent objects.  To clear this reference, call setUserEvent(UserEvent) with a null reference.
 *
 * @param	userEvents Vector of UserEvent objects to be saved.
 * @see	UserEvent
 * @see	DataSession#setUserEvent(UserEvent)
 */
	public void setUserEvent(Vector userEvents) {
		this.userEvents = userEvents;
	}
	
/**
 * Returns the UserEvent identified by the unique user event id.
 *
 * @param	userEventID unique id of the UserEvent object to be saved.
 * @return	UserEvent object identified by the unique user event id.
 * @see	UserEvent
 */
	abstract public UserEvent getUserEvent(int userEventID) ;
	
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
	abstract public ListIterator getFunctionData();

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
	abstract public ListIterator getUserEventData();

	abstract public void getFunctionDetail(Function function) ;

/**
 * Saves the trial.  If the session is a PerfDBSession, the trial is saved in the database.  Otherwise, it is saved to an XML file.
 *
 */

	public void saveTrial() {
		System.out.println("Nothing saved.  Please override the DataSession::saveTrial() method.");
	}

};

