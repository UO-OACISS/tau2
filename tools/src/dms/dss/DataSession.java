package dms.dss;

import java.util.*;
import dms.dss.*;

public abstract class DataSession {
	protected Application application = null;
	protected Experiment experiment = null;
	protected Vector trials = null;
	protected Vector nodes = null;
	protected Vector contexts = null;
	protected Vector threads = null;
	protected Vector functions = null;
	protected Vector functionData = null;
	protected Vector userEvents = null;
	protected Vector userEventData = null;

	public static int PERFDB_NONE = -1;

	public DataSession () {
		super();
	}

	// initialization work
	abstract public void initialize(Object obj) ;  // formerly "open"

	// termination work
	abstract public void terminate() ;   // formerly "close"

	// returns Vector of Application objects
	abstract public ListIterator getApplicationList() ;

	// returns Vector of Experiment objects
	abstract public ListIterator getExperimentList() ;

	// returns Vector of Trial objects
	abstract public ListIterator getTrialList() ;

	// set the Application for this session
	// Usage Note: use null to clear the application
	public void setApplication(Application application) {
		this.application = application;
	}

	abstract public Application setApplication(int id) ;
	abstract public Application setApplication(String name, String version) ;

	// set the Experiment for this session
	// Usage Note: use null to clear the experiment
	public void setExperiment(Experiment experiment) {
		this.experiment = experiment;
	}

	// set the experiment for this session
	abstract public Experiment setExperiment(int id) ;

	// set the Trial for this session
	public void setTrial(Trial trial) {
		trials = new Vector();
		trials.addElement(trial);
	}

	public void setTrial(Vector trials) {
		this.trials = trials;
	}

	// set the trial for this session
	abstract public Trial setTrial(int id) ;

	// node id, or PERFDB_NONE
	public void setNode(int node) {
		if (node == PERFDB_NONE) {
			this.nodes = null;
		} else {
			Integer iNode = new Integer(node);
			this.nodes = new Vector();
			this.nodes.addElement(iNode);
		}
	}

	public void setNode(Vector nodes) {
		this.nodes = nodes;
	}

	// context id, or PERFDB_NONE
	public void setContext(int context) {
		if (context == PERFDB_NONE) {
			this.contexts = null;
		} else {
			Integer iContext = new Integer(context);
			this.contexts = new Vector();
			this.contexts.addElement(iContext);
		}
	}

	public void setContext(Vector contexts) {
		this.contexts = contexts;
	}

	// thread id, or PERFDB_NONE
	public void setThread(int thread) {
		if (thread == PERFDB_NONE) {
			this.threads = null;
		} else {
			Integer iThread = new Integer(thread);
			this.threads = new Vector();
			this.threads.addElement(iThread);
		}
	}

	public void setThread(Vector threads) {
		this.threads = threads;
	}

	// returns a Vector of Functions
	abstract public ListIterator getFunctions() ;

	// sets the current function
	abstract public Function setFunction(int id) ;

	// sets the current function
	public void setFunction(Function function) {
		this.functions = new Vector();
		this.functions.addElement(function);
	}
	
	// sets the current functions
	public void setFunction(Vector functions) {
		this.functions = functions;
	}
	
	// gets the function referenced by the id
	abstract public Function getFunction(int functionID) ;
	
	// returns a Vector of UserEvents
	abstract public ListIterator getUserEvents() ;

	// sets the current user event
	abstract public UserEvent setUserEvent(int id) ;

	// sets the current user event
	public void setUserEvent(UserEvent userEvent) {
		this.userEvents = new Vector();
		this.userEvents.addElement(userEvent);
	}
	
	// sets the current user events
	public void setUserEvent(Vector userEvents) {
		this.userEvents = userEvents;
	}
	
	// get the user event referenced by the id
	abstract public UserEvent getUserEvent(int userEventID) ;
	
	// get the function data for the current function(s)
	abstract public ListIterator getFunctionData();

	// get the user event data for the current user event(s)
	abstract public ListIterator getUserEventData();
};

