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

	public DataSession () {
		super();
	}

	// initialization work
	abstract public void open() ;

	// termination work
	abstract public void close() ;

	// returns Vector of Application objects
	abstract public ListIterator getAppList() ;

	// returns Vector of Experiment objects
	abstract public ListIterator getExpList() ;

	// returns Vector of Trial objects
	abstract public ListIterator getTrialList() ;

	// set the Application for this session
	public void setApplication(Application application) {
		this.application = application;
	}

	abstract public Application setApplication(int id) ;

	abstract public Application setApplication(String name) ;

	// set the Experiment for this session
	public void setExperiment(Experiment experiment) {
		this.experiment = experiment;
	}

	// is this necessary? 
	abstract public Experiment setExperiment(int id) ;

	// set the Trial for this session
	public void setTrial(Trial trial) {
		trials = new Vector();
		trials.addElement(trial);
	}

	public void setTrial(Vector trials) {
		this.trials = trials;
	}

	// is this necessary? 
	abstract public Trial setTrial(int id) ;

	abstract public int getNumberOfNodes() ;
	abstract public int getNumberOfContexts() ;
	abstract public int getNumberOfThreads() ;

	// node id, or PERFDB_NONE
	public void setNode(int node) {
		Integer iNode = new Integer(node);
		this.nodes = new Vector();
		this.nodes.addElement(iNode);
	}

	public void setNode(Vector nodes) {
		this.nodes = nodes;
	}

	// context id, or PERFDB_NONE
	public void setContext(int context) {
		Integer iContext = new Integer(context);
		this.contexts = new Vector();
		this.contexts.addElement(iContext);
	}

	public void setContext(Vector contexts) {
		this.contexts = contexts;
	}

	// thread id, or PERFDB_NONE
	public void setThread(int thread) {
		Integer iThread = new Integer(thread);
		this.threads = new Vector();
		this.threads.addElement(iThread);
	}

	public void setThread(Vector threads) {
		this.threads = threads;
	}

	// returns a Vector of Functions
	abstract public ListIterator getFunctions() ;
	abstract public ListIterator getUserEvents() ;

	abstract public Function setFunction(int id) ;

	public void setFunction(Function function) {
		this.functions = new Vector();
		this.functions.addElement(function);
	}
	
	public void setFunction(Vector functions) {
		this.functions = functions;
	}
	
	abstract public ListIterator getFunctionData();
	abstract public ListIterator getUserEventData();
};

