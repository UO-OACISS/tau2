package dms.dss;

import dms.perfdb.*;
import java.util.*;
import java.sql.*;
import java.util.Date;

/**
 * This is the top level class for the Database implementation of the API.
 *
 * <P>CVS $Id: PerfDBSession.java,v 1.1 2004/03/27 01:02:57 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 */
public class PerfDBSession extends DataSession {

    private DB db = null;
    private ConnectionManager connector;
	private Hashtable functionHash = null;
	private Hashtable userEventHash = null;

	public PerfDBSession () {
		super();
	}

// Initialization / termination routines

	public void initialize (Object obj) {
		String configFileName = (String)(obj);
		// initialize the connection to the database,
		// using the configuration settings.
		try {
			connector = new ConnectionManager(configFileName);
			connector.connect();
			db = connector.getDB();
		} catch ( Exception e ) {
			System.exit(0);
		}
	}

	public void initialize (Object obj, String password) {
		String configFileName = (String)(obj);
		// initialize the connection to the database,
		// using the configuration settings.
		try {
			connector = new ConnectionManager(configFileName, password);
			connector.connect();
			db = connector.getDB();
		} catch ( Exception e ) {
		}
	}

	public void terminate () {
		connector.dbclose();
	}

    public ConnectionManager getConnector(){
		return connector;
    }

	// returns Vector of ALL Application objects
	public ListIterator getApplicationList() {
		String whereClause = "";
		return new DataSessionIterator(Application.getApplicationList(db, whereClause));
	}

	// returns Vector of Experiment objects
	public ListIterator getExperimentList() {
		String whereClause = "";
		if (application != null)
			whereClause = "where application = " + application.getID();
		return new DataSessionIterator(Experiment.getExperimentList(db, whereClause));
	}

	// returns Vector of Trial objects
	public ListIterator getTrialList() {
		StringBuffer whereClause = new StringBuffer();
		if (experiment != null) {
			whereClause.append("where t.experiment = " + experiment.getID());
		} else if (application != null) {
			whereClause.append("where e.application = " + application.getID());
		}
		return new DataSessionIterator(Trial.getTrialList(db, whereClause.toString()));
	}

	// set the Application for this session
	public Application setApplication(int id) {
		this.application = null;
		this.experiment = null;
		this.trial = null;
		this.functionHash = null;
		this.userEventHash = null;
		// create a string to hit the database
		String whereClause = " where id = " + id;
		Vector applications = Application.getApplicationList(db, whereClause);
		if (applications.size() == 1) {
			this.application = (Application)applications.elementAt(0);
		} // else exception?
		return this.application;
	}

	// set the Application for this session
	public Application setApplication(String name, String version) {
		this.application = null;
		this.experiment = null;
		this.trial = null;
		this.functionHash = null;
		this.userEventHash = null;
		// create a string to hit the database
		StringBuffer whereClause = new StringBuffer();
		whereClause.append(" where name = '" + name + "'");
		if (version != null) {
			whereClause.append(" and version = " + version);
		}
		Vector applications = Application.getApplicationList(db, whereClause.toString());
		if (applications.size() == 1) {
			this.application = (Application)applications.elementAt(0);
		} // else exception?
		return this.application;
	}
	
	// set the Experiment for this session
	public Experiment setExperiment(int id) {
		this.experiment = null;
		this.trial = null;
		this.functionHash = null;
		this.userEventHash = null;
		// create a string to hit the database
		String whereClause;
		whereClause = " where id = " + id;
		Vector experiments = Experiment.getExperimentList(db, whereClause);
		if (experiments.size() == 1) {
			this.experiment = (Experiment)experiments.elementAt(0);
		} //else exception?

		return this.experiment;
	}

	// set the Trial for this session
	public Trial setTrial(int id) {
		this.trial = null;
		this.functionHash = null;
		this.userEventHash = null;
		// create a string to hit the database
		String whereClause;
		whereClause = " where t.id = " + id;
		Vector trials = Trial.getTrialList(db, whereClause);
		if (trials.size() == 1) {
			this.trial = (Trial)trials.elementAt(0);
		} //else exception?
		
		return this.trial;
	}

	// returns a ListIterator of Functions
	public ListIterator getFunctions() {
		String whereClause = new String();
		if (trial != null) {
			whereClause = " where trial = " + trial.getID();
		} else if (experiment != null) {
			whereClause = " where experiment = " + experiment.getID();
		} else if (application != null) {
			whereClause = " where application = " + application.getID();
		}

		functions = Function.getFunctions(this, db, whereClause);
		if (functionHash == null)
			functionHash = new Hashtable();
		Function fun;
        for(Enumeration en = functions.elements(); en.hasMoreElements() ;) {
			fun = (Function) en.nextElement();
			functionHash.put(new Integer(fun.getIndexID()),fun);
		}
		return new DataSessionIterator(functions);
	}

	// gets the mean & total data for a function
	public void getFunctionDetail(Function function) {
		StringBuffer buf = new StringBuffer();
		buf.append(" where id = " + function.getIndexID());
		if (metrics != null) {
			buf.append(" and metric in (");
			Metric metric;
        	for(Enumeration en = metrics.elements(); en.hasMoreElements() ;) {
				metric = (Metric) en.nextElement();
				buf.append(metric.getID());
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
		}
		FunctionDataObject.getFunctionDetail(db, function, buf.toString());
	}

	// returns a ListIterator of UserEvents
	public ListIterator getUserEvents() {
		String whereClause = new String();
		if (trial != null) {
			whereClause = " where t.id = " + trial.getID();
		} else if (experiment != null) {
			whereClause = " where t.experiment = " + experiment.getID();
		} else if (application != null) {
			whereClause = " where e.application = " + application.getID();
		}
		Vector events = UserEvent.getUserEvents(db, whereClause);
		if (userEventHash == null)
			userEventHash = new Hashtable();
		UserEvent ue;
        for(Enumeration en = events.elements(); en.hasMoreElements() ;) {
			ue = (UserEvent) en.nextElement();
			userEventHash.put(new Integer(ue.getUserEventID()), ue);
		}
		return new DataSessionIterator(events);
	}

	// sets the current function
	public Function setFunction(int id) {
		Function function = null;
		this.functions = new Vector();
		function = getFunction(id);
		if (function != null)
			this.functions.addElement(function);
		return function;
	}

	// sets the current user event
	public UserEvent setUserEvent(int id) {
		UserEvent userEvent = null;
		this.userEvents = new Vector();
		userEvent = getUserEvent(id);
		if (userEvent != null)
			this.userEvents.addElement(userEvent);
		return userEvent;
	}

	public ListIterator getFunctionData() {
		// check to make sure this is a meaningful request
		if (trial == null) {
			System.out.println("Please select a trial before getting function data.");
			return null;
		}

		// get the hash of function names first
		if (functions == null) {
			getFunctions();
		}

		// get the metric count
		int metricCount = 0;
		if (metrics != null) {
			metricCount = metrics.size();
		} else {
			metricCount = trial.getMetricCount();
		}

		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append(" where trial = " + trial.getID());
		if (nodes != null) {
			buf.append(" and node in (");
			Integer node;
        	for(Enumeration en = nodes.elements(); en.hasMoreElements() ;) {
				node = (Integer) en.nextElement();
				buf.append(node);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
		}
		if (contexts != null) {
			buf.append(" and context in (");
			Integer context;
        	for(Enumeration en = contexts.elements(); en.hasMoreElements() ;) {
				context = (Integer) en.nextElement();
				buf.append(context);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
		}
		if (threads != null) {
			buf.append(" and thread in (");
			Integer thread;
        	for(Enumeration en = threads.elements(); en.hasMoreElements() ;) {
				thread = (Integer) en.nextElement();
				buf.append(thread);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
		}
		if (functions != null) {
			buf.append(" and function in (");
			Function function;
        	for(Enumeration en = functions.elements(); en.hasMoreElements() ;) {
				function = (Function) en.nextElement();
				buf.append(function.getIndexID());
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
		}
		if (metrics != null) {
			buf.append(" and metric in (");
			Metric metric;
        	for(Enumeration en = metrics.elements(); en.hasMoreElements() ;) {
				metric = (Metric) en.nextElement();
				buf.append(metric.getID());
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
		}
		return FunctionDataObject.getFunctionData(db, metricCount, buf.toString());
	}
	
	public ListIterator getUserEventData() {
		// check to make sure this is a meaningful request
		if (trial == null) {
			System.out.println("Please select a trial before getting user event data.");
			return null;
		}

		// get the hash of userEvent names first
		if (userEvents == null)
			getUserEvents();

		StringBuffer buf = new StringBuffer();
		buf.append(" where t.id = " + trial.getID());
		if (nodes != null) {
			buf.append(" and p.node in (");
			Integer node;
        	for(Enumeration en = nodes.elements(); en.hasMoreElements() ;) {
				node = (Integer) en.nextElement();
				buf.append(node);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
		}
		if (contexts != null) {
			buf.append(" and p.context in (");
			Integer context;
        	for(Enumeration en = contexts.elements(); en.hasMoreElements() ;) {
				context = (Integer) en.nextElement();
				buf.append(context);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
		}
		if (threads != null) {
			buf.append(" and p.thread in (");
			Integer thread;
        	for(Enumeration en = threads.elements(); en.hasMoreElements() ;) {
				thread = (Integer) en.nextElement();
				buf.append(thread);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
		}
		if (userEvents != null) {
			buf.append(" and u.id in (");
			UserEvent userEvent;
        	for(Enumeration en = userEvents.elements(); en.hasMoreElements() ;) {
				userEvent = (UserEvent) en.nextElement();
				buf.append(userEvent.getUserEventID());
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
		}

		return new DataSessionIterator(UserEventDataObject.getUserEventData(db, buf.toString()));
	}
	
	public Function getFunction(int id) {
		Function function = null;
		if (functionHash != null) {
			function = (Function)functionHash.get(new Integer(id));
		}
		if (function == null) {
			// create a string to hit the database
			String whereClause;
			whereClause = " where id = " + id;
			Vector functions = Function.getFunctions(this, db, whereClause);
			if (functions.size() == 1) {
				function = (Function)functions.elementAt(0);
			} //else exception?
			if (functionHash == null)
				functionHash = new Hashtable();
			functionHash.put(new Integer(function.getIndexID()),function);
		}
		return function;
	}
	
	public UserEvent getUserEvent(int id) {
		UserEvent userEvent = null;
		if (userEventHash != null) {
			userEvent = (UserEvent)userEventHash.get(new Integer(id));
		}
		if (userEvent == null) {
			// create a string to hit the database
			String whereClause;
			whereClause = " where u.id = " + id;
			Vector userEvents = UserEvent.getUserEvents(db, whereClause);
			if (userEvents.size() == 1) {
				userEvent = (UserEvent)userEvents.elementAt(0);
			} //else exception?
			if (userEventHash == null)
				userEventHash = new Hashtable();
			userEventHash.put(new Integer(userEvent.getUserEventID()), userEvent);
		}
		return userEvent;
	}
	
	// override the saveTrial method
	public void saveTrial () {
		int newTrialID = trial.saveTrial(db);
		saveFunctions(newTrialID);
		return;
	}

	// save the functions
	private void saveFunctions(int newTrialID) {
		Hashtable newFunHash = new Hashtable();
		Enumeration enum = functions.elements();
		Function function;
		while (enum.hasMoreElements()) {
			function = (Function)enum.nextElement();
			int newFunctionID = function.saveFunction(db, newTrialID, metrics);
			newFunHash.put (new Integer(function.getIndexID()), new Integer(newFunctionID));
		}
		saveFunctionData(newFunHash);
	}

	// save the function data
	private void saveFunctionData(Hashtable newFunHash) {
	}

};

