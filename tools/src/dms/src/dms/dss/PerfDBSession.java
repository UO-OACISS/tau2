package dms.dss;

import dms.perfdb.*;
import java.util.*;
import java.sql.*;
import java.util.Date;

/**
 * This is the top level class for the Database implementation of the API.
 *
 * <P>CVS $Id: PerfDBSession.java,v 1.21 2004/04/09 22:12:56 khuck Exp $</P>
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
		return setTrial(id, true);
	}

	private Trial setTrial(int id, boolean clearHashes) {
		this.trial = null;
		if (clearHashes) {
			this.functionHash = null;
			this.userEventHash = null;
		}
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
			functionHash.put(new Integer(fun.getID()),fun);
		}
		return new DataSessionIterator(functions);
	}

	// gets the mean & total data for a function
	public void getFunctionDetail(Function function) {
		StringBuffer buf = new StringBuffer();
		buf.append(" where id = " + function.getID());
		if (metrics != null && metrics.size() > 0) {
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
		userEvents = UserEvent.getUserEvents(db, whereClause);
		if (userEventHash == null)
			userEventHash = new Hashtable();
		UserEvent ue;
        for(Enumeration en = userEvents.elements(); en.hasMoreElements() ;) {
			ue = (UserEvent) en.nextElement();
			userEventHash.put(new Integer(ue.getUserEventID()), ue);
		}
		return new DataSessionIterator(userEvents);
	}

	// sets the current function
	public Function setFunction(int id) {
		Function function = null;
		this.functions = new Vector();
		function = getFunction(id);
		if (function != null)
			this.functions.addElement(function);
		// we need this to get the metric data...
		setTrial(function.getTrialID(), false);
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
		if (trial == null && functions == null) {
			System.out.println("Please select a trial or a set of functions before getting function data.");
			return null;
		}

		// get the hash of function names first
		if (functions == null) {
			getFunctions();
		}

		// get the metric count
		int metricCount = 0;
		if (metrics != null && metrics.size() > 0) {
			metricCount = metrics.size();
		} else {
			metricCount = trial.getMetricCount();
		}

		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append(" where trial = " + trial.getID());
		if (nodes != null && nodes.size() > 0) {
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
		if (contexts != null && contexts.size() > 0) {
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
		if (threads != null && threads.size() > 0) {
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
		if (functions != null && functions.size() > 0) {
			buf.append(" and function in (");
			Function function;
        	for(Enumeration en = functions.elements(); en.hasMoreElements() ;) {
				function = (Function) en.nextElement();
				buf.append(function.getID());
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
		}
		if (metrics != null && metrics.size() > 0) {
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
		functionData = FunctionDataObject.getFunctionData(db, metricCount, buf.toString());
		return new DataSessionIterator(functionData);
	}
	
	public ListIterator getUserEventData() {
		// check to make sure this is a meaningful request
		if (trial == null && userEvents == null) {
			System.out.println("Please select a trial or a set of user events before getting user event data.");
			return null;
		}

		// get the hash of userEvent names first
		if (userEvents == null)
			getUserEvents();

		StringBuffer buf = new StringBuffer();
		buf.append(" where t.id = " + trial.getID());
		if (nodes != null && nodes.size() > 0) {
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
		if (contexts != null && contexts.size() > 0) {
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
		if (threads != null && threads.size() > 0) {
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
		if (userEvents != null && userEvents.size() > 0) {
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

		userEventData = UserEventDataObject.getUserEventData(db, buf.toString());
		return new DataSessionIterator(userEventData);
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
			functionHash.put(new Integer(function.getID()),function);
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
	public int saveTrial () {
		int newTrialID = trial.saveTrial(db);
		Hashtable newMetHash = saveMetrics(newTrialID, trial, -1);
		Hashtable newFunHash = saveFunctions(newTrialID, newMetHash, -1);
		saveFunctionData(newFunHash, newMetHash, -1);
		Hashtable newUEHash = saveUserEvents(newTrialID);
		saveUserEventData(newUEHash);
		return newTrialID;
	}

	// save the metrics
	private Hashtable saveMetrics(int newTrialID, Trial trial, int saveMetricIndex) {
		System.out.print("Saving the metrics: ");
		Hashtable newMetHash = new Hashtable();
	    Enumeration enum = trial.getDataSession().getMetrics().elements();
	    Metric metric;
		int i = 0;
	    while (enum.hasMoreElements()) {
			metric = (Metric)enum.nextElement();
			int newMetricID = 0;
			if (saveMetricIndex < 0 || saveMetricIndex == i) {
				newMetricID = metric.saveMetric(db, newTrialID);
				System.out.print("\rSaving the metrics: " + (i+1) + " records saved...");
			}
			newMetHash.put(new Integer(i), new Integer(newMetricID));
			i++;
	    }
		System.out.print("\n");
		return newMetHash;
	}

	// save the functions
	private Hashtable saveFunctions(int newTrialID, Hashtable newMetHash, int saveMetricIndex) {
		System.out.print("Saving the functions: ");
		Hashtable newFunHash = new Hashtable();
		Enumeration enum = functions.elements();
		Function function;
		int count = 0;
		while (enum.hasMoreElements()) {
			function = (Function)enum.nextElement();
			int newFunctionID = function.saveFunction(db, newTrialID, newMetHash, saveMetricIndex);
			newFunHash.put (new Integer(function.getID()), new Integer(newFunctionID));
			System.out.print("\rSaving the functions: " + ++count + " records saved...");
		}
		System.out.print("\n");
		return newFunHash;
	}

	// save the function data
	private void saveFunctionData(Hashtable newFunHash, Hashtable newMetHash, int saveMetricIndex) {
		System.out.print("Saving the function data: ");
		Enumeration enum = functionData.elements();
		FunctionDataObject fdo;
		int count = 0;
		while (enum.hasMoreElements()) {
			fdo = (FunctionDataObject)enum.nextElement();
			Integer newFunctionID = (Integer)newFunHash.get(new Integer(fdo.getFunctionIndexID()));
			fdo.saveFunctionData(db, newFunctionID.intValue(), newMetHash, saveMetricIndex);
			System.out.print("\rSaving the function data: " + ++count + " records saved...");
		}
		System.out.print("\n");
	}

	// save the functions
	private Hashtable saveUserEvents(int newTrialID) {
		System.out.print("Saving the user events:");
		Hashtable newUEHash = new Hashtable();
		Enumeration enum = userEvents.elements();
		int count = 0;
		UserEvent userEvent;
		while (enum.hasMoreElements()) {
			userEvent = (UserEvent)enum.nextElement();
			int newUserEventID = userEvent.saveUserEvent(db, newTrialID);
			newUEHash.put (new Integer(userEvent.getUserEventID()), new Integer(newUserEventID));
			System.out.print("\rSaving the user events: " + ++count + " records saved...");
		}
		System.out.print("\n");
		return newUEHash;
	}

	// save the function data
	private void saveUserEventData(Hashtable newUEHash) {
		System.out.print("Saving the user event data:");
		Enumeration enum = userEventData.elements();
		UserEventDataObject uedo;
		int count = 0;
		while (enum.hasMoreElements()) {
			uedo = (UserEventDataObject)enum.nextElement();
			Integer newUserEventID = (Integer)newUEHash.get(new Integer(uedo.getUserEventID()));
			uedo.saveUserEventData(db, newUserEventID.intValue());
			System.out.print("\rSaving the user event data: " + ++count + " records saved...");
		}
		System.out.print("\n");
	}

/**
 * Saves the Trial.
 *
 * @param function
 * @return database index ID of the saved trial record
 */
	public int saveTrial(Trial trial) {
		return trial.saveTrial(db);
	}

/**
 * Saves the Function.
 *
 * @param function
 * @return database index ID of the saved function record
 */
	public int saveFunction(Function function, int newTrialID, Hashtable newMetHash) {
		return function.saveFunction(db, newTrialID, newMetHash, -1);
	}

/**
 * Saves the FunctionDataObject.
 *
 * @param functionData
 * @return database index ID of the saved interval_location_profile record
 */
	public void saveFunctionData(FunctionDataObject functionData, int newFunctionID, Hashtable newMetHash) {
		functionData.saveFunctionData(db, newFunctionID, newMetHash, -1);
		return;
	}

/**
 * Saves the UserEvent object.
 *
 * @param userEvent
 * @return database index ID of the saved user_event record
 */
	public int saveUserEvent(UserEvent userEvent, int newTrialID) {
		return userEvent.saveUserEvent(db, newTrialID);
	}

/**
 * Saves the userEventData object.
 *
 * @param userEventData
 * @return database index ID of the saved atomic_location_profile record
 */
	public void saveUserEventData(UserEventDataObject userEventData, int newUserEventID) {
		userEventData.saveUserEventData(db, newUserEventID);
		return;
	}

/**
 * Saves the ParaProfTrial object to the database
 * 
 * @param paraProfTrial
 * @return the database index ID of the saved trial record
 */

	public int saveParaProfTrial(Trial trial, int saveMetricIndex) {
		GlobalMapping mapping = trial.getDataSession().getGlobalMapping();
	
		//Build an array of group names.  This speeds lookup of group names.
		Vector groups = mapping.getMapping(1);
		String[] groupNames = new String[groups.size()];
		int position = 0;
		for(Enumeration e = groups.elements(); e.hasMoreElements() ;){
	    	GlobalMappingElement group = (GlobalMappingElement) e.nextElement();
	    	groupNames[position++] = group.getMappingName();
		}

		// get the metric count
		metrics = trial.getDataSession().getMetrics();
		int metricCount = metrics.size();
		System.out.println("Found " + metricCount + " metrics...");

		// create the Vectors to store the data
		functions = new Vector();
		functionData = new Vector();
		userEvents = new Vector();
		userEventData = new Vector();

		int fcount = 0;
		int ucount = 0;
		// create the functions
		System.out.print("Getting the functions:");
		for(Enumeration e = mapping.getMapping(0).elements(); e.hasMoreElements() ;) {
			GlobalMappingElement element = (GlobalMappingElement) e.nextElement();
			if(element!=null) {
				// create a function
				Function function = new Function(this);
				function.setName(element.getMappingName());
				function.setID(element.getMappingID());
				// function.setTrialID(newTrialID);
				// build the group name
				int[] groupIDs = element.getGroups();
				StringBuffer buf = new StringBuffer();
				for (int i = 0; i < element.getNumberOfGroups() ; i++) {
					if (i > 0) buf.append ("|");
					buf.append(groupNames[groupIDs[i]]);
				}
				function.setGroup(buf.toString());
				// put the function in the vector
				functions.add(function);

				// get the total data
				System.out.print("\rGetting the functions: " + ++fcount + " functions found...");
				FunctionDataObject funTS = new FunctionDataObject(metricCount);
				FunctionDataObject funMS = new FunctionDataObject(metricCount);
				for (int i = 0 ; i < metricCount ; i++) {
		    		funTS.setNumCalls((int)element.getTotalNumberOfCalls());
		    		funTS.setNumSubroutines((int)element.getTotalNumberOfSubRoutines());
		    		funTS.setInclusivePercentage(i, element.getTotalInclusivePercentValue(i));
		    		funTS.setInclusive(i, element.getTotalInclusiveValue(i));
		    		funTS.setExclusivePercentage(i, element.getTotalExclusivePercentValue(i));
		    		funTS.setExclusive(i, element.getTotalExclusiveValue(i));
		    		funTS.setInclusivePerCall(i, element.getTotalUserSecPerCall(i));
		    		funMS.setNumCalls((int)element.getMeanNumberOfCalls());
		    		funMS.setNumSubroutines((int)element.getMeanNumberOfSubRoutines());
		    		funMS.setInclusivePercentage(i, element.getMeanInclusivePercentValue(i));
					// System.out.println("Inclusive(" + i + "): " + element.getMeanInclusivePercentValue(i));
		    		funMS.setInclusive(i, element.getMeanInclusiveValue(i));
		    		funMS.setExclusivePercentage(i, element.getMeanExclusivePercentValue(i));
		    		funMS.setExclusive(i, element.getMeanExclusiveValue(i));
		    		funMS.setInclusivePerCall(i, element.getMeanUserSecPerCall(i));
				}
				function.setTotalSummary(funTS);
				function.setMeanSummary(funMS);
	    	}
	    }

		// create the user events
		System.out.print("\nGetting user events:");
		for(Enumeration e = mapping.getMapping(2).elements(); e.hasMoreElements() ;) {
			GlobalMappingElement element = (GlobalMappingElement) e.nextElement();
			if(element!=null) {
				System.out.print(".");
				System.out.print("\rGetting the user events: " + ++ucount + " user events found...");
				// create a user event
				UserEvent userEvent = new UserEvent();
				userEvent.setName(element.getMappingName());
				userEvent.setUserEventID(element.getMappingID());
				// build the group name
				int[] groupIDs = element.getGroups();
				StringBuffer buf = new StringBuffer();
				for (int i = 0; i < element.getNumberOfGroups() ; i++) {
					if (i > 0) buf.append ("|");
					buf.append(groupNames[groupIDs[i]]);
				}
				userEvent.setGroup(buf.toString());
				// put the userEvent in the vector
				userEvents.add(userEvent);
	    	}
	    }

		fcount = 0;
		ucount = 0;
		System.out.print("\nGetting the function / user event data:");
	    StringBuffer groupsStringBuffer = new StringBuffer(10);
	    Vector nodes = trial.getDataSession().getNCT().getNodes();
	    for(Enumeration e1 = nodes.elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		Vector contexts = node.getContexts();
		for(Enumeration e2 = contexts.elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    Vector threads = context.getThreads();
		    for(Enumeration e3 = threads.elements(); e3.hasMoreElements() ;){
			dms.dss.Thread thread = (dms.dss.Thread) e3.nextElement();
			Vector functions = thread.getFunctionList();
			Vector userevents = thread.getUsereventList();
			//Write out function data for this thread.
			for(Enumeration e4 = functions.elements(); e4.hasMoreElements() ;){
			    GlobalThreadDataElement function = (GlobalThreadDataElement) e4.nextElement();
			    if (function!=null){
					System.out.print("\rGetting the function / user event data: " + ++fcount + " / " + ucount + " found...");
					FunctionDataObject fdo = new FunctionDataObject(metricCount);
					fdo.setNode(thread.getNodeID());
					fdo.setContext(thread.getContextID());
					fdo.setThread(thread.getThreadID());
					fdo.setFunctionIndexID(function.getMappingID());
					fdo.setNumCalls(function.getNumberOfCalls());
					fdo.setNumSubroutines(function.getNumberOfSubRoutines());
					// fdo.setInclusivePerCall(function.getUserSecPerCall());
					for (int i = 0 ; i < metricCount ; i++) {
						fdo.setInclusive(i, function.getInclusiveValue(i));
						fdo.setExclusive(i, function.getExclusiveValue(i));
						fdo.setInclusivePercentage(i, function.getInclusivePercentValue(i));
						fdo.setExclusivePercentage(i, function.getExclusivePercentValue(i));
						fdo.setInclusivePerCall(i, function.getUserSecPerCall(i));
					}
					functionData.add(fdo);
			    }
			}

			//Write out user event data for this thread.
			if(userevents!=null){
			    for(Enumeration e4 = userevents.elements(); e4.hasMoreElements() ;){
				GlobalThreadDataElement userevent = (GlobalThreadDataElement) e4.nextElement();
				if (userevent!=null){
					System.out.print("\rGetting the function / user event data: " + fcount + " / " + ++ucount + " found...");
					UserEventDataObject udo = new UserEventDataObject();
				    udo.setUserEventID(userevent.getMappingID());
					udo.setNode(thread.getNodeID());
					udo.setContext(thread.getContextID());
					udo.setThread(thread.getThreadID());
				    udo.setProfileID(userevent.getUserEventNumberValue());
				    udo.setMaximumValue(userevent.getUserEventMaxValue());
				    udo.setMinimumValue(userevent.getUserEventMinValue());
				    udo.setMeanValue(userevent.getUserEventMeanValue());
				    // udo.setStandardDeviation(userevent.getUserEventStdDevValue());
					userEventData.add(udo);
				}
			    }
			}
		    }
		}    
	    }

		int newTrialID = 0;
		// output the trial data, which also saves the functions, 
		// function data, user events and user event data
		if (saveMetricIndex < 0) {
			System.out.println("\nSaving the trial...");
			newTrialID = trial.saveTrial(db);
			Hashtable newMetHash = saveMetrics(newTrialID, trial, saveMetricIndex);

			if (functions != null && functions.size() > 0) {
				Hashtable newFunHash = saveFunctions(newTrialID, newMetHash, saveMetricIndex);
				saveFunctionData(newFunHash, newMetHash, saveMetricIndex);
			}
			if (userEvents != null && userEvents.size() > 0) {
				Hashtable newUEHash = saveUserEvents(newTrialID);
				if (userEventData != null && userEventData.size() > 0) {
					saveUserEventData(newUEHash);
				}
			}

			System.out.println("New Trial ID: " + newTrialID);
		} else {
			newTrialID = trial.getID();
			System.out.println("\nSaving the metric...");
			Hashtable newMetHash = saveMetrics(newTrialID, trial, saveMetricIndex);
			if (functions != null && functions.size() > 0) {
				Hashtable newFunHash = saveFunctions(newTrialID, newMetHash, saveMetricIndex);
				saveFunctionData(newFunHash, newMetHash, saveMetricIndex);
			}

			System.out.println("Modified Trial ID: " + newTrialID);
		}

		vacuumDatabase();
		return newTrialID;
    }

	private void vacuumDatabase() {
		// don't do this for mysql or db2
	    if (db.getDBType().compareTo("mysql") == 0)
			return;
		else if (db.getDBType().compareTo("db2") == 0)
			return;
		String vacuum = "vacuum;";
		String analyze = "analyze;";
		try {
			System.out.println("Vacuuming database...");
			db.executeUpdate(vacuum);
			System.out.println("Analyzing database...");
			db.executeUpdate(analyze);
		} catch (SQLException e) {
	    	System.out.println("An error occurred while vacuuming the database.");
	    	e.printStackTrace();
		}
	}
};

