 package dms.dss;

import perfdb.util.dbinterface.*;
import perfdb.util.io.*;
import perfdb.dbmanager.*;
import java.io.*;
import java.util.*;
import java.net.*;
import java.sql.*;

/**
 * This is the top level class for the Database implementation of the API.
 *
 * <P>CVS $Id: PerfDBSession.java,v 1.18 2003/08/07 20:23:08 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	%I%, %G%
 */
public class PerfDBSession extends DataSession {

    private DB db = null;
    private perfdb.ConnectionManager connector;
	private Hashtable functionHash = null;
	private Hashtable userEventHash = null;

	public PerfDBSession () {
		super();
	}

	public void initialize (Object obj) {
		String configFileName = (String)(obj);
		// initialize the connection to the database,
		// using the configuration settings.
		connector = new perfdb.ConnectionManager(configFileName);
		connector.connect();
		db = connector.getDB();
	}

	public void terminate () {
		connector.dbclose();
	}

    public perfdb.ConnectionManager getConnector(){
		return connector;
    }

	// returns Vector of ALL Application objects
	public ListIterator getApplicationList() {
		String whereClause = "";
		return new DataSessionIterator(getApplicationList(whereClause));
	}

	// returns Vector of Application objects, filtered by where clause
	private Vector getApplicationList(String whereClause) {
		Vector applications = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct id, name, version, description, ");
		buf.append(" language, para_diag, usage_text, execution_options, ");
		buf.append(" experiment_table_name from application");
		buf.append(whereClause);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	while (resultSet.next() != false) {
				Application app = new Application();
				app.setID(resultSet.getInt(1));
				app.setName(resultSet.getString(2));
				app.setVersion(resultSet.getString(3));
				app.setDescription(resultSet.getString(4));
				app.setLanguage(resultSet.getString(5));
				app.setParaDiag(resultSet.getString(6));
				app.setUsage(resultSet.getString(7));
				app.setExecutableOptions(resultSet.getString(8));
				app.setExperimentTableName(resultSet.getString(9));
				applications.addElement(app);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return applications;
	}

	// returns Vector of Experiment objects
	public ListIterator getExperimentList() {
		String whereClause = "";
		if (application != null)
			whereClause = "where application = " + application.getID();
		return new DataSessionIterator(getExperimentList(whereClause));
	}

	// returns Vector of Experiment objects
	public Vector getExperimentList(String whereClause) {
		Vector experiments = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct id, application, system_info, ");
		buf.append("configuration_info, instrumentation_info, ");
		buf.append("compiler_info, trial_table_name from experiment ");
		buf.append(whereClause);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	while (resultSet.next() != false) {
				Experiment exp = new Experiment();
				exp.setID(resultSet.getInt(1));
				exp.setApplicationID(resultSet.getInt(2));
				exp.setSystemInfo(resultSet.getString(3));
				exp.setConfigurationInfo(resultSet.getString(4));
				exp.setInstrumentationInfo(resultSet.getString(5));
				exp.setCompilerInfo(resultSet.getString(6));
				exp.setTrialTableName(resultSet.getString(7));
				experiments.addElement(exp);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return experiments;
	}

	// returns Vector of Trial objects
	public ListIterator getTrialList() {
		StringBuffer whereClause = new StringBuffer();
		boolean gotWhere = false;
		if (application != null) {
			whereClause.append("where e.application = " + application.getID());
			gotWhere = true;
		}
		if (experiment != null) {
			if (gotWhere)
				whereClause.append(" and");
			else
				whereClause.append(" where");
			whereClause.append(" t.experiment = " + experiment.getID());
		}
		return new DataSessionIterator(getTrialList(whereClause.toString()));
	}

	// returns Vector of Trial objects
	public Vector getTrialList(String whereClause) {
		Vector trials = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct t.id, t.experiment, e.application, ");
		buf.append("t.time, t.problem_size, t.node_count, ");
		buf.append("t.contexts_per_node, t.threads_per_context ");
		buf.append("from trial t inner join experiment e ");
		buf.append("on t.experiment = e.id ");
		buf.append(whereClause);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	while (resultSet.next() != false) {
				Trial trial = new Trial();
				trial.setID(resultSet.getInt(1));
				trial.setExperimentID(resultSet.getInt(2));
				trial.setApplicationID(resultSet.getInt(3));
				trial.setTime(resultSet.getString(4));
				trial.setProblemSize(resultSet.getInt(5));
				trial.setNodeCount(resultSet.getInt(6));
				trial.setNumContextsPerNode(resultSet.getInt(7));
				trial.setNumThreadsPerContext(resultSet.getInt(8));
				trials.addElement(trial);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return trials;
	}

	// set the Application for this session
	public Application setApplication(int id) {
		this.application = null;
		this.experiment = null;
		this.trials = null;
		this.functionHash = null;
		this.userEventHash = null;
		// create a string to hit the database
		String whereClause = " where id = " + id;
		Vector applications = getApplicationList(whereClause);
		if (applications.size() != 1) {
			this.application = (Application)applications.elementAt(0);
		} // else exception?
		return this.application;
	}

	public Application setApplication(String name, String version) {
		this.application = null;
		this.experiment = null;
		this.trials = null;
		this.functionHash = null;
		this.userEventHash = null;
		// create a string to hit the database
		StringBuffer whereClause = new StringBuffer();
		whereClause.append(" where name = '" + name + "'");
		if (version != null) {
			whereClause.append(" and version = " + version);
		}
		Vector applications = getApplicationList(whereClause.toString());
		if (applications.size() == 1) {
			this.application = (Application)applications.elementAt(0);
		} // else exception?
		return this.application;
	}
	
	// set the Experiment for this session
	public Experiment setExperiment(int id) {
		this.experiment = null;
		this.trials = null;
		this.functionHash = null;
		this.userEventHash = null;
		// create a string to hit the database
		String whereClause;
		whereClause = " where id = " + id;
		Vector experiments = getExperimentList(whereClause);
		if (experiments.size() == 1) {
			this.experiment = (Experiment)experiments.elementAt(0);
		} //else exception?
		return this.experiment;
	}

	// set the Trial for this session
	public Trial setTrial(int id) {
		Trial trial = null;
		this.trials = null;
		this.functionHash = null;
		this.userEventHash = null;
		// create a string to hit the database
		String whereClause;
		whereClause = " where t.id = " + id;
		Vector trials = getTrialList(whereClause);
		if (trials.size() == 1) {
			trial = (Trial)trials.elementAt(0);
			this.trials = trials;
		} //else exception?
		
		return trial;
	}

	// returns a ListIterator of Functions
	public ListIterator getFunctions() {
		StringBuffer whereClause = new StringBuffer();
		boolean gotWhile = false;
		if (application != null) {
			whereClause.append(" where e.application = " + application.getID());
			gotWhile = true;
		}
		if (experiment != null) {
			if (gotWhile)
				whereClause.append(" and");
			else
				whereClause.append(" where");
			whereClause.append(" t.experiment = " + experiment.getID());
			gotWhile = true;
		}
		if (trials != null) {
			if (gotWhile)
				whereClause.append(" and t.id in (");
			else
				whereClause.append(" where t.id in (");
			Trial trial;
        	for(Enumeration en = trials.elements(); en.hasMoreElements() ;) {
				trial = (Trial) en.nextElement();
				whereClause.append(trial.getID());
				if (en.hasMoreElements())
					whereClause.append(", ");
				else
					whereClause.append(") ");
			}
		}

		return new DataSessionIterator(getFunctions(whereClause.toString()));
	}

	// returns a Vector of Functions
	public Vector getFunctions(String whereClause) {
		if (functionHash == null)
			functionHash = new Hashtable();
		Vector funs = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct f.id, f.function_number, f.name, ");
		buf.append("f.group_name, f.trial, t.experiment, e.application, ");
		buf.append("m.name, ");
		buf.append("ms.inclusive_percentage, ms.inclusive, ");
		buf.append("ms.exclusive_percentage, ms.exclusive, ");
		buf.append("ms.call, ms.subroutines, ms.inclusive_per_call, ");
		buf.append("m2.name, ");
		buf.append("ts.inclusive_percentage, ts.inclusive, ");
		buf.append("ts.exclusive_percentage, ts.exclusive, ");
		buf.append("ts.call, ts.subroutines, ts.inclusive_per_call ");
		buf.append("from function f inner join trial t on f.trial = t.id ");
		buf.append("inner join experiment e on t.experiment = e.id ");
		buf.append("inner join interval_mean_summary ms on f.id = ms.function ");
		buf.append("inner join interval_total_summary ts on f.id = ts.function ");
		buf.append("inner join metric m on ms.metric = m.id ");
		buf.append("inner join metric m2 on ts.metric = m.id ");
		buf.append(whereClause);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
			Function tmpFunction = null;
	    	while (resultSet.next() != false) {
				Function fun = new Function();
				fun.setIndexID(resultSet.getInt(1));
				fun.setFunctionID(resultSet.getInt(2));
				fun.setName(resultSet.getString(3));
				fun.setGroup(resultSet.getString(4));
				fun.setTrialID(resultSet.getInt(5));
				fun.setExperimentID(resultSet.getInt(6));
				fun.setApplicationID(resultSet.getInt(7));
				// get the mean summary data
				FunctionDataObject funMS = new FunctionDataObject();
				funMS.setMetric(0, resultSet.getString(8));
				funMS.setInclusivePercentage(0, resultSet.getDouble(9));
				funMS.setInclusive(0, resultSet.getDouble(10));
				funMS.setExclusivePercentage(0, resultSet.getDouble(11));
				funMS.setExclusive(0, resultSet.getDouble(12));
				funMS.setNumCalls((int)(resultSet.getDouble(13)));
				funMS.setNumSubroutines((int)(resultSet.getDouble(14)));
				funMS.setInclusivePerCall(0, resultSet.getDouble(15));
				fun.setMeanSummary(funMS);
				// get the total summary data
				FunctionDataObject funTS = new FunctionDataObject();
				funTS.setMetric(0, resultSet.getString(16));
				funTS.setInclusivePercentage(0, resultSet.getDouble(17));
				funTS.setInclusive(0, resultSet.getDouble(18));
				funTS.setExclusivePercentage(0, resultSet.getDouble(19));
				funTS.setExclusive(0, resultSet.getDouble(20));
				funTS.setNumCalls((int)(resultSet.getDouble(21)));
				funTS.setNumSubroutines((int)(resultSet.getDouble(22)));
				funTS.setInclusivePerCall(0, resultSet.getDouble(23));
				fun.setTotalSummary(funTS);
				funs.addElement(fun);
				tmpFunction = (Function)functionHash.get(new Integer(fun.getIndexID()));
				if (tmpFunction == null)
					functionHash.put(new Integer(fun.getIndexID()), fun);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return funs;
	}

	// returns a ListIterator of UserEvents
	public ListIterator getUserEvents() {
		StringBuffer whereClause = new StringBuffer();
		boolean gotWhile = false;
		if (application != null) {
			whereClause.append(" where e.application = " + application.getID());
			gotWhile = true;
		}
		if (experiment != null) {
			if (gotWhile)
				whereClause.append(" and");
			else
				whereClause.append(" where");
			whereClause.append(" t.experiment = " + experiment.getID());
			gotWhile = true;
		}
		if (trials != null) {
			if (gotWhile)
				whereClause.append(" and t.id in (");
			else
				whereClause.append(" where t.id in (");
			Trial trial;
        	for(Enumeration en = trials.elements(); en.hasMoreElements() ;) {
				trial = (Trial) en.nextElement();
				whereClause.append(trial.getID());
				if (en.hasMoreElements())
					whereClause.append(", ");
				else
					whereClause.append(") ");
			}
		}

		return new DataSessionIterator(getUserEvents(whereClause.toString()));
	}

	// returns a Vector of UserEvents
	public Vector getUserEvents(String whereClause) {
		if (userEventHash == null)
			userEventHash = new Hashtable();
		Vector userEvents = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct u.id, u.trial, u.name, ");
		buf.append("u.group_name ");
		buf.append("from user_event u inner join trial t on u.trial = t.id ");
		buf.append("inner join experiment e on t.experiment = e.id ");
		buf.append(whereClause);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
			UserEvent tmpUserEvent = null;
	    	while (resultSet.next() != false) {
				UserEvent ue = new UserEvent();
				ue.setUserEventID(resultSet.getInt(1));
				ue.setTrialID(resultSet.getInt(2));
				ue.setName(resultSet.getString(3));
				ue.setGroup(resultSet.getString(4));
				userEvents.addElement(ue);
				tmpUserEvent = (UserEvent)userEventHash.get(new Integer(ue.getUserEventID()));
				if (tmpUserEvent == null)
					userEventHash.put(new Integer(ue.getUserEventID()), ue);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return userEvents;
	}

	public Function setFunction(int id) {
		Function function = null;
		this.functions = new Vector();
		function = getFunction(id);
		if (function != null)
			this.functions.addElement(function);
		return function;
	}

	public UserEvent setUserEvent(int id) {
		UserEvent userEvent = null;
		this.userEvents = new Vector();
		userEvent = getUserEvent(id);
		if (userEvent != null)
			this.userEvents.addElement(userEvent);
		return userEvent;
	}

	public ListIterator getFunctionData() {
		// get the hash of function names first
		if (functions == null)
			getFunctions();

		Vector functionData = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct p.inclusive_percentage, ");
		buf.append("p.inclusive, p.exclusive_percentage, p.exclusive, ");
		buf.append("p.call, p.subroutines, p.inclusive_per_call, ");
		buf.append("f.trial, p.node, p.context, p.thread, p.function, m.name, p.metric ");
		buf.append("from interval_location_profile p ");
		buf.append("inner join function f on f.id = p.function ");
		buf.append("inner join trial t on f.trial = t.id ");
		buf.append("inner join experiment e on e.id = t.experiment ");
		buf.append("inner join metric m on p.metric = m.id ");
		boolean gotWhile = false;
		if (application != null) {
			buf.append(" where e.application = " + application.getID());
			gotWhile = true;
		}
		if (experiment != null) {
			if (gotWhile)
				buf.append(" and");
			else
				buf.append(" where");
			buf.append(" t.experiment = " + experiment.getID());
			gotWhile = true;
		}
		if (trials != null) {
			if (gotWhile)
				buf.append(" and t.id in (");
			else
				buf.append(" where t.id in (");
			Trial trial;
        	for(Enumeration en = trials.elements(); en.hasMoreElements() ;) {
				trial = (Trial) en.nextElement();
				buf.append(trial.getID());
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
			gotWhile = true;
		}
		if (nodes != null) {
			if (gotWhile)
				buf.append(" and p.node in (");
			else
				buf.append(" where p.node in (");
			Integer node;
        	for(Enumeration en = nodes.elements(); en.hasMoreElements() ;) {
				node = (Integer) en.nextElement();
				buf.append(node);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
			gotWhile = true;
		}
		if (contexts != null) {
			if (gotWhile)
				buf.append(" and p.context in (");
			else
				buf.append(" where p.context in (");
			Integer context;
        	for(Enumeration en = contexts.elements(); en.hasMoreElements() ;) {
				context = (Integer) en.nextElement();
				buf.append(context);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
			gotWhile = true;
		}
		if (threads != null) {
			if (gotWhile)
				buf.append(" and p.thread in (");
			else
				buf.append(" where p.thread in (");
			Integer thread;
        	for(Enumeration en = threads.elements(); en.hasMoreElements() ;) {
				thread = (Integer) en.nextElement();
				buf.append(thread);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
			gotWhile = true;
		}
		if (functions != null) {
			if (gotWhile)
				buf.append(" and f.id in (");
			else
				buf.append(" where f.id in (");
			Function function;
        	for(Enumeration en = functions.elements(); en.hasMoreElements() ;) {
				function = (Function) en.nextElement();
				buf.append(function.getFunctionID());
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
			gotWhile = true;
		}
		buf.append(" order by f.trial, p.function, p.metric, p.node, p.context, p.thread ");
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	while (resultSet.next() != false) {
				FunctionDataObject funDO = new FunctionDataObject();
				funDO.setInclusivePercentage(0, resultSet.getDouble(1));
				funDO.setInclusive(0, resultSet.getDouble(2));
				funDO.setExclusivePercentage(0, resultSet.getDouble(3));
				funDO.setExclusive(0, resultSet.getDouble(4));
				funDO.setNumCalls((int)(resultSet.getDouble(5)));
				funDO.setNumSubroutines((int)(resultSet.getDouble(6)));
				funDO.setInclusivePerCall(0, resultSet.getDouble(7));
				funDO.setNode(resultSet.getInt(9));
				funDO.setContext(resultSet.getInt(10));
				funDO.setThread(resultSet.getInt(11));
				funDO.setFunctionIndexID(resultSet.getInt(12));
				funDO.setMetric(0, resultSet.getString(13));
				functionData.addElement(funDO);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		return new DataSessionIterator(functionData);
	}
	
	public ListIterator getUserEventData() {
		// get the hash of userEvent names first
		if (userEvents == null)
			getUserEvents();

		Vector userEventData = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct p.id, p.user_event, p.node, ");
		buf.append("p.context, p.thread, p.sample_count, ");
		buf.append("p.maximum_value, p.minimum_value, p.mean_value, ");
		buf.append("p.standard_deviation, u.trial ");
		buf.append("from atomic_location_profile p ");
		buf.append("inner join user_event u on u.id = p.user_event ");
		buf.append("inner join trial t on u.trial = t.id ");
		buf.append("inner join experiment e on e.id = t.experiment ");
		boolean gotWhile = false;
		if (application != null) {
			buf.append(" where e.application = " + application.getID());
			gotWhile = true;
		}
		if (experiment != null) {
			if (gotWhile)
				buf.append(" and");
			else
				buf.append(" where");
			buf.append(" t.experiment = " + experiment.getID());
			gotWhile = true;
		}
		if (trials != null) {
			if (gotWhile)
				buf.append(" and t.id in (");
			else
				buf.append(" where t.id in (");
			Trial trial;
        	for(Enumeration en = trials.elements(); en.hasMoreElements() ;) {
				trial = (Trial) en.nextElement();
				buf.append(trial.getID());
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
			gotWhile = true;
		}
		if (nodes != null) {
			if (gotWhile)
				buf.append(" and p.node in (");
			else
				buf.append(" where p.node in (");
			Integer node;
        	for(Enumeration en = nodes.elements(); en.hasMoreElements() ;) {
				node = (Integer) en.nextElement();
				buf.append(node);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
			gotWhile = true;
		}
		if (contexts != null) {
			if (gotWhile)
				buf.append(" and p.context in (");
			else
				buf.append(" where p.context in (");
			Integer context;
        	for(Enumeration en = contexts.elements(); en.hasMoreElements() ;) {
				context = (Integer) en.nextElement();
				buf.append(context);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
			gotWhile = true;
		}
		if (threads != null) {
			if (gotWhile)
				buf.append(" and p.thread in (");
			else
				buf.append(" where p.thread in (");
			Integer thread;
        	for(Enumeration en = threads.elements(); en.hasMoreElements() ;) {
				thread = (Integer) en.nextElement();
				buf.append(thread);
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
			gotWhile = true;
		}
		if (userEvents != null) {
			if (gotWhile)
				buf.append(" and u.id in (");
			else
				buf.append(" where u.id in (");
			UserEvent userEvent;
        	for(Enumeration en = userEvents.elements(); en.hasMoreElements() ;) {
				userEvent = (UserEvent) en.nextElement();
				buf.append(userEvent.getUserEventID());
				if (en.hasMoreElements())
					buf.append(", ");
				else
					buf.append(") ");
			}
			gotWhile = true;
		}
		buf.append(" order by u.trial, p.node, p.context, p.thread, p.user_event");
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	while (resultSet.next() != false) {
				UserEventDataObject ueDO = new UserEventDataObject();
				ueDO.setUserEventIndexID(resultSet.getInt(1));
				ueDO.setNode(resultSet.getInt(2));
				ueDO.setContext(resultSet.getInt(3));
				ueDO.setThread(resultSet.getInt(4));
				ueDO.setSampleCount(resultSet.getInt(5));
				ueDO.setMaximumValue(resultSet.getDouble(6));
				ueDO.setMinimumValue(resultSet.getDouble(7));
				ueDO.setMeanValue(resultSet.getDouble(8));
				ueDO.setStandardDeviation(resultSet.getDouble(9));
				userEventData.addElement(ueDO);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
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
			whereClause = " where f.id = " + id;
			Vector functions = getFunctions(whereClause);
			if (functions.size() == 1) {
				function = (Function)functions.elementAt(0);
			} //else exception?
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
			Vector userEvents = getUserEvents(whereClause);
			if (userEvents.size() == 1) {
				userEvent = (UserEvent)userEvents.elementAt(0);
			} //else exception?
		}
		return userEvent;
	}
	

};

