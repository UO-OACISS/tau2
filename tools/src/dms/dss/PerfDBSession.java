package dms.dss;

import perfdb.util.dbinterface.*;
import perfdb.util.io.*;
import perfdb.dbmanager.*;
import java.io.*;
import java.util.*;
import java.net.*;
import java.sql.*;

public class PerfDBSession extends DataSession {

    private DB db = null;
    private perfdb.ConnectionManager connector;

	public PerfDBSession (String configFileName) {
		super();
		// initialize the connection to the database,
		// using the configuration settings.
		connector = new perfdb.ConnectionManager(configFileName);
	}

	public void open () {
		connector.connect();
		db = connector.getDB();
	}

	public void close () {
		connector.dbclose();
	}

    public perfdb.ConnectionManager getConnector(){
		return connector;
    }

	// returns Vector of Application objects
	public ListIterator getAppList() {
		Vector apps = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct id, name, version, description from application");
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
				apps.addElement(app);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return new DataSessionIterator(apps);
	}

	// returns Vector of Experiment objects
	public ListIterator getExpList() {
		Vector exps = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct id, application from experiment ");
		if (application != null)
			buf.append("where application = " + application.getID());
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	while (resultSet.next() != false) {
				Experiment exp = new Experiment();
				exp.setID(resultSet.getInt(1));
				exp.setApplicationID(resultSet.getInt(2));
				exps.addElement(exp);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return new DataSessionIterator(exps);
	}

	// returns Vector of Trial objects
	public ListIterator getTrialList() {
		Vector trials = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct t.id, t.experiment, e.application, t.time, t.problem_size, t.node_count, t.contexts_per_node, t.threads_per_context from trial t inner join experiment e on t.experiment = e.id ");
		if (application != null) {
			buf.append("where e.application = " + application.getID());
			if (experiment != null) {
				buf.append(" and t.experiment = " + experiment.getID());
			}
		} else {
			if (experiment != null) {
				buf.append("where t.experiment = " + experiment.getID());
			}
		}
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
		
		return new DataSessionIterator(trials);
	}

	// set the Application for this session
	public Application setApplication(int id) {
		// create a string to hit the database
		Application app = null;
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct id, name, version, description from application ");
		buf.append("where id = " + id);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	if (resultSet.next() != false) {
				app = new Application();
				app.setID(resultSet.getInt(1));
				app.setName(resultSet.getString(2));
				app.setVersion(resultSet.getString(3));
				app.setDescription(resultSet.getString(4));
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		this.application = app;
		return app;
	}

	public Application setApplication(String name, String version) {
		// create a string to hit the database
		Application app = null;
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct id, name, version, description from application");
		buf.append(" where name = '" + name + "'");
		if (version != null) {
			buf.append(" and version = " + version);
		}
		System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	if (resultSet.next() != false) {
				app = new Application();
				app.setID(resultSet.getInt(1));
				app.setName(resultSet.getString(2));
				app.setVersion(resultSet.getString(3));
				app.setDescription(resultSet.getString(4));
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		this.application = app;
		return app;
	}

	// set the Experiment for this session
	public Experiment setExperiment(int id) {
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		Experiment exp = null;
		buf.append("select distinct id, application from experiment");
		if (application != null) {
			buf.append(" where application = " + application.getID());
			buf.append(" and id = " + id);
		} else {
			buf.append(" where id = " + id);
		}
		System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	if (resultSet.next() != false) {
				exp = new Experiment();
				exp.setID(resultSet.getInt(1));
				exp.setApplicationID(resultSet.getInt(2));
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		this.experiment = exp;
		return exp;
	}

	// set the Trial for this session
	public Trial setTrial(int id) {
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		Trial trial = null;
		buf.append("select distinct t.id, t.experiment, e.application, t.time, t.problem_size, t.node_count, t.contexts_per_node, t.threads_per_context from trial t inner join experiment e on t.experiment = e.id");
		buf.append(" where t.id = " + id);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	if (resultSet.next() != false) {
				trial = new Trial();
				trial.setID(resultSet.getInt(1));
				trial.setExperimentID(resultSet.getInt(2));
				trial.setApplicationID(resultSet.getInt(3));
				trial.setTime(resultSet.getString(4));
				trial.setProblemSize(resultSet.getInt(5));
				trial.setNodeCount(resultSet.getInt(6));
				trial.setNumContextsPerNode(resultSet.getInt(7));
				trial.setNumThreadsPerContext(resultSet.getInt(8));
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		this.trials.addElement(trial);
		return trial;
	}

	public int getNumberOfNodes() {
		// get the number of nodes from the trials table
		int numNodes = 0;

		if (trials == null) {
			// create a string to hit the database
			StringBuffer buf = new StringBuffer();
			buf.append("select sum(t.node_count) from trial t inner join experiment e on t.experiment = e.id ");
			if (application != null) {
				buf.append("where e.application = " + application.getID());
				if (experiment != null) {
					buf.append(" and t.experiment = " + experiment.getID());
				}
			} else {
				if (experiment != null) {
					buf.append("where t.experiment = " + experiment.getID());
				}
			}
			// System.out.println(buf.toString());
	
			// get the results
			try {
	    		ResultSet resultSet = db.executeQuery(buf.toString());	
	    		while (resultSet.next() != false) {
					numNodes += resultSet.getInt(1);
	    		}
				resultSet.close(); 
			}catch (Exception ex) {
	    		ex.printStackTrace();
	    		return 0;
			}
		}
		else
		{
			Trial trial;
			// add up the node totals in the trials
        	for(Enumeration en = trials.elements(); en.hasMoreElements() ;) {
				trial = (Trial) en.nextElement();
				numNodes += trial.getNodeCount();
			}
		}

		return numNodes;
	}

	public int getNumberOfContexts() {
		// get the number of contexts from the trials table
		int numContexts = 0;

		if (trials == null) {
			// create a string to hit the database
			StringBuffer buf = new StringBuffer();
			buf.append("select sum(t.contexts_per_node) from trial t inner join experiment e on t.experiment = e.id ");
			if (application != null) {
				buf.append("where e.application = " + application.getID());
				if (experiment != null) {
					buf.append(" and t.experiment = " + experiment.getID());
				}
			} else {
				if (experiment != null) {
					buf.append("where t.experiment = " + experiment.getID());
				}
			}
			// System.out.println(buf.toString());
	
			// get the results
			try {
	    		ResultSet resultSet = db.executeQuery(buf.toString());	
	    		while (resultSet.next() != false) {
					numContexts += resultSet.getInt(1);
	    		}
				resultSet.close(); 
			}catch (Exception ex) {
	    		ex.printStackTrace();
	    		return 0;
			}
		}
		else
		{
			Trial trial;
			// add up the node totals in the trials
        	for(Enumeration en = trials.elements(); en.hasMoreElements() ;) {
				trial = (Trial) en.nextElement();
				numContexts += trial.getNumContextsPerNode();
			}
		}

		return numContexts;
	}

	public int getNumberOfThreads() {
		// get the number of threads from the trials table
		int numThreads = 0;

		if (trials == null) {
			// create a string to hit the database
			StringBuffer buf = new StringBuffer();
			buf.append("select sum(t.threads_per_context) from trial t inner join experiment e on t.experiment = e.id ");
			if (application != null) {
				buf.append("where e.application = " + application.getID());
				if (experiment != null) {
					buf.append(" and t.experiment = " + experiment.getID());
				}
			} else {
				if (experiment != null) {
					buf.append("where t.experiment = " + experiment.getID());
				}
			}
			// System.out.println(buf.toString());
	
			// get the results
			try {
	    		ResultSet resultSet = db.executeQuery(buf.toString());	
	    		while (resultSet.next() != false) {
					numThreads += resultSet.getInt(1);
	    		}
				resultSet.close(); 
			}catch (Exception ex) {
	    		ex.printStackTrace();
	    		return 0;
			}
		}
		else
		{
			Trial trial;
			// add up the node totals in the trials
        	for(Enumeration en = trials.elements(); en.hasMoreElements() ;) {
				trial = (Trial) en.nextElement();
				numThreads += trial.getNumThreadsPerContext();
			}
		}

		return numThreads;
	}

	// returns a Vector of Functions
	public ListIterator getFunctions() {
		// clean out the old function name hashtable
		if (functionHash != null) 
			functionHash = null;

		// create a new function name hashtable
		functionHash = new Hashtable();

		Vector funs = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct f.id, f.function_number, f.name, ");
		buf.append("f.group_name, f.trial, t.experiment, e.application, ");
		buf.append("ms.inclusive_percentage, ms.inclusive, ms.exclusive_percentage, ms.exclusive, ms.call, ms.subroutines, ms.inclusive_per_call, ");
		buf.append("ts.inclusive_percentage, ts.inclusive, ts.exclusive_percentage, ts.exclusive, ts.call, ts.subroutines, ts.inclusive_per_call ");
		buf.append("from function f inner join trial t on f.trial = t.id ");
		buf.append("inner join experiment e on t.experiment = e.id ");
		buf.append("inner join interval_mean_summary ms on f.id = ms.function ");
		buf.append("inner join interval_total_summary ts on f.id = ts.function ");
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
		}
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
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
				funMS.setInclusivePercentage(resultSet.getDouble(8));
				funMS.setInclusive(resultSet.getDouble(9));
				funMS.setExclusivePercentage(resultSet.getDouble(10));
				funMS.setExclusive(resultSet.getDouble(11));
				funMS.setNumCalls((int)(resultSet.getDouble(12)));
				funMS.setNumSubroutines((int)(resultSet.getDouble(13)));
				funMS.setInclusivePerCall(resultSet.getDouble(14));
				fun.setMeanSummary(funMS);
				// get the total summary data
				FunctionDataObject funTS = new FunctionDataObject();
				funTS.setInclusivePercentage(resultSet.getDouble(15));
				funTS.setInclusive(resultSet.getDouble(16));
				funTS.setExclusivePercentage(resultSet.getDouble(17));
				funTS.setExclusive(resultSet.getDouble(18));
				funTS.setNumCalls((int)(resultSet.getDouble(19)));
				funTS.setNumSubroutines((int)(resultSet.getDouble(20)));
				funTS.setInclusivePerCall(resultSet.getDouble(21));
				fun.setTotalSummary(funTS);
				funs.addElement(fun);
				functionHash.put(new Integer(fun.getIndexID()), fun);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return new DataSessionIterator(funs);
	}

	public ListIterator getUserEvents() {
		return new DataSessionIterator(new Vector());
	}

	public Function setFunction(int id) {
		// create a string to hit the database
		Function fun = null;
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct f.id, f.function_number, f.name, ");
		buf.append("f.group_name, f.trial, t.experiment, e.application, ");
		buf.append("ms.inclusive_percentage, ms.inclusive, ms.exclusive_percentage, ms.exclusive, ms.call, ms.subroutines, ms.inclusive_per_call, ");
		buf.append("ts.inclusive_percentage, ts.inclusive, ts.exclusive_percentage, ts.exclusive, ts.call, ts.subroutines, ts.inclusive_per_call ");
		buf.append("from function f inner join trial t on f.trial = t.id ");
		buf.append("inner join experiment e on t.experiment = e.id ");
		buf.append("inner join interval_mean_summary ms on f.id = ms.function ");
		buf.append("inner join interval_total_summary ts on f.id = ts.function ");
		buf.append(" where f.id = " + id);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	if (resultSet.next() != false) {
				fun = new Function();
				fun.setIndexID(resultSet.getInt(1));
				fun.setFunctionID(resultSet.getInt(2));
				fun.setName(resultSet.getString(3));
				fun.setGroup(resultSet.getString(4));
				fun.setTrialID(resultSet.getInt(5));
				fun.setExperimentID(resultSet.getInt(6));
				fun.setApplicationID(resultSet.getInt(7));
				// get the mean summary data
				FunctionDataObject funMS = new FunctionDataObject();
				funMS.setInclusivePercentage(resultSet.getDouble(8));
				funMS.setInclusive(resultSet.getDouble(9));
				funMS.setExclusivePercentage(resultSet.getDouble(10));
				funMS.setExclusive(resultSet.getDouble(11));
				funMS.setNumCalls((int)(resultSet.getDouble(12)));
				funMS.setNumSubroutines((int)(resultSet.getDouble(13)));
				funMS.setInclusivePerCall(resultSet.getDouble(14));
				fun.setMeanSummary(funMS);
				// get the total summary data
				FunctionDataObject funTS = new FunctionDataObject();
				funTS.setInclusivePercentage(resultSet.getDouble(15));
				funTS.setInclusive(resultSet.getDouble(16));
				funTS.setExclusivePercentage(resultSet.getDouble(17));
				funTS.setExclusive(resultSet.getDouble(18));
				funTS.setNumCalls((int)(resultSet.getDouble(19)));
				funTS.setNumSubroutines((int)(resultSet.getDouble(20)));
				funTS.setInclusivePerCall(resultSet.getDouble(21));
				fun.setTotalSummary(funTS);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		return fun;
	}

	public UserEvent setUserEvent(int id) {
		UserEvent userEvent = null;
		// todo - fill in this function!
		return userEvent;
	}

	public ListIterator getFunctionData() {
		// get the hash of function names first
		if (functions == null)
			getFunctions();

		Vector functionData = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct p.inclusive_percentage, p.inclusive, p.exclusive_percentage, p.exclusive, ");
		buf.append("p.call, p.subroutines, p.inclusive_per_call, ");
		buf.append("f.trial, p.node, p.context, p.thread, p.function, f.metric ");
		buf.append("from interval_location_profile p ");
		buf.append("inner join function f on f.id = p.function ");
		buf.append("inner join trial t on f.trial = t.id ");
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
		buf.append(" order by f.trial, p.node, p.context, p.thread, p.function");
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	while (resultSet.next() != false) {
				FunctionDataObject funDO = new FunctionDataObject();
				funDO.setInclusivePercentage(resultSet.getDouble(1));
				funDO.setInclusive(resultSet.getDouble(2));
				funDO.setExclusivePercentage(resultSet.getDouble(3));
				funDO.setExclusive(resultSet.getDouble(4));
				funDO.setNumCalls((int)(resultSet.getDouble(5)));
				funDO.setNumSubroutines((int)(resultSet.getDouble(6)));
				funDO.setInclusivePerCall(resultSet.getDouble(7));
				funDO.setNode(resultSet.getInt(9));
				funDO.setContext(resultSet.getInt(10));
				funDO.setThread(resultSet.getInt(11));
				funDO.setFunctionIndexID(resultSet.getInt(12));
				funDO.setMetric(resultSet.getString(13));
				functionData.addElement(funDO);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}

		//Added by Robert.  Looks like the correct place to add it. :-)
		return new DataSessionIterator(functionData);

	}
	
	public ListIterator getUserEventData() {
		return new DataSessionIterator(new Vector());
	}
	

};

