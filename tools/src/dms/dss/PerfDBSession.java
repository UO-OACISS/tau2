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

	public PerfDBSession () {
		super();
		// initialize the connection to the database,
		// using the configuration settings.
		connector = new perfdb.ConnectionManager();
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
	public Vector getAppList() {
		Vector apps = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct appid, appname, version, description from applications");
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
		
		return apps;
	}

	// returns Vector of Experiment objects
	public Vector getExpList() {
		Vector exps = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct expid, appid from experiments ");
		if (application != null)
			buf.append("where appid = " + application.getID());
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
		
		return exps;
	}

	// returns Vector of Trial objects
	public Vector getTrialList() {
		Vector trials = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct t.trialid, t.expid, e.appid, t.time, t.metric, t.problemsize, t.nodenum, t.contextpnode, t.threadpcontext, t.xmlfileid from trials t inner join experiments e on t.expid = e.expid ");
		if (application != null) {
			buf.append("where e.appid = " + application.getID());
			if (experiment != null) {
				buf.append(" and t.expid = " + experiment.getID());
			}
		} else {
			if (experiment != null) {
				buf.append("where t.expid = " + experiment.getID());
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
				trial.setMetric(resultSet.getString(5));
				trial.setProblemSize(resultSet.getInt(6));
				trial.setNumNodes(resultSet.getInt(7));
				trial.setNumContextsPerNode(resultSet.getInt(8));
				trial.setNumThreadsPerContext(resultSet.getInt(9));
				trial.setXMLFileID(resultSet.getInt(10));
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
		// create a string to hit the database
		Application app = null;
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct appid, appname, version, description from applications ");
		buf.append("where appid = " + id);
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
		return app;
	}

	public Application setApplication(String name /*, String version = null */) {
		// create a string to hit the database
		Application app = null;
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct appid, appname, version, description from applications");
		buf.append(" where appname = '" + name + "'");
		/* 
		if (version != null) {
			buf.append(" and version = " + version);
		}
		*/
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
		return app;
	}

	// set the Experiment for this session
	public Experiment setExperiment(int id) {
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		Experiment exp = null;
		buf.append("select distinct expid, appid from experiments");
		if (application != null) {
			buf.append(" where appid = " + application.getID());
			buf.append(" and expid = " + id);
		} else {
			buf.append(" where expid = " + id);
		}
		// System.out.println(buf.toString());

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
		
		return exp;
	}

	// set the Trial for this session
	public Trial setTrial(int id) {
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		Trial trial = null;
		buf.append("select distinct t.trialid, t.expid, e.appid, t.time, t.metric, t.problemsize, t.nodenum, t.contextpnode, t.threadpcontext, t.xmlfileid from trials t inner join experiments e on t.expid = e.expid");
		if (application != null) {
			buf.append(" where e.appid = " + application.getID());
			if (experiment != null) {
				buf.append(" and t.expid = " + experiment.getID());
			}
			buf.append(" and t.expid = " + id);
		} else {
			if (experiment != null) {
				buf.append(" where t.expid = " + experiment.getID());
			} else {
				buf.append(" where t.expid = " + id);
			}
		}
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
				trial.setMetric(resultSet.getString(5));
				trial.setProblemSize(resultSet.getInt(6));
				trial.setNumNodes(resultSet.getInt(7));
				trial.setNumContextsPerNode(resultSet.getInt(8));
				trial.setNumThreadsPerContext(resultSet.getInt(9));
				trial.setXMLFileID(resultSet.getInt(10));
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return trial;
	}

	public int getNumberOfNodes() {
		// get the number of nodes from the trials table
		int numNodes = 0;

		if (trials == null) {
			// create a string to hit the database
			StringBuffer buf = new StringBuffer();
			buf.append("select sum(t.nodenum) from trials t inner join experiments e on t.expid = e.expid ");
			if (application != null) {
				buf.append("where e.appid = " + application.getID());
				if (experiment != null) {
					buf.append(" and t.expid = " + experiment.getID());
				}
			} else {
				if (experiment != null) {
					buf.append("where t.expid = " + experiment.getID());
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
				numNodes += trial.getNumNodes();
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
			buf.append("select sum(t.contextpnode) from trials t inner join experiments e on t.expid = e.expid ");
			if (application != null) {
				buf.append("where e.appid = " + application.getID());
				if (experiment != null) {
					buf.append(" and t.expid = " + experiment.getID());
				}
			} else {
				if (experiment != null) {
					buf.append("where t.expid = " + experiment.getID());
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
			buf.append("select sum(t.threadpcontext) from trials t inner join experiments e on t.expid = e.expid ");
			if (application != null) {
				buf.append("where e.appid = " + application.getID());
				if (experiment != null) {
					buf.append(" and t.expid = " + experiment.getID());
				}
			} else {
				if (experiment != null) {
					buf.append("where t.expid = " + experiment.getID());
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
	public Vector getFunctions() {
		Vector funs = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct f.funindexid, f.funid, f.funname, f.trialid, ");
		buf.append("e.expid, e.appid, ");
		buf.append("ms.inclperc, ms.incl, ms.exclperc, ms.excl, ms.call, ms.subrs, ms.inclpcall, ");
		buf.append("ts.inclperc, ts.incl, ts.exclperc, ts.excl, ts.call, ts.subrs, ts.inclpcall ");
		buf.append("from funindex f inner join trials t on f.trialid = t.trialid ");
		buf.append("inner join experiments e on e.expid = t.expid ");
		buf.append("inner join meansummary ms on f.funindexid = ms.funindexid ");
		buf.append("inner join totalsummary ts on f.funindexid = ts.funindexid ");
		boolean gotWhile = false;
		if (application != null) {
			buf.append(" where e.appid = " + application.getID());
			gotWhile = true;
		}
		if (experiment != null) {
			if (gotWhile)
				buf.append(" and");
			else
				buf.append(" where");
			buf.append(" t.expid = " + experiment.getID());
			gotWhile = true;
		}
		if (trials != null) {
			if (gotWhile)
				buf.append(" and");
			else
				buf.append(" where");
			Trial trial;
        	for(Enumeration en = trials.elements(); en.hasMoreElements() ;) {
				trial = (Trial) en.nextElement();
				buf.append(" t.trialid = " + trial.getID());
				if (en.hasMoreElements())
					buf.append(" and");
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
				fun.setTrialID(resultSet.getInt(4));
				fun.setExperimentID(resultSet.getInt(5));
				fun.setApplicationID(resultSet.getInt(6));
				// get the mean summary data
				FunctionDataObject funMS = new FunctionDataObject();
				funMS.setInclusivePercentage(resultSet.getDouble(7));
				funMS.setInclusive(resultSet.getDouble(8));
				funMS.setExclusivePercentage(resultSet.getDouble(9));
				funMS.setExclusive(resultSet.getDouble(10));
				funMS.setNumCalls((int)(resultSet.getDouble(11)));
				funMS.setNumSubroutines((int)(resultSet.getDouble(12)));
				funMS.setInclusivePerCall(resultSet.getDouble(13));
				fun.setMeanSummary(funMS);
				// get the total summary data
				FunctionDataObject funTS = new FunctionDataObject();
				funTS.setInclusivePercentage(resultSet.getDouble(14));
				funTS.setInclusive(resultSet.getDouble(15));
				funTS.setExclusivePercentage(resultSet.getDouble(16));
				funTS.setExclusive(resultSet.getDouble(17));
				funTS.setNumCalls((int)(resultSet.getDouble(18)));
				funTS.setNumSubroutines((int)(resultSet.getDouble(19)));
				funTS.setInclusivePerCall(resultSet.getDouble(20));
				fun.setTotalSummary(funTS);
				funs.addElement(fun);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return funs;
	}

	public Vector getUserEvents() {
		return new Vector();
	}

	public Function setFunction(int id) {
		// create a string to hit the database
		Function fun = null;
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct f.funindexid, f.funid, f.funname, f.trialid, ");
		buf.append("e.expid, e.appid, ");
		buf.append("ms.inclperc, ms.incl, ms.exclperc, ms.excl, ms.call, ms.subrs, ms.inclpcall, ");
		buf.append("ts.inclperc, ts.incl, ts.exclperc, ts.excl, ts.call, ts.subrs, ts.inclpcall ");
		buf.append("from funindex f inner join trials t on f.trialid = t.trialid ");
		buf.append("inner join experiments e on e.expid = t.expid ");
		buf.append("inner join meansummary ms on f.funindexid = ms.funindexid ");
		buf.append("inner join totalsummary ts on f.funindexid = ts.funindexid ");
		buf.append(" where f.funindexid = " + id);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	if (resultSet.next() != false) {
				fun = new Function();
				fun.setIndexID(resultSet.getInt(1));
				fun.setFunctionID(resultSet.getInt(2));
				fun.setName(resultSet.getString(3));
				fun.setTrialID(resultSet.getInt(4));
				fun.setExperimentID(resultSet.getInt(5));
				fun.setApplicationID(resultSet.getInt(6));
				// get the mean summary data
				FunctionDataObject funMS = new FunctionDataObject();
				funMS.setInclusivePercentage(resultSet.getDouble(7));
				funMS.setInclusive(resultSet.getDouble(8));
				funMS.setExclusivePercentage(resultSet.getDouble(9));
				funMS.setExclusive(resultSet.getDouble(10));
				funMS.setNumCalls((int)(resultSet.getDouble(11)));
				funMS.setNumSubroutines((int)(resultSet.getDouble(12)));
				funMS.setInclusivePerCall(resultSet.getDouble(13));
				fun.setMeanSummary(funMS);
				// get the total summary data
				FunctionDataObject funTS = new FunctionDataObject();
				funTS.setInclusivePercentage(resultSet.getDouble(14));
				funTS.setInclusive(resultSet.getDouble(15));
				funTS.setExclusivePercentage(resultSet.getDouble(16));
				funTS.setExclusive(resultSet.getDouble(17));
				funTS.setNumCalls((int)(resultSet.getDouble(18)));
				funTS.setNumSubroutines((int)(resultSet.getDouble(19)));
				funTS.setInclusivePerCall(resultSet.getDouble(20));
				fun.setTotalSummary(funTS);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		return fun;
	}

	public void getFunctionData() {
		Vector functionData = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct p.inclperc, p.incl, p.exclperc, p.excl, ");
		buf.append("p.call, p.subrs, p.inclpcall, ");
		buf.append("t.trialid, l.nodeid, l.contextid, l.threadid, l.funindexid ");
		buf.append("from pprof p ");
		buf.append("inner join locationindex l on p.locid = l.locid ");
		buf.append("inner join funindex f on f.funindexid = l.funindexid ");
		buf.append("inner join trials t on f.trialid = t.trialid ");
		buf.append("inner join experiments e on e.expid = t.expid ");
		boolean gotWhile = false;
		if (application != null) {
			buf.append(" where e.appid = " + application.getID());
			gotWhile = true;
		}
		if (experiment != null) {
			if (gotWhile)
				buf.append(" and");
			else
				buf.append(" where");
			buf.append(" t.expid = " + experiment.getID());
			gotWhile = true;
		}
		if (trials != null) {
			if (gotWhile)
				buf.append(" and t.trialid in (");
			else
				buf.append(" where t.trialid in (");
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
		if (nodes != null) {
			if (gotWhile)
				buf.append(" and l.nodeid in (");
			else
				buf.append(" where l.nodeid in (");
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
			if (gotWhile)
				buf.append(" and l.contextid in (");
			else
				buf.append(" where l.contextid in (");
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
			if (gotWhile)
				buf.append(" and l.threadid in (");
			else
				buf.append(" where l.threadid in (");
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
			if (gotWhile)
				buf.append(" and l.funindexid in (");
			else
				buf.append(" where l.funindexid in (");
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
		buf.append(" order by t.trialid, l.nodeid, l.contextid, l.threadid, l.funindexid");
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
				functionData.addElement(funDO);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return;
		}

		// print the data
		/*
		FunctionDataObject function = null;
        for(Enumeration en = functionData.elements(); en.hasMoreElements() ;)
		{
			function = (FunctionDataObject) en.nextElement();
			System.out.println ("Function Data = " + function.getInclusivePercentage() + ", " + function.getInclusive() + ", " + function.getExclusivePercentage() + ", " + function.getExclusive() + ", " + function.getNumCalls() + ", " + function.getNumSubroutines() + ", " + function.getInclusivePerCall());
		}
		*/
	}
	
	public Vector getUserEventData() {
		return new Vector();
	}
	

};

