/* 
   Name:       DatabaseAPI.java
   Author:     Robert Bell
  
  
   Description: API to the PerfDMF (Performance Database Management Framework).

   Things to do:
*/

package edu.uoregon.tau.dms.dss;

import edu.uoregon.tau.dms.database.*;

import java.util.*;
import java.sql.*;

//import java.sql.Result;
import java.util.Date;
import edu.uoregon.tau.dms.database.DB;

public class DatabaseAPI{

    //String configFile;
    //String password;


    //####################################
    //Constructor(s).
    //####################################
    public DatabaseAPI() { 
    }

    public void connect(String configFile) throws SQLException {
	connector = new ConnectionManager(configFile, false);
	connector.connect();
	db = connector.getDB();
    }


    public void connect(String configFile, String password) throws SQLException {
	connector = new ConnectionManager(configFile, password);
	connector.connect();
	db = connector.getDB();
    }


    public void disconnect() {
	connector.dbclose();
    }

    //####################################
    //End - Constructor(s).
    //####################################

    //####################################
    //Public section.
    //####################################
    public void setDB(DB db){
	this.db = db;}

    public DB getDB(){
	return this.db;}



//     String appFieldNames[];
//     int appFieldTypes[];

//     //    String expFields[];
//     //String trialFields[];

//     public void getMetaData() throws SQLException {

// 	DatabaseMetaData dbMeta = db.getMetaData();
	    
// 	ResultSet resultSet = dbMeta.getColumns(null, db.getSchemaPrefix(), "application", "%");
	    
// 	Vector nameList = new Vector();
// 	Vector typeList = new Vector();
// 	while (resultSet.next() != false) {

// 	    int ctype = resultSet.getInt("DATA_TYPE");
// 	    String cname = resultSet.getString("COLUMN_NAME");

// 	    // only integer and string types (for now)
// 	    // don't do name and id, we already know about them
// 	    if ((ctype == 4 || ctype == 12) && cname.compareTo("id") != 0 && cname.compareTo("name") != 0) {

// 		nameList.add(resultSet.getString("COLUMN_NAME"));
// 		typeList.add(new Integer(ctype));

// 		String typename = resultSet.getString("TYPE_NAME");
// 		System.out.println ("column: " + cname + ", type: " + ctype + ", typename: " + typename);

// 	    }


// 	}
// 	resultSet.close();



// 	appFieldNames = new String[nameList.size()];
// 	appFieldTypes = new int[typeList.size()];
// 	for (int i=0; i<typeList.size(); i++) {
// 	    appFieldNames[i] = (String) nameList.get(i);
// 	    appFieldTypes[i] = ((Integer)typeList.get(i)).intValue();
// 	}
//     }

//     public DataSessionIterator getApplicationList() throws SQLException {
// 	ResultSet resultSet = null;
// 	Vector applications = new Vector();
	
// 	this.getMetaData();

// 	StringBuffer buf = new StringBuffer("select id, name, ");

// 	for (int i=0; i<appFieldNames.length; i++) {
// 	    buf.append(appFieldNames[i]);
// 	    if (i != appFieldNames.length - 1) { // if not last
// 		buf.append(", ");
// 	    }
// 	}

// 	buf.append(" from ");
// 	buf.append(this.db.getSchemaPrefix());
// 	buf.append("application");

// 	if (db.getDBType().compareTo("oracle") == 0) {
// 	    buf.append(" order by dbms_lob.substr(name) asc");
// 	} else {
// 	    buf.append(" order by name asc ");
// 	}
	    
// 	System.out.println ("query: " + buf.toString());
// 	resultSet = db.executeQuery(buf.toString());	
// 	while (resultSet.next() != false) {
// 	    Application application = new Application(appFieldNames, appFieldTypes);

// 	    application.setID(resultSet.getInt(1));
// 	    application.setName(resultSet.getString(2));
		
// 	    for (int i=0; i<appFieldNames.length; i++) {

// 		if (appFieldTypes[i] == 4) { // integer type
// 		    application.setField(i, (new Integer(resultSet.getInt(appFieldNames[i]))).toString());
// 		}

// 		if (appFieldTypes[i] == 12) { // string type
// 		    application.setField(i, resultSet.getString(appFieldNames[i]));
// 		}
// 	    }
		
// 	    //Add the application.
// 	    applications.addElement(application);
// 	}
// 	//Cleanup resources.
// 	resultSet.close();

// 	return new DataSessionIterator(applications);
//     }


//     public int saveApplication(Application app) throws SQLException {

// 	boolean itExists = false;

// 	// First, determine whether it exists already (whether we are doing an insert or update)
// 	PreparedStatement statement = db.prepareStatement("SELECT name FROM application WHERE id = ?");
// 	statement.setInt(1, app.getID());
// 	ResultSet results = statement.executeQuery();
// 	while (results.next() != false) {
// 	    itExists = true;
// 	    break;
// 	}
// 	results.close();
// 	statement.close();

// 	// this needs to be done somewhere else, globally, for now, it's here
// 	this.getMetaData();

	


// 	StringBuffer buf = new StringBuffer();
// 	if (itExists) {
// 	    buf.append("UPDATE application SET name = ?");
// 	    for (int i=0; i<appFieldNames.length; i++) {
// 		buf.append(", " + appFieldNames[i] + " = ?");
// 	    }
// 	    buf.append(" WHERE id = ?");
// 	} else {
// 	    buf.append("INSERT INTO application(name");
// 	    for (int i=0; i<appFieldNames.length; i++) {
// 		buf.append(", " + appFieldNames[i]);
// 	    }
// 	    buf.append(") VALUES (name");
// 	    for (int i=0; i<appFieldNames.length; i++) {
// 		buf.append(", ?");
// 	    }
// 	    buf.append(")");
// 	}


// 	statement = db.prepareStatement(buf.toString());
	
// 	statement.setString(1, app.getName());

// 	for (int i=0; i<appFieldNames.length; i++) {
// 	    statement.setString(i+2, app.getField(i));
// 	}

// 	if (itExists) {
// 	    statement.setInt(2+app.getNumFields(), app.getID());
// 	}
// 	statement.executeUpdate();
// 	statement.close();

// 	int newApplicationID = 0;

// 	if (itExists) {
// 	    newApplicationID = app.getID();
// 	} else {
// 	    String tmpStr = new String();
// 	    if (db.getDBType().compareTo("mysql") == 0) {
// 		tmpStr = "select LAST_INSERT_ID();";
// 	    } else if (db.getDBType().compareTo("db2") == 0) {
// 		tmpStr = "select IDENTITY_VAL_LOCAL() FROM application";
// 	    } else if (db.getDBType().compareTo("oracle") == 0) {
// 		tmpStr = "SELECT application_id_seq.currval FROM DUAL";
// 	    } else { // postgresql 
// 		tmpStr = "select currval('application_id_seq');";
// 	    }
// 	    newApplicationID = Integer.parseInt(db.getDataItem(tmpStr));
// 	}
// 	return newApplicationID;
//     }

//     public DataSessionIterator getExperimentList(int applicationID) throws DatabaseAPIException{
// 	ResultSet resultSet = null;
// 	Vector experiments = new Vector();

// 	try{
// 	    StringBuffer buf = new StringBuffer();
// 	    buf.append("select id, application, name, system_name, ");
// 	    buf.append("system_machine_type, system_arch, system_os, ");
// 	    buf.append("system_memory_size, system_processor_amt, ");
// 	    buf.append("system_l1_cache_size, system_l2_cache_size, ");
// 	    buf.append("system_userdata, ");
// 	    buf.append("configure_prefix, configure_arch, configure_cpp, ");
// 	    buf.append("configure_cc, configure_jdk, configure_profile, ");
// 	    buf.append("configure_userdata, ");
// 	    buf.append("compiler_cpp_name, compiler_cpp_version, ");
// 	    buf.append("compiler_cc_name, compiler_cc_version, ");
// 	    buf.append("compiler_java_dirpath, compiler_java_version, ");
// 	    buf.append("compiler_userdata, userdata from experiment ");
// 	    buf.append("where application = " + applicationID);
// 	    buf.append(" order by name asc ");
	    
	
// 	    resultSet = db.executeQuery(buf.toString());	
// 	    while (resultSet.next() != false) {
// 		Experiment experiment = new Experiment();
// 		experiment.setID(resultSet.getInt(1));
// 		experiment.setApplicationID(resultSet.getInt(2));
// 		experiment.setName(resultSet.getString(3));
// 		experiment.setSystemName(resultSet.getString(4));
// 		experiment.setSystemMachineType(resultSet.getString(5));
// 		experiment.setSystemArch(resultSet.getString(6));
// 		experiment.setSystemOS(resultSet.getString(7));
// 		experiment.setSystemMemorySize(resultSet.getString(8));
// 		experiment.setSystemProcessorAmount(resultSet.getString(9));
// 		experiment.setSystemL1CacheSize(resultSet.getString(10));
// 		experiment.setSystemL2CacheSize(resultSet.getString(11));
// 		experiment.setSystemUserData(resultSet.getString(12));
// 		experiment.setConfigurationPrefix(resultSet.getString(13));
// 		experiment.setConfigurationArchitecture(resultSet.getString(14));
// 		experiment.setConfigurationCpp(resultSet.getString(15));
// 		experiment.setConfigurationCc(resultSet.getString(16));
// 		experiment.setConfigurationJdk(resultSet.getString(17));
// 		experiment.setConfigurationProfile(resultSet.getString(18));
// 		experiment.setConfigurationUserData(resultSet.getString(19));
// 		experiment.setCompilerCppName(resultSet.getString(20));
// 		experiment.setCompilerCppVersion(resultSet.getString(21));
// 		experiment.setCompilerCcName(resultSet.getString(22));
// 		experiment.setCompilerCcVersion(resultSet.getString(23));
// 		experiment.setCompilerJavaDirpath(resultSet.getString(24));
// 		experiment.setCompilerJavaVersion(resultSet.getString(25));
// 		experiment.setCompilerUserData(resultSet.getString(26));
// 		experiment.setUserData(resultSet.getString(27));

// 		//Add the experiment.
// 		experiments.addElement(experiment);
// 	    }
// 	    //Cleanup resources.
// 	    resultSet.close();
// 	}
// 	catch(Exception e){
// 	    throw new DatabaseAPIException(e);
// 	}
// 	return new DataSessionIterator(experiments);
//     }

//     public DataSessionIterator getTrialList(int experiementID) throws DatabaseAPIException{
// 	ResultSet resultSet = null;
// 	Vector trials = new Vector();

// 	try{
// 	    StringBuffer buf = new StringBuffer();
// 	    buf.append("select t.id, t.experiment, e.application, ");
// 	    buf.append("t.time, t.problem_definition, t.node_count, ");
// 	    buf.append("t.contexts_per_node, t.threads_per_context, ");
// 	    buf.append("t.name, t.userdata ");
// 	    buf.append("from trial t inner join experiment e ");
// 	    buf.append("on t.experiment = e.id ");
// 	    buf.append("where t.experiment = " + experiementID);
// 	    buf.append(" order by t.node_count, t.contexts_per_node, t.threads_per_context, t.id ");
	    
// 	    resultSet = db.executeQuery(buf.toString());	
// 	    while(resultSet.next() != false){
// 		Trial trial = new Trial();
// 		trial.setID(resultSet.getInt(1));
// 		trial.setExperimentID(resultSet.getInt(2));
// 		trial.setApplicationID(resultSet.getInt(3));
// 		trial.setTime(resultSet.getString(4));
// 		trial.setProblemDefinition(resultSet.getString(5));
// 		trial.setNodeCount(resultSet.getInt(6));
// 		trial.setNumContextsPerNode(resultSet.getInt(7));
// 		trial.setNumThreadsPerContext(resultSet.getInt(8));
// 		trial.setName(resultSet.getString(9));
// 		trial.setUserData(resultSet.getString(10));

// 		//Add the trial.
// 		trials.addElement(trial);
// 	    }
// 	    //Cleanup resources.
// 	    resultSet.close();
// 	}
// 	catch(Exception e){
// 	    throw new DatabaseAPIException(e);
// 	}
// 	return new DataSessionIterator(trials);
//     }
    
    
//     // gets the metric data for the trial
//     private DataSessionIterator getMetricList(int trialID) throws DatabaseAPIException{
// 	ResultSet resultSet = null;
// 	Vector metrics = new Vector();

// 	try{
// 	    StringBuffer buf = new StringBuffer();
// 	    buf.append("select id, name ");
// 	    buf.append("from metric where trial = " + trialID);
// 	    buf.append(" order by id ");

// 	    while (resultSet.next() != false) {
// 		Metric metric = new Metric();
// 		metric.setID(resultSet.getInt(1));
// 		metric.setName(resultSet.getString(2));
// 		metric.setTrialID(trialID);
		
// 		//Add the metric.
// 		metrics.addElement(metric);
// 	    }
// 	    //Cleanup resources.
// 	    resultSet.close();
// 	}
// 	catch(Exception e){
// 	    throw new DatabaseAPIException(e);
// 	}
// 	return new DataSessionIterator(metrics);
//     }




//     // returns a ListIterator of IntervalEvents
//     public ListIterator getIntervalEvents(int trialID) throws SQLException {

// 	StringBuffer buf = new StringBuffer();
// 	buf.append("SELECT id, name, group_name, trial ");
// 	buf.append("FROM interval_event WHERE trial = " + trialID);


// 	/*	if (db.getDBType().compareTo("oracle") == 0) {
// 	    buf.append(" order by dbms_lob.substr(name) asc");
// 	} else {
// 	    buf.append(" order by name asc ");
// 	    }*/


// 	Vector events = new Vector();

// 	ResultSet resultSet = db.executeQuery(buf.toString());	
// 	IntervalEvent tmpIntervalEvent = null;
// 	while (resultSet.next() != false) {
// 	    IntervalEvent event = new IntervalEvent(dataSession);
// 	    event.setID(resultSet.getInt(1));
// 	    event.setName(resultSet.getString(2));
// 	    event.setGroup(resultSet.getString(3));
// 	    event.setTrialID(resultSet.getInt(4));
// 	    events.addElement(event);
// 	}
// 	resultSet.close(); 

// 	/*
// 	if (intervalEventHash == null)
// 	    intervalEventHash = new Hashtable();
// 	IntervalEvent func;
// 	for (Enumeration en = intervalEvents.elements(); en.hasMoreElements() ;) {
// 	    func = (IntervalEvent) en.nextElement();
// 	    intervalEventHash.put(new Integer(func.getID()),fun);
// 	}
// 	*/
// 	return new DataSessionIterator(events);
//     }


//     public Vector getIntervalEventData(IntervalEvent intervalEvent) throws SQLException {

// 	StringBuffer buf = new StringBuffer();
// 	buf.append("select p.interval_event, p.metric, p.node, p.context, p.thread, ");
// 	buf.append("p.inclusive_percentage, ");

// 	if (db.getDBType().compareTo("oracle") == 0) {
// 	    buf.append("p.inclusive, p.exclusive_percentage, p.excl, ");
// 	} else {
// 	    buf.append("p.inclusive, p.exclusive_percentage, p.exclusive, ");
// 	}

// 	buf.append("p.call, p.subroutines, p.inclusive_per_call ");
// 	buf.append("from interval_event e inner join interval_location_profile p ");
// 	buf.append("on e.id = p.interval_event ");

// 	buf.append(" WHERE e.trial = " + trial.getID());

// 	buf.append(whereClause);
// 	buf.append(" order by p.interval_event, p.node, p.context, p.thread, p.metric ");
// 	// System.out.println(buf.toString());

// 	int size = 0;
// 	Vector intervalLocationProfiles = new Vector();
// 	// get the results
// 	try {
//             ResultSet resultSet = db.executeQuery(buf.toString());
// 	    while (resultSet.next() != false) {
// 		int metricIndex = 0;
// 		IntervalLocationProfile intervalLocationProfile = new IntervalLocationProfile();
//                 intervalLocationProfile.setIntervalEventID(resultSet.getInt(1));
//                 intervalLocationProfile.setNode(resultSet.getInt(3));
//                 intervalLocationProfile.setContext(resultSet.getInt(4));
//                 intervalLocationProfile.setThread(resultSet.getInt(5));
//                 intervalLocationProfile.setInclusivePercentage(metricIndex, resultSet.getDouble(6));
// 		intervalLocationProfile.setInclusive(metricIndex, resultSet.getDouble(7));
// 		intervalLocationProfile.setExclusivePercentage(metricIndex, resultSet.getDouble(8));
//                 intervalLocationProfile.setExclusive(metricIndex, resultSet.getDouble(9));
//                 intervalLocationProfile.setNumCalls((int)(resultSet.getDouble(10)));
//                 intervalLocationProfile.setNumSubroutines((int)(resultSet.getDouble(11)));
//                 intervalLocationProfile.setInclusivePerCall(metricIndex, resultSet.getDouble(12));
// 		for (int i = 1 ; i < metricCount ; i++) {
// 		    if (resultSet.next() == false) { break; }
// 		    metricIndex++;
// 		    intervalLocationProfile.setInclusivePercentage(metricIndex, resultSet.getDouble(6));
// 		    intervalLocationProfile.setInclusive(metricIndex, resultSet.getDouble(7));
// 		    intervalLocationProfile.setExclusivePercentage(metricIndex, resultSet.getDouble(8));
// 		    intervalLocationProfile.setExclusive(metricIndex, resultSet.getDouble(9));
// 		    intervalLocationProfile.setInclusivePerCall(metricIndex, resultSet.getDouble(12));
// 		}
// 		intervalLocationProfiles.addElement(intervalLocationProfile);
// 	    }
// 	    resultSet.close(); 
// 	return (intervalLocationProfiles);


//     }

//     // gets the mean & total data for a intervalEvent
//     public void getIntervalEventMeanAndTotal(IntervalEvent intervalEvent) throws SQLException {

// 	// create a string to hit the database
// 	StringBuffer buf = new StringBuffer();
// 	buf.append("select ms.interval_event, ");
// 	buf.append("ms.inclusive_percentage, ms.inclusive, ");

// 	if (db.getDBType().compareTo("oracle") == 0) {
// 	    buf.append("ms.exclusive_percentage, ms.excl, ");
// 	} else {
// 	    buf.append("ms.exclusive_percentage, ms.exclusive, ");
// 	}
	
// 	buf.append("ms.call, ms.subroutines, ms.inclusive_per_call, ");
// 	buf.append("ms.metric, ");
// 	buf.append("ts.inclusive_percentage, ts.inclusive, ");

// 	if (db.getDBType().compareTo("oracle") == 0) {
// 	    buf.append("ts.exclusive_percentage, ts.excl, ");
// 	} else {
// 	    buf.append("ts.exclusive_percentage, ts.exclusive, ");
// 	}
// 	buf.append("ts.call, ts.subroutines, ts.inclusive_per_call ");
// 	buf.append("from interval_mean_summary ms inner join ");
// 	buf.append("interval_total_summary ts ");
// 	buf.append("on ms.interval_event = ts.interval_event ");
// 	buf.append("and ms.metric = ts.metric ");

// 	buf.append(" WHERE ms.interval_event = " + intervalEvent.getID());
// 	if (metrics != null && metrics.size() > 0) {
// 	    buf.append(" AND ms.metric in (");
// 	    Metric metric;
// 	    for(Enumeration en = metrics.elements(); en.hasMoreElements() ;) {
// 		metric = (Metric) en.nextElement();
// 		buf.append(metric.getID());
// 		if (en.hasMoreElements())
// 		    buf.append(", ");
// 		else
// 		    buf.append(") ");
// 	    }
// 	}

// 	buf.append(" order by ms.interval_event, ms.metric");
// 	// System.out.println(buf.toString());

// 	// get the results
// 	ResultSet resultSet = db.executeQuery(buf.toString());	
// 	int metricIndex = 0;
// 	IntervalLocationProfile eMS = new IntervalLocationProfile();
// 	IntervalLocationProfile eTS = new IntervalLocationProfile();
// 	while (resultSet.next() != false) {
// 	    // get the mean summary data
// 	    eMS.setIntervalEventID(resultSet.getInt(1));
// 	    eMS.setInclusivePercentage(metricIndex, resultSet.getDouble(2));
// 	    eMS.setInclusive(metricIndex, resultSet.getDouble(3));
// 	    eMS.setExclusivePercentage(metricIndex, resultSet.getDouble(4));
// 	    eMS.setExclusive(metricIndex, resultSet.getDouble(5));
// 	    eMS.setNumCalls((int)(resultSet.getDouble(6)));
// 	    eMS.setNumSubroutines((int)(resultSet.getDouble(7)));
// 	    eMS.setInclusivePerCall(metricIndex, resultSet.getDouble(8));
// 	    // get the total summary data
// 	    eTS.setInclusivePercentage(metricIndex, resultSet.getDouble(10));
// 	    eTS.setInclusive(metricIndex, resultSet.getDouble(11));
// 	    eTS.setExclusivePercentage(metricIndex, resultSet.getDouble(12));
// 	    eTS.setExclusive(metricIndex, resultSet.getDouble(13));
// 	    eTS.setNumCalls((int)(resultSet.getDouble(14)));
// 	    eTS.setNumSubroutines((int)(resultSet.getDouble(15)));
// 	    eTS.setInclusivePerCall(metricIndex, resultSet.getDouble(16));
// 	    metricIndex++;
// 	}
// 	intervalEvent.setMeanSummary(eMS);
// 	intervalEvent.setTotalSummary(eTS);
// 	resultSet.close(); 
//     }


    /*
    //This version just does the usual ... build a list of objects as the old api did.
    //We'll try and integrate later.
    public DataSessionIterator getIntervalEvents() throws DatabaseAPIException{
	ResultSet resultSet = null;
	Vector intervalEvents = new Vector();

	try{
	    while (resultSet.next() != false) {
		//Add the interval event.
		//intervalEvents.addElement(intervalEvent);
	    }
	    //Cleanup resources.
	    resultSet.close();
	}
	catch(Exception e){
	    throw new DatabaseAPIException(e);
	}
	return new DataSessionIterator(intervalEvents);
    }
    */

    /*
    public int getNumberOfMetrics(int trialID) {
	StringBuffer buf = new StringBuffer();
	buf.append("SELECT id, name ");
	buf.append("FROM metric ");
	buf.append("WHERE trial = ");
	buf.append(trialID);
	buf.append(" ORDER BY id ");
	// System.out.println(buf.toString());

	// get the results
	try {
	    ResultSet resultSet = db.executeQuery(buf.toString());	
	    int counter = 0;
	    while (resultSet.next() != false) {
		counter++;
	    }
	    resultSet.close();
	    return counter;
	} catch (Exception ex) {
	    ex.printStackTrace();
	    return -1;
	}
    }
*/


//     // gets the metric data for the trial
//     private Vector getMetrics(int trialID) {
// 	// create a string to hit the database
// 	StringBuffer buf = new StringBuffer();
// 	buf.append("select id, name ");
// 	buf.append("from metric ");
// 	buf.append("where trial = ");
// 	buf.append(trialID);
// 	buf.append(" order by id ");
// 	// System.out.println(buf.toString());

// 	Vector metrics;

// 	// get the results
// 	try {
// 	    ResultSet resultSet = db.executeQuery(buf.toString());	
// 	    while (resultSet.next() != false) {
// 		Metric tmp = new Metric();
// 		tmp.setID(resultSet.getInt(1));
// 		tmp.setName(resultSet.getString(2));
// 		tmp.setTrialID(getID());
// 		metrics.addElement(tmp);
// 	    }
// 	    resultSet.close(); 
// 	    return metrics;
// 	} catch (Exception ex) {
// 	    ex.printStackTrace();
// 	    return null;
// 	}
//     }


    //####################################
    //End - Public section.
    //####################################

    //####################################
    //Protected section.
    //####################################
    //####################################
    //End - Protected section.
    //####################################

    //####################################
    //Private section.
    //####################################
    //####################################
    //End - Private section.
    //####################################


    //####################################
    //Instance data.
    //####################################
    private DB db = null;
    ConnectionManager connector;
    //####################################
    //End - Instance data.
    //####################################
}

class DatabaseAPIException extends Exception{
    
    public DatabaseAPIException(String exceptionString){
	super(exceptionString);}

    public DatabaseAPIException(Exception e){
	super(e.toString());}
}
