package edu.uoregon.tau.dms.dss;

import edu.uoregon.tau.dms.database.DB;
import java.sql.SQLException;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.Vector;
import java.util.Enumeration;
import java.lang.String;

/**
 * Holds all the data for a trial in the database.
 * This object is returned by the DataSession class and all of its subtypes.
 * The Trial object contains all the information associated with
 * an trial from which the TAU performance data has been generated.
 * A trial is associated with one experiment and one application, and has one or more
 * interval_events and/or user events associated with it.  A Trial has information
 * related to the particular run, including the number of nodes used,
 * the number of contexts per node, the number of threads per context
 * and the metrics collected during the run.
 *
 * <P>CVS $Id: Trial.java,v 1.1 2004/05/05 17:43:39 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DataSession#getTrialList
 * @see		DataSession#setTrial
 * @see		Application
 * @see		Experiment
 * @see		Function
 * @see		UserEvent
 */
public class Trial {
    private int trialID;
    private int experimentID;
    private int applicationID;
    private String name;
    private String time;
    private int nodeCount;
    private int contextsPerNode;
    private int threadsPerContext;
    private Vector metric;
    private String userData;
    private String problemDefinition;
    protected DataSession dataSession = null;

    /**
     * Gets the unique identifier of the current trial object.
     *
     * @return	the unique identifier of the trial
     */
    public int getID () {
	return trialID;
    }
    
    /**
     * Gets the unique identifier for the experiment associated with this trial.
     *
     * @return	the unique identifier of the experiment
     */
    public int getExperimentID () {
	return experimentID;
    }
    
    /**
     * Gets the unique identifier for the application associated with this trial.
     *
     * @return	the unique identifier of the application
     */
    public int getApplicationID () {
	return applicationID;
    }

    /**
     * Gets the name of the current trial object.
     *
     * @return	the name of the trial
     */
    public String getName() {
	return name;
    }

    /**
     * Gets the time required to execute this trial.
     *
     * @return	time required to execute this trial.
     */
    public String getTime () {
	return this.time;
    }

    /**
     * Gets the problem description for this trial.
     *
     * @return	problem description for this trial.
     */
    public String getProblemDefinition () {
	return this.problemDefinition;
    }

    /**
     * Gets the data session for this trial.
     *
     * @return	data dession for this trial.
     */
    public DataSession getDataSession () {
	return this.dataSession;
    }

    /**
     * Gets the node count for this trial.
     *
     * @return	node count for this trial.
     */
    public int getNodeCount () {
	return this.nodeCount;
    }

    /**
     * Gets the context (per node) count for this trial.
     *
     * @return	context count for this trial.
     */
    public int getNumContextsPerNode () {
	return this.contextsPerNode;
    }

    /**
     * Gets the thread (per context) count for this trial.
     *
     * @return	thread count for this trial.
     */
    public int getNumThreadsPerContext () {
	return this.threadsPerContext;
    }

    /**
     * Does the same as getNodeCount, getNumContextsPerNode, getNumThreadsPerContext
     * combined.
     *
     * @return	an array containing nodeCount, contextsPerNode and  threadsPerContext in that order.
     */
    public int[] getMaxNCTNumbers(){
	int[] nct = new int[3];
	nct[0] = nodeCount;
	nct[1] = contextsPerNode;
	nct[2] = threadsPerContext;
	return nct;
    }
    
    /**
     * Gets the user data of the current trial object.
     *
     * @return	the user data of the trial
     */
    public String getUserData() {
	return userData;
    }

    /**
     * Gets the number of metrics collected in this trial.
     *
     * @return	metric count for this trial.
     */
    public int getMetricCount() {
	return this.metric.size();
    }

    /**
     * Gets the metrics collected in this trial.
     *
     * @return	metric vector
     */
    public Vector getMetrics() {
	return this.metric;
    }

    /**
     * Sets the unique ID associated with this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	id unique ID associated with this trial
     */
    public void setID (int id) {
	this.trialID = id;
    }

    /**
     * Sets the experiment ID associated with this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	experimentID experiment ID associated with this trial
     */
    public void setExperimentID (int experimentID) {
	this.experimentID = experimentID;
    }

    /**
     * Sets the application ID associated with this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	applicationID application ID associated with this trial
     */
    public void setApplicationID (int applicationID) {
	this.applicationID = applicationID;
    }

    /**
     * Sets the name of the current trial object.
     * <i>Note: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	name the trial name
     */
    public void setName(String name) {
	this.name = name;
    }

    /**
     * Sets the time to run this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	time execution time required to run this trial
     */
    public void setTime (String time) {
	this.time = time;
    }

    /**
     * Sets the problem description for this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	problemDefinition problem description for this trial
     */
    public void setProblemDefinition (String problemDefinition) {
	this.problemDefinition = problemDefinition;
    }

    /**
     * Sets the data session for this trial.
     *
     * @param	 Data session for this trial
     */
    public void setDataSession (DataSession dataSession) {
	this.dataSession = dataSession;
    }

    /**
     * Sets the node count for this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	nodeCount node count for this trial
     */
    public void setNodeCount (int nodeCount) {
	this.nodeCount = nodeCount;
    }

    /**
     * Sets the context (per node) count for this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	contextsPerNode context count for this trial
     */
    public void setNumContextsPerNode (int contextsPerNode) {
	this.contextsPerNode = contextsPerNode;
    }

    /**
     * Sets the thread (per context) count for this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	threadsPerContext thread count for this trial
     */
    public void setNumThreadsPerContext (int threadsPerContext) {
	this.threadsPerContext = threadsPerContext;
    }
 
    /**
     * Sets the user data of the current trial object.
     * <i>Note: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	userData the trial user data
     */
    public void setUserData(String userData) {
	this.userData = userData;
    }

    /**
     * Adds a metric to this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	metric Adds a metric to this trial
     */
    public void addMetric (Metric metric) {
	if (this.metric == null)
	    this.metric = new Vector();
	this.metric.addElement (metric);
    }

    // gets the metric data for the trial
    private void getTrialMetrics(DB db) {
	// create a string to hit the database
	StringBuffer buf = new StringBuffer();
	buf.append("select id, name ");
	buf.append("from metric ");
	buf.append("where trial = ");
	buf.append(getID());
	buf.append(" order by id ");
	// System.out.println(buf.toString());

	// get the results
	try {
	    ResultSet resultSet = db.executeQuery(buf.toString());	
	    while (resultSet.next() != false) {
		Metric tmp = new Metric();
		tmp.setID(resultSet.getInt(1));
		tmp.setName(resultSet.getString(2));
		tmp.setTrialID(getID());
		addMetric(tmp);
	    }
	    resultSet.close(); 
	}catch (Exception ex) {
	    ex.printStackTrace();
	    return;
	}
	return;
    }

    public static Vector getTrialList(DB db, String whereClause) {
	Vector trials = new Vector();
	// create a string to hit the database
	StringBuffer buf = new StringBuffer();
	buf.append("select t.id, t.experiment, e.application, ");
	buf.append("t.time, t.problem_definition, t.node_count, ");
	buf.append("t.contexts_per_node, t.threads_per_context, ");
	buf.append("t.name, t.userdata ");
	buf.append("from trial t inner join experiment e ");
	buf.append("on t.experiment = e.id ");
	buf.append(whereClause);
	buf.append(" order by t.node_count, t.contexts_per_node, t.threads_per_context ");
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
		trial.setProblemDefinition(resultSet.getString(5));
		trial.setNodeCount(resultSet.getInt(6));
		trial.setNumContextsPerNode(resultSet.getInt(7));
		trial.setNumThreadsPerContext(resultSet.getInt(8));
		trial.setName(resultSet.getString(9));
		trial.setUserData(resultSet.getString(10));
		trials.addElement(trial);
	    }
	    resultSet.close(); 
	}catch (Exception ex) {
	    ex.printStackTrace();
	    return null;
	}
		
	// get the function details
	Enumeration enum = trials.elements();
	Trial trial;
	while (enum.hasMoreElements()) {
	    trial = (Trial)enum.nextElement();
	    trial.getTrialMetrics(db);
	}

	return trials;
    }

    public int saveTrial(DB db) {
	int newTrialID = 0;
	try {
	    // save this trial
	    PreparedStatement statement = null;
	    statement = db.prepareStatement("INSERT INTO trial (name, experiment, time, problem_definition, node_count, contexts_per_node, threads_per_context, userdata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)");
	    statement.setString(1, name);
	    statement.setInt(2, experimentID);
	    statement.setString(3, time);
	    statement.setString(4, problemDefinition);
	    statement.setInt(5, nodeCount);
	    statement.setInt(6, contextsPerNode);
	    statement.setInt(7, threadsPerContext);
	    statement.setString(8, userData);
	    statement.executeUpdate();
	    String tmpStr = new String();
	    if (db.getDBType().compareTo("mysql") == 0)
			tmpStr = "select LAST_INSERT_ID();";
		if (db.getDBType().compareTo("db2") == 0)
			tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
	    else
		tmpStr = "select currval('trial_id_seq');";
	    newTrialID = Integer.parseInt(db.getDataItem(tmpStr));
	} catch (SQLException e) {
	    System.out.println("An error occurred while saving the trial.");
	    e.printStackTrace();
	    System.exit(0);
	}
	return newTrialID;
    }

    public static void deleteTrial(DB db, int trialID) {
		try {
	    	// save this trial
	    	PreparedStatement statement = null;
	    	statement = db.prepareStatement("delete from atomic_location_profile where atomic_event in (select id from atomic_event where trial = ?)");
	    	statement.setInt(1, trialID);
	    	statement.execute();
	    	statement = db.prepareStatement(" delete from atomic_event where trial = ?"); 
	    	statement.setInt(1, trialID);
	    	statement.execute();
	    	statement = db.prepareStatement(" delete from interval_location_profile where interval_event in (select id from interval_event where trial = ?)");
	    	statement.setInt(1, trialID);
	    	statement.execute();
	    	statement = db.prepareStatement(" delete from interval_mean_summary where interval_event in (select id from interval_event where trial = ?)");
	    	statement.setInt(1, trialID);
	    	statement.execute();
	    	statement = db.prepareStatement(" delete from interval_total_summary where interval_event in (select id from interval_event where trial = ?)");
	    	statement.setInt(1, trialID);
	    	statement.execute();
	    	statement = db.prepareStatement(" delete from interval_event where trial = ?");
	    	statement.setInt(1, trialID);
	    	statement.execute();
	    	statement = db.prepareStatement(" delete from metric where trial = ?");
	    	statement.setInt(1, trialID);
	    	statement.execute();
	    	statement = db.prepareStatement(" delete from trial where id = ?");
	    	statement.setInt(1, trialID);
	    	statement.execute();
		} catch (SQLException e) {
	    	System.out.println("An error occurred while deleting the trial.");
	    	e.printStackTrace();
	    	System.exit(0);
		}
    }

}
