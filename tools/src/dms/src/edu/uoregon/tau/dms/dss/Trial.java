package edu.uoregon.tau.dms.dss;

import edu.uoregon.tau.dms.database.*;
import java.sql.*;
import java.util.Vector;
import java.util.Enumeration;
import java.lang.String;
import java.io.Serializable;

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
 * <P>CVS $Id: Trial.java,v 1.11 2004/10/29 22:43:10 amorris Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DataSession#getTrialList
 * @see		DataSession#setTrial
 * @see		Application
 * @see		Experiment
 * @see		IntervalEvent
 * @see		AtomicEvent
 */
public class Trial implements Serializable {
    private int trialID;
    private int experimentID;
    private int applicationID;
    private String name;
    private Vector metric;

    protected DataSession dataSession = null;


    private String fields[];
    private static String fieldNames[];
    private static int fieldTypes[];


    public Trial(int numFields) {
	fields = new String[numFields];
    }

//     public Trial(String fieldNames[], int fieldTypes[]) {
// 	this.fields = new String[fieldNames.length];
// 	this.fieldNames = fieldNames;
// 	this.fieldTypes = fieldTypes;
//     }

    // copy constructor (sort of)
    public Trial(Trial trial) {
	this.name = trial.getName();
	this.applicationID = trial.getApplicationID();
	this.experimentID = trial.getExperimentID();
	this.trialID = trial.getID();
	
	this.fields = trial.fields;
// 	this.fieldNames = trial.fieldNames;
// 	this.fieldTypes = trial.fieldTypes;
    }



    // these are here because I don't have time to fix the analysis
    // routines that are (but shouldn't be) using them
    public int getNodeCount() {
	System.err.println ("Do not use trial.getNodeCount");
	return -1;
    }

    public int getNumContextsPerNode() {
	System.err.println ("Do not use trial.getNumContextsPerNode");
	return -1;
    }

    public int getNumThreadsPerContext() {
	System.err.println ("Do not use trial.getNumThreadsPerContext");
	return -1;
    }

    public int[] getMaxNCTNumbers() {
	System.err.println ("Do not use trial.getMaxNCTNumbers");
	return new int[3];
    }
    ///////////////////////////////////////////////////////


    public int getNumFields() {
	return fields.length;
    }

    public String getFieldName(int idx) {
	return Trial.fieldNames[idx];
    }

    public int getFieldType(int idx) {
	return Trial.fieldTypes[idx];
    }

    public String getField(int idx) {
	return fields[idx];
    }

    public void setField(int idx, String field) {
	if (DBConnector.isIntegerType(fieldTypes[idx]) && field != null) {
	    try {
		int test = Integer.parseInt(field);
	    } catch (java.lang.NumberFormatException e) {
		return;
	    }
	}
	
	if (DBConnector.isFloatingPointType(fieldTypes[idx]) && field != null) {
	    try {
		double test = Double.parseDouble(field);
	    } catch (java.lang.NumberFormatException e) {
		return;
	    }
	}

	fields[idx] = field;
    }


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

    public String toString() {
	return name;
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
     * Gets the number of metrics collected in this trial.
     *
     * @return	metric count for this trial.
     */
    public int getMetricCount() {
	if (this.metric == null)
	    return 0;
	else
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
     * Get the metric name corresponding to the given id.  The DataSession object will maintain a reference to the Vector of metric values.  To clear this reference, call setMetric(String) with null.
     *
     * @param	metricID metric id.
     *
     * @return	The metric name as a String.
     */
    public String getMetricName(int metricID) {
	
	//Try getting the metric name.
	if ((this.metric!=null) && (metricID < this.metric.size()))
	    return ((Metric)this.metric.elementAt(metricID)).getName();
	else
	    return null;
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
     * Sets the data session for this trial.
     *
     * @param	 dataSession DataSession for this trial
     */
    public void setDataSession (DataSession dataSession) {
	this.dataSession = dataSession;
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
	buf.append("from " + db.getSchemaPrefix() + "metric ");
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


    public static void getMetaData(DB db) {
	// see if we've already have them
	if (Trial.fieldNames != null)
	    return;

	try {
	    ResultSet resultSet = null;
	    
	    String trialFieldNames[] = null;
	    int trialFieldTypes[] = null;
	    
	    DatabaseMetaData dbMeta = db.getMetaData();
	    
	    if (db.getDBType().compareTo("oracle") == 0) {
		resultSet = dbMeta.getColumns(null, null, "TRIAL", "%");
	    } else {
		resultSet = dbMeta.getColumns(null, null, "trial", "%");
	    }
	    
	    Vector nameList = new Vector();
	    Vector typeList = new Vector();
	    while (resultSet.next() != false) {
		
		int ctype = resultSet.getInt("DATA_TYPE");
		String cname = resultSet.getString("COLUMN_NAME");
		String typename = resultSet.getString("TYPE_NAME");
		
		// only integer and string types (for now)
		// don't do name and id, we already know about them
		
		if (DBConnector.isReadAbleType(ctype) 
		    && cname.toUpperCase().compareTo("ID") != 0 
		    && cname.toUpperCase().compareTo("NAME") != 0 
		    && cname.toUpperCase().compareTo("APPLICATION") != 0 
		    && cname.toUpperCase().compareTo("EXPERIMENT") != 0) { 
		    
		    nameList.add(resultSet.getString("COLUMN_NAME"));
		    typeList.add(new Integer(ctype));
		}
	    }
	    resultSet.close();
	    
	    Trial.fieldNames = new String[nameList.size()];
	    Trial.fieldTypes = new int[typeList.size()];
	    for (int i=0; i<typeList.size(); i++) {
		Trial.fieldNames[i] = (String) nameList.get(i);
		Trial.fieldTypes[i] = ((Integer)typeList.get(i)).intValue();
	    }
	} catch (SQLException e) {
	    e.printStackTrace();
	}
    }
    public static Vector getTrialList(DB db, String whereClause) {
	
	try {
	    
	    Trial.getMetaData(db);
	    
	    // create a string to hit the database
	    StringBuffer buf = new StringBuffer();
	    buf.append("select t.id, t.experiment, e.application, ");
	    buf.append("t.name");

	    for (int i=0; i<Trial.fieldNames.length; i++) {
		buf.append(", t." + Trial.fieldNames[i]);
	    }

	    buf.append(" from " + db.getSchemaPrefix() + "trial t inner join " 
		       + db.getSchemaPrefix() + "experiment e ");
	    buf.append("on t.experiment = e.id ");
	    buf.append(whereClause);
	    buf.append(" order by t.id ");

	    
	    Vector trials = new Vector();
	    ResultSet resultSet = db.executeQuery(buf.toString());	
	    while (resultSet.next() != false) {
		Trial trial = new Trial(Trial.fieldNames.length);

		int pos = 1;
		trial.setID(resultSet.getInt(pos++));
		trial.setExperimentID(resultSet.getInt(pos++));
		trial.setApplicationID(resultSet.getInt(pos++));
		trial.setName(resultSet.getString(pos++));

		for (int i=0; i<Trial.fieldNames.length; i++) {
		    trial.setField(i, resultSet.getString(pos++));
		}

		trials.addElement(trial);
	    }
	    resultSet.close(); 
	

	    // get the function details
	    Enumeration enum = trials.elements();
	    Trial trial;
	    while (enum.hasMoreElements()) {
		trial = (Trial)enum.nextElement();
		trial.getTrialMetrics(db);
	    }

	    return trials;

	} catch (Exception ex) {
	    ex.printStackTrace();
	    return null;
	}
		

    }

    public int saveTrial(DB db) {
	boolean itExists = exists(db);
	int newTrialID = 0;

	try {

	    StringBuffer buf = new StringBuffer();
	    
	    if (itExists) {
		buf.append("UPDATE " + db.getSchemaPrefix() + "trial SET name = ?, experiment = ?");
		for (int i=0; i < this.getNumFields(); i++) {
		    if (DBConnector.isWritableType(this.getFieldType(i)))
			buf.append(", " + this.getFieldName(i) + " = ?");
		}
		buf.append(" WHERE id = ?");
	    } else {
		buf.append("INSERT INTO " + db.getSchemaPrefix() + "trial (name, experiment");
		for (int i=0; i < this.getNumFields(); i++) {
		    if (DBConnector.isWritableType(this.getFieldType(i)))
			buf.append(", " + this.getFieldName(i));
		}
		buf.append(") VALUES (?, ?");
		for (int i=0; i < this.getNumFields(); i++) {
		    if (DBConnector.isWritableType(this.getFieldType(i)))
			buf.append(", ?");
		}
		buf.append(")");
	    }
	    
	    
	    PreparedStatement statement = db.prepareStatement(buf.toString());

	    int pos = 1;

	    statement.setString(pos++, name);
	    statement.setInt(pos++, experimentID);
	    for (int i=0; i < this.getNumFields(); i++) {
		if (DBConnector.isWritableType(this.getFieldType(i)))
		    statement.setString(pos++, this.getField(i));
	    }

	    if (itExists) {
		statement.setInt(pos, trialID);
	    }
	    statement.executeUpdate();
	    statement.close();
	    if (itExists) {
		newTrialID = trialID;
	    } else {
		String tmpStr = new String();
		if (db.getDBType().compareTo("mysql") == 0)
		    tmpStr = "select LAST_INSERT_ID();";
		else if (db.getDBType().compareTo("db2") == 0)
		    tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
		else if (db.getDBType().compareTo("oracle") == 0) 
		    tmpStr = "select " + db.getSchemaPrefix() + "trial_id_seq.currval FROM dual";
		else
		    tmpStr = "select currval('trial_id_seq');";
		newTrialID = Integer.parseInt(db.getDataItem(tmpStr));
	    }


	    // get the fields since this is an insert
	    if (!itExists) {

		Trial.getMetaData(db);
		
		this.fields = new String[Trial.fieldNames.length];
	    }

	} catch (SQLException e) {
	    System.out.println("An error occurred while saving the trial.");
	    e.printStackTrace();
	}
	return newTrialID;
    }

    public static void deleteTrial(DB db, int trialID) {
	try {
	    // save this trial
	    PreparedStatement statement = null;

	    // delete from the atomic_location_profile table
	    if (db.getDBType().compareTo("mysql") == 0) {
		statement = db.prepareStatement(" DELETE atomic_location_profile.* FROM " 
						+ db.getSchemaPrefix() 
						+ "atomic_location_profile LEFT JOIN " 
						+ db.getSchemaPrefix() 
						+ "atomic_event ON atomic_location_profile.atomic_event = atomic_event.id WHERE atomic_event.trial = ?");
	    } else {
		// Postgresql, oracle, and DB2?
		statement = db.prepareStatement(" DELETE FROM " 
						+ db.getSchemaPrefix() 
						+ "atomic_location_profile WHERE atomic_event in (SELECT id FROM " 
						+ db.getSchemaPrefix() 
						+ "atomic_event WHERE trial = ?)");
	    }
	    statement.setInt(1, trialID);
	    statement.execute();
	    statement.close();

	    // delete the from the atomic_events table
	    statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "atomic_event WHERE trial = ?"); 
	    statement.setInt(1, trialID);
	    statement.execute();
	    statement.close();

	    // delete from the interval_location_profile table
	    if (db.getDBType().compareTo("mysql") == 0) {
		statement = db.prepareStatement(" DELETE interval_location_profile.* FROM " 
						+ db.getSchemaPrefix() 
						+ "interval_location_profile LEFT JOIN " 
						+ db.getSchemaPrefix() 
						+ "interval_event ON interval_location_profile.interval_event = interval_event.id WHERE interval_event.trial = ?");
	    } else {
		// Postgresql and DB2?
		statement = db.prepareStatement(" DELETE FROM " 
						+ db.getSchemaPrefix() 
						+ "interval_location_profile WHERE interval_event IN (SELECT id FROM " 
						+ db.getSchemaPrefix() 
						+ "interval_event WHERE trial = ?)");
	    }
	    statement.setInt(1, trialID);
	    statement.execute();
	    statement.close();

	    // delete from the interval_mean_summary table
	    if (db.getDBType().compareTo("mysql") == 0) {
		statement = db.prepareStatement(" DELETE interval_mean_summary.* FROM interval_mean_summary LEFT JOIN interval_event ON interval_mean_summary.interval_event = interval_event.id WHERE interval_event.trial = ?");
	    } else {
		// Postgresql and DB2?
		statement = db.prepareStatement(" DELETE FROM " 
						+ db.getSchemaPrefix() 
						+ "interval_mean_summary WHERE interval_event IN (SELECT id FROM " 
						+ db.getSchemaPrefix() 
						+ "interval_event WHERE trial = ?)");
	    }
	    statement.setInt(1, trialID);
	    statement.execute();
	    statement.close();


	    if (db.getDBType().compareTo("mysql") == 0) {
		statement = db.prepareStatement(" DELETE interval_total_summary.* FROM interval_total_summary LEFT JOIN interval_event ON interval_total_summary.interval_event = interval_event.id WHERE interval_event.trial = ?");
	    } else {
		// Postgresql and DB2?
		statement = db.prepareStatement(" DELETE FROM " 
						+ db.getSchemaPrefix() 
						+ "interval_total_summary WHERE interval_event IN (SELECT id FROM " 
						+ db.getSchemaPrefix() 
						+ "interval_event WHERE trial = ?)");
	    }

	    statement.setInt(1, trialID);
	    statement.execute();
	    statement.close();

	    statement = db.prepareStatement(" DELETE FROM " 
					    + db.getSchemaPrefix() 
					    + "interval_event WHERE trial = ?");
	    statement.setInt(1, trialID);
	    statement.execute();
	    statement.close();

	    statement = db.prepareStatement(" DELETE FROM " 
					    + db.getSchemaPrefix() 
					    + "metric WHERE trial = ?");
	    statement.setInt(1, trialID);
	    statement.execute();
	    statement.close();

	    statement = db.prepareStatement(" DELETE FROM " 
					    + db.getSchemaPrefix() + "trial WHERE id = ?");
	    statement.setInt(1, trialID);
	    statement.execute();
	    statement.close();

	} catch (SQLException e) {
	    System.out.println("An error occurred while deleting the trial.");
	    e.printStackTrace();
	}
    }

    private boolean exists(DB db) {
	boolean retval = false;
	try {
	    PreparedStatement statement = db.prepareStatement("SELECT name FROM " 
							      + db.getSchemaPrefix() 
							      + "trial WHERE id = ?");
	    statement.setInt(1, trialID);
	    ResultSet results = statement.executeQuery();
	    while (results.next() != false) {
		retval = true;
		break;
	    }
	    results.close();
	} catch (SQLException e) {
	    System.out.println("An error occurred while saving the application.");
	    e.printStackTrace();
	}
	return retval;
    }
}
