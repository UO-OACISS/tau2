package edu.uoregon.tau.dms.dss;

import edu.uoregon.tau.dms.database.DB;
import java.sql.SQLException;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.io.Serializable;

/**
 * Holds all the data for a metric in the database.
 *
 * <P>CVS $Id: Metric.java,v 1.6 2004/10/27 21:34:30 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 */
public class Metric implements Serializable {
    private int metricID;
    private int trialID;
    private String name;

    public boolean equals (Metric inMetric) {
	return (this.name.equals(inMetric.getName())) ? true : false;
    }

    public boolean equals (Object inObject) {
	Metric inMetric = (Metric)inObject;
	return equals(inMetric);
    }

    /**
     * Gets the unique identifier of the current metric object.
     *
     * @return	the unique identifier of the metric
     */
    public int getID () {
	return metricID;
    }

    /**
     * Gets the unique trial identifier of the current metric object.
     *
     * @return	the unique trial identifier of the metric
     */
    public int getTrialID () {
	return trialID;
    }

    /**
     * Gets the name of the current metric object.
     *
     * @return	the name of the metric
     */
    public String getName() {
	return name;
    }

    public String toString() {
	return name;
    }

    /**
     * Sets the unique ID associated with this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	id unique ID associated with this trial
     */
    public void setID (int id) {
	this.metricID = id;
    }

    /**
     * Sets the unique trial ID associated with this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	trial unique trial ID associated with this trial
     */
    public void setTrialID (int trial) {
	this.trialID = trial;
    }

    /**
     * Sets the name of the current metric object.
     * <i>Note: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	name the metric name
     */
    public void setName(String name) {
	this.name = name;
    }

    public int saveMetric(DB db, int newTrialID) {
	int newMetricID = 0;
	try {
	    PreparedStatement stmt1 = null;
	    stmt1 = db.prepareStatement("INSERT INTO metric (name, trial) VALUES (?, ?)");
	    stmt1.setString(1, getName());
	    stmt1.setInt(2, newTrialID);
	    stmt1.executeUpdate();
	    stmt1.close();

	    String tmpStr = new String();
	    if (db.getDBType().compareTo("mysql") == 0)
		tmpStr = "select LAST_INSERT_ID();";
	    else if (db.getDBType().compareTo("db2") == 0)
		tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
	    else if (db.getDBType().compareTo("oracle") == 0)
		tmpStr = "select metric_id_seq.currval FROM dual";
	    else
		tmpStr = "select currval('metric_id_seq');";
	    newMetricID = Integer.parseInt(db.getDataItem(tmpStr));
	} catch (SQLException e) {
	    System.out.println("An error occurred while saving the trial.");
	    e.printStackTrace();
	    System.exit(0);
	}
	return newMetricID;
    }
}
