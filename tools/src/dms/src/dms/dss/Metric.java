package dms.dss;

/**
 * Holds all the data for a metric in the database.
 *
 * <P>CVS $Id: Metric.java,v 1.1 2004/03/27 01:02:56 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 */
public class Metric {
	private int metricID;
	private int trialID;
	private String name;

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
}
