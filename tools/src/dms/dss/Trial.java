package dms.dss;

import java.util.*;

/**
 * Holds all the data for a trial in the database.
 * This object is returned by the DataSession class and all of its subtypes.
 * The Trial object contains all the information associated with
 * an trial from which the TAU performance data has been generated.
 * A trial is associated with one experiment and one application, and has one or more
 * functions and/or user events associated with it.  A Trial has information
 * related to the particular run, including the number of nodes used,
 * the number of contexts per node, the number of threads per context
 * and the metrics collected during the run.
 *
 * <P>CVS $Id: Trial.java,v 1.11 2003/12/03 23:46:32 bertie Exp $</P>
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
 * Gets the metric name at the particular index.
 *
 * @return	a metric name.
 */
	public String getMetricName (int metricIndex) {
		return (String)this.metric.elementAt(metricIndex);
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
	public void addMetric (String metric) {
		if (this.metric == null)
			this.metric = new Vector();
		this.metric.addElement (metric);
	}

/**
 * Clears the metric vector in the trial.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 */
	public void clearMetric() {
		this.metric = null;
	}

}
