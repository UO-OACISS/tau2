package dms.dss;

import java.util.*;

/**
 * Holds all the data for a trial in the database.
 *
 * <P>CVS $Id: Trial.java,v 1.6 2003/08/11 07:41:44 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	%I%, %G%
 */
public class Trial {
	private int trialID;
	private int experimentID;
	private int applicationID;
	private String time;
	private int problemSize;
	private int nodeCount;
	private int contextsPerNode;
	private int threadsPerContext;
	private Vector metric;

	public void setID (int id) {
		this.trialID = id;
	}

	public void setExperimentID (int id) {
		this.experimentID = id;
	}

	public void setApplicationID (int id) {
		this.applicationID = id;
	}

	public void setTime (String id) {
		this.time = id;
	}

	public void setProblemSize (int id) {
		this.problemSize = id;
	}

	public void setNodeCount (int id) {
		this.nodeCount = id;
	}

	public void setNumContextsPerNode (int id) {
		this.contextsPerNode = id;
	}

	public void setNumThreadsPerContext (int id) {
		this.threadsPerContext = id;
	}

	public void addMetric (String metric) {
		if (this.metric == null)
			this.metric = new Vector();
		this.metric.addElement (metric);
	}

	public void clearMetric() {
		this.metric = null;
	}

	public int getID () {
		return trialID;
	}

	public int getExperimentID () {
		return experimentID;
	}

	public int getApplicationID () {
		return applicationID;
	}

	public String getTime () {
		return this.time;
	}

	public int getProblemSize () {
		return this.problemSize;
	}

	public int getNodeCount () {
		return this.nodeCount;
	}

	public int getNumContextsPerNode () {
		return this.contextsPerNode;
	}

	public int getNumThreadsPerContext () {
		return this.threadsPerContext;
	}

	public int getMetricCount() {
		return this.metric.size();
	}

	public String getMetric (int metricIndex) {
		return (String)this.metric.elementAt(metricIndex);
	}
}

