package dms.dss;

/**
 * Holds all the data for a function in the database.
 *
 * <P>CVS $Id: Function.java,v 1.6 2003/08/07 20:23:07 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	%I%, %G%
 */
public class Function {
	private int functionIndexID;
	private int functionID;
	private String name;
	private String group;
	private int trialID;
	private int experimentID;
	private int applicationID;
	private FunctionDataObject meanSummary;
	private FunctionDataObject totalSummary;

	public void setIndexID (int id) {
		this.functionIndexID = id;
	}

	public void setFunctionID (int id) {
		this.functionID = id;
	}

	public void setName (String name) {
		this.name = name;
	}

	public void setGroup (String group) {
		this.group = group;
	}

	public void setTrialID (int id) {
		this.trialID = id;
	}

	public void setExperimentID (int id) {
		this.experimentID = id;
	}

	public void setApplicationID (int id) {
		this.applicationID = id;
	}

	public void setMeanSummary (FunctionDataObject meanSummary) {
		this.meanSummary = meanSummary;
	}

	public void setTotalSummary (FunctionDataObject totalSummary) {
		this.totalSummary = totalSummary;
	}

	public int getIndexID () {
		return this.functionIndexID;
	}

	public int getFunctionID () {
		return this.functionID;
	}

	public String getName () {
		return this.name;
	}

	public String getGroup () {
		return this.group;
	}

	public int getTrialID () {
		return this.trialID;
	}

	public int getExperimentID () {
		return this.experimentID;
	}

	public int getApplicationID () {
		return this.applicationID;
	}

	public FunctionDataObject getMeanSummary () {
		return this.meanSummary;
	}

	public FunctionDataObject getTotalSummary () {
		return this.totalSummary;
	}
}

