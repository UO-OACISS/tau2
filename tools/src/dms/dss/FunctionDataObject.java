package dms.dss;

/**
 * Holds all the data for a function data object in the database.
 *
 * <P>CVS $Id: FunctionDataObject.java,v 1.9 2003/08/01 21:38:22 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	%I%, %G%
 */
public class FunctionDataObject {
	private int node;
	private int context;
	private int thread;
	private int functionID;
	private double inclusivePercentage;
	private double inclusive;
	private double exclusivePercentage;
	private double exclusive;
	private int numCalls;
	private int numSubroutines;
	private double inclusivePerCall;
	private String metric;

	public void setNode (int node) {
		this.node = node;
	}

	public void setContext (int context) {
		this.context = context;
	}

	public void setThread (int thread) {
		this.thread = thread;
	}

	public void setFunctionIndexID (int functionID) {
		this.functionID = functionID;
	}

	public void setInclusivePercentage (double inclusivePercentage) {
		this.inclusivePercentage = inclusivePercentage;
	}

	public void setInclusive (double inclusive) {
		this.inclusive = inclusive;
	}

	public void setExclusivePercentage (double exclusivePercentage) {
		this.exclusivePercentage = exclusivePercentage;
	}

	public void setExclusive (double exclusive) {
		this.exclusive = exclusive;
	}

	public void setNumCalls (int numCalls) {
		this.numCalls = numCalls;
	}

	public void setNumSubroutines (int numSubroutines) {
		this.numSubroutines = numSubroutines;
	}

	public void setInclusivePerCall (double inclusivePerCall) {
		this.inclusivePerCall = inclusivePerCall;
	}

	public void setMetric (String metric) {
		this.metric = metric;
	}

	public int getNode () {
		return this.node;
	}

	public int getContext () {
		return this.context;
	}

	public int getThread () {
		return this.thread;
	}

	public int getFunctionIndexID () {
		return this.functionID;
	}

	public double getInclusivePercentage () {
		return this.inclusivePercentage;
	}

	public double getInclusive () {
		return this.inclusive;
	}

	public double getExclusivePercentage () {
		return this.exclusivePercentage;
	}

	public double getExclusive () {
		return this.exclusive;
	}

	public int getNumCalls () {
		return this.numCalls;
	}

	public int getNumSubroutines () {
		return this.numSubroutines;
	}

	public double getInclusivePerCall () {
		return this.inclusivePerCall;
	}

	public String getMetric () {
		return this.metric;
	}
}

