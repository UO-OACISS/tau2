package dms.dss;

public class FunctionDataObject {
	private int functionID;
	private int nodeID;
	private int contextID;
	private int threadID;
	private double inclusivePercentage;
	private double inclusive;
	private double exclusivePercentage;
	private double exclusive;
	private int numCalls;
	private int numSubroutines;
	private double inclusivePerCall;

	public void setFunctionID (int functionID) {
		this.functionID = functionID;
	}

	public void setNodeID (int nodeID) {
		this.nodeID = nodeID;
	}

	public void setContextID (int contextID) {
		this.contextID = contextID;
	}

	public void setThreadID (int threadID) {
		this.threadID = threadID;
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

	public int getFunctionID () {
		return this.functionID;
	}

	public int getNodeID () {
		return this.nodeID;
	}

	public int getContextID () {
		return this.contextID;
	}

	public int getThreadID () {
		return this.threadID;
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
}

