package dms.dss;

public class Trial {
	private int trialID;
	private int experimentID;
	private int applicationID;
	private String time;
	private int problemSize;
	private int numNodes;
	private int contextPNode;
	private int threadPContext;
	private int xmlFileID;

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

	public void setNumNodes (int id) {
		this.numNodes = id;
	}

	public void setNumContextsPerNode (int id) {
		this.contextPNode = id;
	}

	public void setNumThreadsPerContext (int id) {
		this.threadPContext = id;
	}

	public void setXMLFileID (int id) {
		this.xmlFileID = id;
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

	public int getNumNodes () {
		return this.numNodes;
	}

	public int getNumContextsPerNode () {
		return this.contextPNode;
	}

	public int getNumThreadsPerContext () {
		return this.threadPContext;
	}

	public int getXMLFileID () {
		return this.xmlFileID;
	}

}

