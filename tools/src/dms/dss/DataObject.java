package dms.dss;

public class DataObject {
    private int nodeID;
    private int contextID;
    private int threadID;
    
    public void setNodeID (int nodeID) {
	this.nodeID = nodeID;
    }
    
    public void setContextID (int contextID) {
	this.contextID = contextID;
    }
    
    public void setThreadID (int threadID) {
	this.threadID = threadID;
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
}

