/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

import java.util.ArrayList;
import java.util.List;

/**
 * @author khuck
 *
 */
public class VirtualNode {

	private String nodeID = null;
	private int numberOfProcessors = 1;
	private int memorySize = 0;
	private int hardDiskSize = 0;
	private List<Integer> dataCacheSize = null;
	private List<Integer> cacheMissPenalty = null;
	private VirtualMachine virtualMachine = null;
	private Network network = null;
	
	/**
	 * 
	 */
	public VirtualNode(String nodeID, VirtualMachine virtualMachine) {
		this.nodeID = nodeID;
		this.virtualMachine = virtualMachine;
		this.virtualMachine.addVirtualNode(this);
		this.dataCacheSize = new ArrayList<Integer>();
		this.cacheMissPenalty = new ArrayList<Integer>();
	}

	public void addDataCacheMissPenalty(int cacheMissPenalty) {
		this.cacheMissPenalty.add(cacheMissPenalty);
	}
	
	/**
	 * @return the cacheMissPenalty
	 */
	public List<Integer> getCacheMissPenalty() {
		return cacheMissPenalty;
	}

	/**
	 * @param cacheMissPenalty the cacheMissPenalty to set
	 */
	public void setCacheMissPenalty(List<Integer> cacheMissPenalty) {
		this.cacheMissPenalty = cacheMissPenalty;
	}

	public void addDataCacheSize(int dataCacheSize) {
		this.dataCacheSize.add(dataCacheSize);
	}
	
	/**
	 * @return the dataCacheSize
	 */
	public List<Integer> getDataCacheSize() {
		return dataCacheSize;
	}

	/**
	 * @param dataCacheSize the dataCacheSize to set
	 */
	public void setDataCacheSize(List<Integer> dataCacheSize) {
		this.dataCacheSize = dataCacheSize;
	}

	/**
	 * @return the hardDiskSize
	 */
	public int getHardDiskSize() {
		return hardDiskSize;
	}

	/**
	 * @param hardDiskSize the hardDiskSize to set
	 */
	public void setHardDiskSize(int hardDiskSize) {
		this.hardDiskSize = hardDiskSize;
	}

	/**
	 * @return the memorySize
	 */
	public int getMemorySize() {
		return memorySize;
	}

	/**
	 * @param memorySize the memorySize to set
	 */
	public void setMemorySize(int memorySize) {
		this.memorySize = memorySize;
	}

	/**
	 * @return the network
	 */
	public Network getNetwork() {
		return network;
	}

	/**
	 * @param network the network to set
	 */
	public void setNetwork(Network network) {
		this.network = network;
		this.network.addVirtualNode(this);
	}

	/**
	 * @return the nodeID
	 */
	public String getNodeID() {
		return nodeID;
	}

	/**
	 * @param nodeID the nodeID to set
	 */
	public void setNodeID(String nodeID) {
		this.nodeID = nodeID;
	}

	/**
	 * @return the numberOfProcessors
	 */
	public int getNumberOfProcessors() {
		return numberOfProcessors;
	}

	/**
	 * @param numberOfProcessors the numberOfProcessors to set
	 */
	public void setNumberOfProcessors(int numberOfProcessors) {
		this.numberOfProcessors = numberOfProcessors;
	}

	/**
	 * @return the virtualMachine
	 */
	public VirtualMachine getVirtualMachine() {
		return virtualMachine;
	}

	/**
	 * @param virtualMachine the virtualMachine to set
	 */
	public void setVirtualMachine(VirtualMachine virtualMachine) {
		this.virtualMachine = virtualMachine;
	}

}
