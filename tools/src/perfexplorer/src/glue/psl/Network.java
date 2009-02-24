/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

import java.util.HashSet;
import java.util.Set;

/**
 * @author khuck
 *
 */
public class Network {

	private String name = null;
	private int bandwidth = 0;
	private int latency = 0;
	private Set<VirtualNode> virtualNodes = null;
	
	/**
	 * 
	 */
	public Network(String name) {
		this.name = name;
		this.virtualNodes = new HashSet<VirtualNode>();
	}

	/**
	 * @return the bandwidth
	 */
	public int getBandwidth() {
		return bandwidth;
	}

	/**
	 * @param bandwidth the bandwidth to set
	 */
	public void setBandwidth(int bandwidth) {
		this.bandwidth = bandwidth;
	}

	/**
	 * @return the latency
	 */
	public int getLatency() {
		return latency;
	}

	/**
	 * @param latency the latency to set
	 */
	public void setLatency(int latency) {
		this.latency = latency;
	}

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @param name the name to set
	 */
	public void setName(String name) {
		this.name = name;
	}

	public void addVirtualNode(VirtualNode virtualNode) {
		this.virtualNodes.add(virtualNode);
	}
	
	/**
	 * @return the virtualNodes
	 */
	public Set<VirtualNode> getVirtualNodes() {
		return virtualNodes;
	}

	/**
	 * @param virtualNodes the virtualNodes to set
	 */
	public void setVirtualNodes(Set<VirtualNode> virtualNodes) {
		this.virtualNodes = virtualNodes;
	}

}
