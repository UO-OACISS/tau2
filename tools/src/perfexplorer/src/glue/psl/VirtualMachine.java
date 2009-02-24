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
public class VirtualMachine {

	private String name = null;
	private Set<VirtualMachine> virtualMachines = null;
	private Set<VirtualNode> virtualNodes = null;
	
	/**
	 * 
	 */
	public VirtualMachine(String name) {
		this.name = name;
		this.virtualMachines = new HashSet<VirtualMachine>();
		this.virtualNodes = new HashSet<VirtualNode>();
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
	
	public void addVirtualMachine(VirtualMachine virtualMachine) {
		this.virtualMachines.add(virtualMachine);
	}

	/**
	 * @return the virtualMachines
	 */
	public Set<VirtualMachine> getVirtualMachines() {
		return virtualMachines;
	}

	/**
	 * @param virtualMachines the virtualMachines to set
	 */
	public void setVirtualMachines(Set<VirtualMachine> virtualMachines) {
		this.virtualMachines = virtualMachines;
	}
	
	public int getNumberOfProcessingUnits() {
		int units = 0;
		
		for (VirtualMachine machine : this.virtualMachines) {
			units += machine.getNumberOfProcessingUnits();
		}
		for (VirtualNode node : this.virtualNodes) {
			units += node.getNumberOfProcessors();
		}
		
		return units;
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
