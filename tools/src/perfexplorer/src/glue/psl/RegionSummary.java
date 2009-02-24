/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class RegionSummary {

	private String processingUnit = null;
	private CodeRegion codeRegion = null;
	private Experiment experiment = null;
	
	// measured fields
	private double inclusiveTime = 0.0;
	private double exclusiveTime = 0.0;
	private double totalInstructions = 0.0;
	private double floatingPointInstructions = 0.0;
	private double[] cacheMisses = {0.0, 0.0, 0.0};
	private double[] cacheAccesses = {0.0, 0.0, 0.0};
	
	// derived fields
	private double overhead = -1.0;
	private double synchronizationTime = -1.0;
	private double receiveTime = -1.0;
	private double lossOfParallelism = -1.0;
	private double controlOfParallelism = -1.0;
	
	/**
	 * 
	 */
	public RegionSummary(Experiment experiment, CodeRegion codeRegion, String processingUnit) {
		this.experiment = experiment;
		this.experiment.addRegionSummary(this);
		this.codeRegion = codeRegion;
	}

	/**
	 * @return the codeRegion
	 */
	public CodeRegion getCodeRegion() {
		return codeRegion;
	}

	/**
	 * @param codeRegion the codeRegion to set
	 */
	public void setCodeRegion(CodeRegion codeRegion) {
		this.codeRegion = codeRegion;
	}

	/**
	 * @return the experiment
	 */
	public Experiment getExperiment() {
		return experiment;
	}

	/**
	 * @param experiment the experiment to set
	 */
	public void setExperiment(Experiment experiment) {
		this.experiment = experiment;
	}

	/**
	 * @return the processingUnit
	 */
	public String getProcessingUnit() {
		return processingUnit;
	}

	/**
	 * @param processingUnit the processingUnit to set
	 */
	public void setProcessingUnit(String processingUnit) {
		this.processingUnit = processingUnit;
	}

	public VirtualNode getExecutionNode() {
		// TODO
		VirtualNode node = null;
		return node;
	}
	
	public double getIdentifiedOverhead() {
		// TODO
		double overhead = 0;
		return overhead;
	}
	
	public double getExecutionTime() {
		return this.inclusiveTime;
	}
	
	public double getSynchronizationTime() {
		// TODO
		double time = 0;
		return time;
	}
	
	public double getReceiveTime() {
		double time = 0;
		if (this.codeRegion.getGroupType() == CodeRegion.GroupType.MPI) {
			if (this.codeRegion.getShortName().toUpperCase().indexOf("RECV") > -1) {
				time = this.inclusiveTime;
			}
		} else {
			// recurse through the children
		}
		return time;
	}
	
	public double getCommunicationTime() {
		// TODO
		double time = 0;
		return time;
	}
	
	public double getLossOfParallelism() {
		// TODO
		double loss = 0;
		return loss;
	}
	
	public double getControlOfParallelism() {
		// TODO
		double control = 0;
		return control;
	}
	
	public double getNumberOfInstructions() {
		// TODO
		double instructions = 0;
		return instructions;
	}
	
	public double getNumberOfCacheMisses(int level) {
		return this.cacheMisses[level];
	}
	
	public double getNumberOfCacheHits(int level) {
		return this.cacheAccesses[level] - this.cacheMisses[level];
	}

	public double getNumberOfCacheAccesses(int level) {
		return this.cacheAccesses[level];
	}

	/**
	 * @param cacheMisses the cacheMisses to set
	 */
	public void setCacheMisses(double[] cacheMisses) {
		this.cacheMisses = cacheMisses;
	}

	/**
	 * @param exclusiveTime the exclusiveTime to set
	 */
	public void setExclusiveTime(double exclusiveTime) {
		this.exclusiveTime = exclusiveTime;
	}

	/**
	 * @param floatingPointInstructions the floatingPointInstructions to set
	 */
	public void setFloatingPointInstructions(double floatingPointInstructions) {
		this.floatingPointInstructions = floatingPointInstructions;
	}

	/**
	 * @param inclusiveTime the inclusiveTime to set
	 */
	public void setInclusiveTime(double inclusiveTime) {
		this.inclusiveTime = inclusiveTime;
	}

	/**
	 * @param totalInstructions the totalInstructions to set
	 */
	public void setTotalInstructions(double totalInstructions) {
		this.totalInstructions = totalInstructions;
	}

	/**
	 * @return the cacheAccesses
	 */
	public double[] getCacheAccesses() {
		return cacheAccesses;
	}

	/**
	 * @param cacheAccesses the cacheAccesses to set
	 */
	public void setCacheAccesses(double[] cacheAccesses) {
		this.cacheAccesses = cacheAccesses;
	}
	
	
}
