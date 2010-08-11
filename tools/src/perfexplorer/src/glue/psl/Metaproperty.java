/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public abstract class Metaproperty {

	private boolean anyHolds = false;
	private double minSeverity = Double.MAX_VALUE;
	private double maxSeverity = Double.MIN_VALUE;
	private double avgSeverity = 0.0;
	private double stdDevSeverity = 0.0;
	private double minConfidence = Double.MAX_VALUE;
	private double maxConfidence = Double.MIN_VALUE;
	private double avgConfidence = 0.0;
	private double stdDevConfidence = 0.0;
	
	/**
	 * 
	 */
	public Metaproperty() {
		// TODO Auto-generated constructor stub
	}
	
	public void add(Class<Inefficiency> propertyClass, Object[] arguments) {
		// TODO
	}

	public void add(Class[] propertyClasses, Object[] arguments) {
		// TODO
	}
	
	public boolean anyHolds() {
		return this.anyHolds;
	}
	
	public double getMinSeverity() {
		return this.minSeverity;
	}
	
	public double getMaxSeverity() {
		return this.maxSeverity;
	}
	
	public double getAvgSeverity() {
		return this.avgSeverity;
	}
	
	public double getStdDevSeverity() {
		return this.stdDevSeverity;
	}
	
	public double getMinConfidence() {
		return this.minConfidence;
	}
	
	public double getMaxConfidence() {
		return this.maxConfidence;
	}
	
	public double getAvgConfidence() {
		return this.avgConfidence;
	}
	
	public double getStdDevConfidence() {
		return this.stdDevConfidence;
	}
}
