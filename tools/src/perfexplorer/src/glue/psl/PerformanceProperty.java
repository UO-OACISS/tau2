/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class PerformanceProperty extends SimpleProperty {

	private Experiment experiment = null;
	private CodeRegion codeRegion = null;
	
	/**
	 * 
	 */
	public PerformanceProperty(CodeRegion codeRegion, Experiment experiment) {
		this.codeRegion = codeRegion;
		codeRegion.addPerformanceProperty(this);
		this.experiment = experiment;
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
		this.codeRegion.addPerformanceProperty(this);
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

}
