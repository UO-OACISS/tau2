/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

import java.util.List;

/**
 * @author khuck
 *
 */
public class AlternativeNonScalability extends Metaproperty {

	private double severity;
	
	/**
	 * 
	 */
	public AlternativeNonScalability(Experiment baseExp, List<Experiment> parallelExps, CodeRegion r) {
		for (Experiment parExp : parallelExps) {
			add(Inefficiency.class, new Object[]{baseExp, parExp, r});
			severity = getMaxSeverity() - getAvgSeverity();
		}
	}
	
	public boolean holds() {
		return severity > 0;
	}
	
	public double getSeverity() {
		return this.severity;
	}
	
	public double getConfidence() {
		return 1.0;
	}

}
