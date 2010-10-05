/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class SynchronizationOverheadForAnyExecution extends
		OverheadForAnyExecution {

	/**
	 * @param property
	 * @param parallelExp
	 * @param rankBasis
	 * @param r
	 */
	public SynchronizationOverheadForAnyExecution(Class<Inefficiency> property,
			Experiment parallelExp, RegionSummary rankBasis, CodeRegion r) {
		super(property, parallelExp, rankBasis, r);
	}

}
