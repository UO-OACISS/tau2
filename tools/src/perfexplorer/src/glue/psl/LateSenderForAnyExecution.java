/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class LateSenderForAnyExecution extends OverheadForAnyExecution {

	/**
	 * @param property
	 * @param parallelExp
	 * @param rankBasis
	 * @param r
	 */
	public LateSenderForAnyExecution(Class<Inefficiency> property, Experiment parallelExp,
			RegionSummary rankBasis, CodeRegion r) {
		super(property, parallelExp, rankBasis, r);
	}

}
