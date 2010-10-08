/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class CommunicationOverheadForAnyExecution extends
		OverheadForAnyExecution {

	/**
	 * @param property
	 * @param parallelExp
	 * @param rankBasis
	 * @param r
	 */
	public CommunicationOverheadForAnyExecution(Class<Inefficiency> property,
			Experiment parallelExp, RegionSummary rankBasis, CodeRegion r) {
		super(property, parallelExp, rankBasis, r);
	}

}
