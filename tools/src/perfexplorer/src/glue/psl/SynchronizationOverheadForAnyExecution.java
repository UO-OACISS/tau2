/**
 * 
 */
package glue.psl;

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
	public SynchronizationOverheadForAnyExecution(Class property,
			Experiment parallelExp, RegionSummary rankBasis, CodeRegion r) {
		super(property, parallelExp, rankBasis, r);
	}

}
