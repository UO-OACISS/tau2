/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class LossOfParallelismOverhead extends SimpleProperty {

	/**
	 * 
	 */
	public LossOfParallelismOverhead(RegionSummary summary, RegionSummary rankBasis) {
		severity = summary.getLossOfParallelism() / rankBasis.getExecutionTime();
	}

}
