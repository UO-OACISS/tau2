/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class LateSender extends SimpleProperty {

	/**
	 * 
	 */
	public LateSender(RegionSummary summary, RegionSummary rankBasis) {
		severity = summary.getReceiveTime() / rankBasis.getExecutionTime();
	}
}
