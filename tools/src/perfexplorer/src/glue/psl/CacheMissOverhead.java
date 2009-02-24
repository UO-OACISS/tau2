/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class CacheMissOverhead extends SimpleProperty {

	/**
	 * 
	 */
	public CacheMissOverhead(RegionSummary summary, RegionSummary rankBasis, int level) {
		severity = summary.getNumberOfCacheMisses(level) * 
			summary.getExecutionNode().getCacheMissPenalty().get(level) / 
			rankBasis.getExecutionTime();
	}

}
