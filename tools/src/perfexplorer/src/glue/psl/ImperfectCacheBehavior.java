/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class ImperfectCacheBehavior extends SimpleProperty {

	/**
	 * 
	 */
	public ImperfectCacheBehavior(RegionSummary summary, int level) {
		severity = summary.getNumberOfCacheMisses(level) /
			summary.getNumberOfCacheAccesses(level);
	}

}
