/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class SynchronizationLoadImbalance extends LoadImbalance {

	/**
	 * @param e
	 * @param r
	 */
	public SynchronizationLoadImbalance(Experiment e, CodeRegion r) {
		super(e, r);
	}

	/* (non-Javadoc)
	 * @see glue.psl.LoadImbalance#getValueOfInterest(glue.psl.RegionSummary)
	 */
	@Override
	protected double getValueOfInterest(RegionSummary rs) {
		return rs.getSynchronizationTime();
	}

}
