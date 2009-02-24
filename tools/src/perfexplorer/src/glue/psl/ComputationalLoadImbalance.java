/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class ComputationalLoadImbalance extends LoadImbalance {

	/**
	 * @param e
	 * @param r
	 */
	public ComputationalLoadImbalance(Experiment e, CodeRegion r) {
		super(e, r);
	}

	/* (non-Javadoc)
	 * @see glue.psl.LoadImbalance#getValueOfInterest(glue.psl.RegionSummary)
	 */
	@Override
	protected double getValueOfInterest(RegionSummary rs) {
		return rs.getNumberOfInstructions();
	}

}
