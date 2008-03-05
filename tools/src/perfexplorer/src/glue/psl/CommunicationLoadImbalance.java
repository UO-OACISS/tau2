/**
 * 
 */
package glue.psl;

/**
 * @author khuck
 *
 */
public class CommunicationLoadImbalance extends LoadImbalance {

	/**
	 * @param e
	 * @param r
	 */
	public CommunicationLoadImbalance(Experiment e, CodeRegion r) {
		super(e, r);
	}

	/* (non-Javadoc)
	 * @see glue.psl.LoadImbalance#getValueOfInterest(glue.psl.RegionSummary)
	 */
	@Override
	protected double getValueOfInterest(RegionSummary rs) {
		return rs.getCommunicationTime();
	}

}
