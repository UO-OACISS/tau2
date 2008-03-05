/**
 * 
 */
package glue.psl;

/**
 * @author khuck
 *
 */
public class CommunicationOverhead extends SimpleProperty {

	/**
	 * 
	 */
	public CommunicationOverhead(RegionSummary summary, RegionSummary rankBasis) {
		severity = summary.getCommunicationTime() / rankBasis.getExecutionTime();
	}

}
