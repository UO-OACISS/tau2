/**
 * 
 */
package glue.psl;

/**
 * @author khuck
 *
 */
public class SynchronizationOverhead extends SimpleProperty {

	public SynchronizationOverhead (RegionSummary summary, RegionSummary rankBasis) {
		severity = summary.getSynchronizationTime() / rankBasis.getExecutionTime();
	}
}
