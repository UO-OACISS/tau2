/**
 * 
 */
package glue.psl;

/**
 * @author khuck
 *
 */
public class ControlOfParallelismOverhead extends SimpleProperty {

	/**
	 * 
	 */
	public ControlOfParallelismOverhead(RegionSummary summary, RegionSummary rankBasis) {
		severity = summary.getControlOfParallelism() / rankBasis.getExecutionTime();
	}

}
