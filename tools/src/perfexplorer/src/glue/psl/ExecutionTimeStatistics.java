/**
 * 
 */
package glue.psl;

/**
 * @author khuck
 *
 */
public class ExecutionTimeStatistics extends Statistics {

	/**
	 * @param iterator
	 */
	public ExecutionTimeStatistics(RegionSummaryIterator iterator) {
		super(iterator);
	}

	/* (non-Javadoc)
	 * @see glue.psl.Statistics#getValue(glue.psl.RegionSummary)
	 */
	@Override
	protected double getValue(RegionSummary summary) {
		return summary.getExecutionTime();
	}
	
}
