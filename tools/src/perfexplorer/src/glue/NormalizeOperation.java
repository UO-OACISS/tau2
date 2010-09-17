/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class NormalizeOperation extends AbstractPerformanceOperation {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1383799654279373763L;

	/**
	 * @param input
	 */
	public NormalizeOperation(PerformanceResult input) {
		super(input);
	}

	/**
	 * @param trial
	 */
	public NormalizeOperation(Trial trial) {
		super(trial);
	}

	/**
	 * @param inputs
	 */
	public NormalizeOperation(List<PerformanceResult> inputs) {
		super(inputs);
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		for (PerformanceResult input : inputs) {
			// create a new output result matrices
			PerformanceResult normalized = new DefaultResult(input, false);
			PerformanceResult min = new MinResult(input, false);
			PerformanceResult max = new MaxResult(input, false);
			min.setIgnoreWarnings(true);
			max.setIgnoreWarnings(true);
	
			double calls = 0;
			// iterate over all threads, and find min and max
			for (Integer thread : input.getThreads()) {
				for (String event : input.getEvents()) {
					calls = input.getCalls(thread, event);
					// if not called, don't bother - everything is zero
					if (calls > 0.0) {
						for (String metric : input.getMetrics()) {
							min.putDataPoint(0, event, metric, MinResult.EXCLUSIVE,
								input.getExclusive(thread, event, metric));
							max.putDataPoint(0, event, metric, MaxResult.EXCLUSIVE, 
								input.getExclusive(thread, event, metric));
							min.putDataPoint(0, event, metric, MinResult.INCLUSIVE, 
								input.getInclusive(thread, event, metric));
							max.putDataPoint(0, event, metric, MaxResult.INCLUSIVE,
								input.getInclusive(thread, event, metric));
						}
						min.putDataPoint(0, event, null, MinResult.CALLS, calls);
						max.putDataPoint(0, event, null, MaxResult.CALLS, calls);
						min.putDataPoint(0, event, null, MinResult.SUBROUTINES,
							input.getSubroutines(thread, event));
						max.putDataPoint(0, event, null, MaxResult.SUBROUTINES,
							input.getSubroutines(thread, event));
					}
				}
			}
			
			// iterate over all threads, and normalize to the range
			for (Integer thread : input.getThreads()) {
				for (String event : input.getEvents()) {
					if (input.getCalls(thread, event) > 0) {
						for (String metric : input.getMetrics()) {
							normalized.putExclusive(thread, event, metric,
								(input.getExclusive(thread, event, metric) - 
								 min.getExclusive(0, event, metric)) / 
								(max.getExclusive(0, event, metric) -
								 min.getExclusive(0, event, metric)));
							normalized.putInclusive(thread, event, metric,
								(input.getInclusive(thread, event, metric) - 
								 min.getInclusive(0, event, metric)) / 
								(max.getInclusive(0, event, metric) -
								 min.getInclusive(0, event, metric)));
						}
						normalized.putCalls(thread, event,
							(input.getCalls(thread, event) -
							 min.getCalls(0, event)) / 
							(max.getCalls(0, event) -
							 min.getCalls(0, event)));
						normalized.putSubroutines(thread, event,
							(input.getSubroutines(thread, event) -
							 min.getSubroutines(0, event)) / 
							(max.getSubroutines(0, event) -
							 min.getSubroutines(0, event)));
					}
				}
			}

			outputs.add(normalized);
		}
		return this.outputs;
	}
}
