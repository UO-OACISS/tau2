/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class ExtractMetricOperation extends AbstractPerformanceOperation {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3545212529943079604L;
	List<String> metrics = null;
	
	/**
	 * @param input
	 */
	public ExtractMetricOperation(PerformanceResult input, List<String> metrics) {
		super(input);
		this.metrics = metrics;
	}

	/**
	 * @param trial
	 */
	public ExtractMetricOperation(Trial trial, List<String> metrics) {
		super(trial);
		this.metrics = metrics;
	}

	/**
	 * @param inputs
	 */
	public ExtractMetricOperation(List<PerformanceResult> inputs, List<String> metrics) {
		super(inputs);
		this.metrics = metrics;
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		this.outputs = new ArrayList<PerformanceResult>();
		for (PerformanceResult input : inputs) {
			PerformanceResult output = new DefaultResult(input, false);
			outputs.add(output);
			for (String event : input.getEvents()) {
				for (String metric : metrics) {
					for (Integer threadIndex : input.getThreads()) {
						output.putExclusive(threadIndex, event, metric, 
								input.getExclusive(threadIndex, event, metric));
						output.putInclusive(threadIndex, event, metric, 
								input.getInclusive(threadIndex, event, metric));
						output.putCalls(threadIndex, event, input.getCalls(threadIndex, event));
						output.putSubroutines(threadIndex, event, input.getSubroutines(threadIndex, event));
					}
				}
			}
			output.updateEventMap();
		}
		return outputs;
	}

	/**
	 * @return the metric
	 */
	public List<String> getMetric() {
		return metrics;
	}

	/**
	 * @param metrics the List of metric names to extract
	 */
	public void setMetric(List<String> metrics) {
		this.metrics = metrics;
	}

}
