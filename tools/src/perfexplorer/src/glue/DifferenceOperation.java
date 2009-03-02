/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.List;
import java.util.Map;
import java.util.Set;


import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.common.EngineType;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerException;

/**
 * @author khuck
 *
 */
public class DifferenceOperation extends AbstractPerformanceOperation {

	private PerformanceDifferenceType differenceType = PerformanceDifferenceType.SAME;
	private double performanceRatio = 1.0;
	
	/**
	 * @param input
	 */
	public DifferenceOperation(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public DifferenceOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public DifferenceOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		PerformanceResult baseline = inputs.get(0);
		PerformanceResult comparison = inputs.get(1);
		
		// create a new output result matrix
		PerformanceResult output = new DefaultResult(inputs.get(0), false);
		
		// get the set of threads
		Set<Integer> totalThreads = baseline.getThreads();
		totalThreads.addAll(comparison.getThreads());
		
		// get the set of events
		Set<String> totalEvents = baseline.getEvents();
		totalEvents.addAll(comparison.getEvents());
		
		// get the set of metrics
		Set<String> totalMetrics = baseline.getMetrics();
		totalMetrics.addAll(comparison.getMetrics());
		
		// subtract the comparison from the baseline
		for (Integer thread : totalThreads) {
			for (String event : totalEvents) {
				for (String metric : totalMetrics) {
					output.putExclusive(thread, event, metric, 
							baseline.getExclusive(thread, event, metric) -
							comparison.getExclusive(thread, event, metric));
					output.putInclusive(thread, event, metric, 
							baseline.getInclusive(thread, event, metric) -
							comparison.getInclusive(thread, event, metric));
				}
				output.putCalls(thread, event, 
						baseline.getCalls(thread, event) -
						comparison.getCalls(thread, event));
				output.putSubroutines(thread, event, 
						baseline.getSubroutines(thread, event) -
						comparison.getSubroutines(thread, event));
			}
		}
		
		// for the main thread, find the main event, and the time metric
		// to determine which trial is faster
		Integer thread = 0;
		String event = baseline.getMainEvent();
		String metric = baseline.getTimeMetric();
		
		this.performanceRatio = comparison.getInclusive(thread, event, metric) / baseline.getInclusive(thread, event, metric);
		if (comparison.getInclusive(thread, event, metric) > baseline.getInclusive(thread, event, metric)) {
			this.differenceType = PerformanceDifferenceType.SLOWER;
		} else if (comparison.getInclusive(thread, event, metric) < baseline.getInclusive(thread, event, metric)) {
			this.differenceType = PerformanceDifferenceType.FASTER;
		}

		outputs.add(output);
		return this.outputs;
	}

	public PerformanceDifferenceType getDifferenceType() {
		return differenceType;
	}
	
	public PerformanceResult getBaseline() {
		return inputs.get(0);
	}
	
	public PerformanceResult getComparison() {
		return inputs.get(1);
	}
	
	public String toString() {
		StringBuilder buf = new StringBuilder();
		if (this.differenceType == PerformanceDifferenceType.SAME) {
			buf.append("The comparison trial (");
			buf.append(getComparison().toString());
			buf.append(") and the baseline trial (");
			buf.append(getBaseline().toString());
			buf.append(") have the same execution time.");
		} else {
			buf.append("The comparison trial (");
			buf.append(getComparison().toString());
			buf.append(") is relatively " + this.differenceType.toString() + " than the baseline trial (");
			buf.append(getBaseline().toString());
			buf.append(").");
		}
		return buf.toString();
	}

	/**
	 * @return the performanceRatio
	 */
	public double getPerformanceRatio() {
		return performanceRatio;
	}

	/**
	 * @param performanceRatio the performanceRatio to set
	 */
	public void setPerformanceRatio(double performanceRatio) {
		this.performanceRatio = performanceRatio;
	}
}
