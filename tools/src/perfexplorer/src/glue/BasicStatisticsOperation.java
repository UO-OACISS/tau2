/**
 * 
 */
package glue;

import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.lang.Math;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class BasicStatisticsOperation extends AbstractPerformanceOperation {

	private boolean combined = false;
	
	/**
	 * @param input
	 */
	public BasicStatisticsOperation(PerformanceResult input) {
		super(input);
	}

	/**
	 * @param input
	 */
	public BasicStatisticsOperation(PerformanceResult input, boolean combined) {
		super(input);
		this.combined = combined;
	}

	/**
	 * @param trial
	 */
	public BasicStatisticsOperation(Trial trial, boolean combined) {
		super(trial);
		this.combined = combined;
	}

	/**
	 * @param inputs
	 */
	public BasicStatisticsOperation(List<PerformanceResult> inputs, boolean combined) {
		super(inputs);
		this.combined = combined;
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		// first, check to see if we are combining ALL trials together, 
		// or whether we want the stats for each individual trial.
		if (!combined) {
			return processDataNotCombined();
		}
		// create a new output result matrices
		PerformanceResult total = new TotalResult();
		PerformanceResult mean = new MeanResult();
		PerformanceResult variance = new VarianceResult();
		PerformanceResult stdev = new StDevResult();
		PerformanceResult min = new MinResult();
		PerformanceResult max = new MaxResult();

		Set<Integer> totalThreads = new TreeSet<Integer>();
		Set<String> totalMetrics = new TreeSet<String>();
		Set<String> totalEvents = new TreeSet<String>();

		for (PerformanceResult input : inputs) {
			totalThreads.addAll(input.getThreads());
			totalEvents.addAll(input.getEvents());
			totalMetrics.addAll(input.getMetrics());
		}
		
//		System.out.println("Totals, mins, maxes...");
		// add each input to the total
		for (PerformanceResult input : inputs) {
			for (Integer thread : totalThreads) {
				for (String event : totalEvents) {
					for (String metric : totalMetrics) {
						total.putExclusive(thread, event, metric, 
								total.getExclusive(thread, event, metric) +
								input.getExclusive(thread, event, metric));
						min.putDataPoint(thread, event, metric, MinResult.EXCLUSIVE, 
								input.getExclusive(thread, event, metric));
						max.putDataPoint(thread, event, metric, MaxResult.EXCLUSIVE, 
								input.getExclusive(thread, event, metric));
						total.putInclusive(thread, event, metric, 
								total.getInclusive(thread, event, metric) +
								input.getInclusive(thread, event, metric));
						min.putDataPoint(thread, event, metric, MinResult.INCLUSIVE,
								input.getInclusive(thread, event, metric));
						max.putDataPoint(thread, event, metric, MaxResult.INCLUSIVE,
								input.getInclusive(thread, event, metric));
					}
					total.putCalls(thread, event, 
							total.getCalls(thread, event) +
							input.getCalls(thread, event));
					min.putDataPoint(thread, event, null, MinResult.CALLS,
							input.getCalls(thread, event));
					max.putDataPoint(thread, event, null, MaxResult.CALLS,
							input.getCalls(thread, event));
					total.putSubroutines(thread, event, 
							total.getSubroutines(thread, event) +
							input.getSubroutines(thread, event));
					min.putDataPoint(thread, event, null, MinResult.SUBROUTINES,
							input.getSubroutines(thread, event));
					max.putDataPoint(thread, event, null, MaxResult.SUBROUTINES, 
							input.getSubroutines(thread, event));
				}
			}
		}
//		System.out.println("Means...");
		int numInputs = inputs.size();
		// divide the total to get the mean
		for (Integer thread : totalThreads) {
			for (String event : totalEvents) {
				for (String metric : totalMetrics) {
					mean.putExclusive(thread, event, metric, 
							total.getExclusive(thread, event, metric) / numInputs);
					mean.putInclusive(thread, event, metric, 
							total.getInclusive(thread, event, metric) / numInputs);
				}
				mean.putCalls(thread, event, 
						total.getCalls(thread, event) / numInputs);
				mean.putSubroutines(thread, event, 
						total.getSubroutines(thread, event) / numInputs);
			}
		}

//		System.out.println("variances...");
		// do the sums for the stddev
		for (PerformanceResult input : inputs) {
			for (Integer thread : totalThreads) {
				for (String event : totalEvents) {
					for (String metric : totalMetrics) {
						variance.putExclusive(thread, event, metric,
								variance.getExclusive(thread, event, metric) + 
								java.lang.Math.pow(mean.getExclusive(thread, event, metric) -
								input.getExclusive(thread, event, metric),2.0));
						variance.putInclusive(thread, event, metric,
								variance.getInclusive(thread, event, metric) +
								java.lang.Math.pow(mean.getInclusive(thread, event, metric) -
								input.getInclusive(thread, event, metric),2.0));
					}
					variance.putCalls(thread, event,
							variance.getCalls(thread, event) +
							java.lang.Math.pow(mean.getCalls(thread, event) -
							input.getCalls(thread, event),2.0));
					variance.putSubroutines(thread, event,
							variance.getSubroutines(thread, event) +
							java.lang.Math.pow(mean.getSubroutines(thread, event) -
							input.getSubroutines(thread, event),2.0));
				}
			}
		}

//		System.out.println("Standard Deviations...");
		numInputs = inputs.size() - 1;
		// divide the variances by n-1
		for (Integer thread : totalThreads) {
			for (String event : totalEvents) {
				for (String metric : totalMetrics) {
					variance.putExclusive(thread, event, metric,
							variance.getExclusive(thread, event, metric) / numInputs);
					stdev.putExclusive(thread, event, metric,
							java.lang.Math.sqrt(variance.getExclusive(thread, event, metric)));
					variance.putInclusive(thread, event, metric, 
							variance.getInclusive(thread, event, metric) / numInputs);
					stdev.putInclusive(thread, event, metric, 
							java.lang.Math.sqrt(variance.getInclusive(thread, event, metric)));
				}
				variance.putCalls(thread, event, 
						variance.getCalls(thread, event) / numInputs);
				stdev.putCalls(thread, event, 
						java.lang.Math.sqrt(variance.getCalls(thread, event)));
				variance.putSubroutines(thread, event, 
						variance.getSubroutines(thread, event) / numInputs);
				stdev.putSubroutines(thread, event, 
						java.lang.Math.sqrt(variance.getSubroutines(thread, event)));
			}
		}
//		System.out.println("Done.");

		outputs.add(total);
		outputs.add(mean);
		outputs.add(variance);
		outputs.add(stdev);
		outputs.add(min);
		outputs.add(max);
		return this.outputs;
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processDataNotCombined() {
		for (PerformanceResult input : inputs) {
			// create a new output result matrices
			PerformanceResult total = new TotalResult();
			PerformanceResult mean = new MeanResult();
			PerformanceResult variance = new VarianceResult();
			PerformanceResult stdev = new StDevResult();
			PerformanceResult min = new MinResult();
			PerformanceResult max = new MaxResult();
	
			Set<Integer> totalThreads = new TreeSet<Integer>();
			Set<String> totalMetrics = new TreeSet<String>();
			Set<String> totalEvents = new TreeSet<String>();
	
			totalThreads.addAll(input.getThreads());
			totalEvents.addAll(input.getEvents());
			totalMetrics.addAll(input.getMetrics());
		
//			System.out.println("Totals, mins, maxes...");
			// add each input to the total
			for (Integer thread : totalThreads) {
				for (String event : totalEvents) {
					for (String metric : totalMetrics) {
						total.putExclusive(0, event, metric, 
								total.getExclusive(0, event, metric) +
								input.getExclusive(thread, event, metric));
						min.putDataPoint(0, event, metric, MinResult.EXCLUSIVE,
								input.getExclusive(thread, event, metric));
						max.putDataPoint(0, event, metric, MaxResult.EXCLUSIVE, 
								input.getExclusive(thread, event, metric));
						total.putInclusive(0, event, metric, 
								total.getInclusive(0, event, metric) +
								input.getInclusive(thread, event, metric));
						min.putDataPoint(0, event, metric, MinResult.INCLUSIVE, 
								input.getInclusive(thread, event, metric));
						max.putDataPoint(0, event, metric, MaxResult.INCLUSIVE,
								input.getInclusive(thread, event, metric));
					}
					total.putCalls(0, event, 
							total.getCalls(0, event) +
							input.getCalls(thread, event));
					min.putDataPoint(0, event, null, MinResult.CALLS,
							input.getCalls(thread, event));
					max.putDataPoint(0, event, null, MaxResult.CALLS,
							input.getCalls(thread, event));
					total.putSubroutines(0, event, 
							total.getSubroutines(0, event) +
							input.getSubroutines(thread, event));
					min.putDataPoint(0, event, null, MinResult.SUBROUTINES,
							input.getSubroutines(thread, event));
					max.putDataPoint(0, event, null, MaxResult.SUBROUTINES,
							input.getSubroutines(thread, event));
				}
			}
			
	//		System.out.println("Means...");
			int numInputs = inputs.size();
			// divide the total to get the mean
			for (String event : totalEvents) {
				for (String metric : totalMetrics) {
					mean.putExclusive(0, event, metric, 
							total.getExclusive(0, event, metric) / input.getThreads().size());
					mean.putInclusive(0, event, metric, 
							total.getInclusive(0, event, metric) / input.getThreads().size());
				}
				mean.putCalls(0, event, 
						total.getCalls(0, event) / input.getThreads().size());
				mean.putSubroutines(0, event, 
						total.getSubroutines(0, event) / input.getThreads().size());
			}

//		System.out.println("variances...");
		// do the sums for the stddev
			for (Integer thread : totalThreads) {
				for (String event : totalEvents) {
					for (String metric : totalMetrics) {
						variance.putExclusive(0, event, metric,
								variance.getExclusive(0, event, metric) + 
								java.lang.Math.pow(mean.getExclusive(0, event, metric) -
								input.getExclusive(thread, event, metric),2.0));
						variance.putInclusive(0, event, metric,
								variance.getInclusive(0, event, metric) +
								java.lang.Math.pow(mean.getInclusive(0, event, metric) -
								input.getInclusive(thread, event, metric),2.0));
					}
					variance.putCalls(0, event,
							variance.getCalls(0, event) +
							java.lang.Math.pow(mean.getCalls(0, event) -
							input.getCalls(thread, event),2.0));
					variance.putSubroutines(0, event,
							variance.getSubroutines(0, event) +
							java.lang.Math.pow(mean.getSubroutines(0, event) -
							input.getSubroutines(thread, event),2.0));
				}
			}

//		System.out.println("Standard Deviations...");
			// divide the variances by n-1
			for (String event : totalEvents) {
				for (String metric : totalMetrics) {
					variance.putExclusive(0, event, metric,
							variance.getExclusive(0, event, metric) / input.getThreads().size()-1);
					stdev.putExclusive(0, event, metric,
							java.lang.Math.sqrt(variance.getExclusive(0, event, metric)));
					variance.putInclusive(0, event, metric, 
							variance.getInclusive(0, event, metric) / input.getThreads().size()-1);
					stdev.putInclusive(0, event, metric, 
							java.lang.Math.sqrt(variance.getInclusive(0, event, metric)));
				}
				variance.putCalls(0, event, 
						variance.getCalls(0, event) / input.getThreads().size()-1);
				stdev.putCalls(0, event, 
						java.lang.Math.sqrt(variance.getCalls(0, event)));
				variance.putSubroutines(0, event, 
						variance.getSubroutines(0, event) / input.getThreads().size()-1);
				stdev.putSubroutines(0, event, 
						java.lang.Math.sqrt(variance.getSubroutines(0, event)));
			}
	//		System.out.println("Done.");
	
			outputs.add(total);
			outputs.add(mean);
			outputs.add(variance);
			outputs.add(stdev);
			outputs.add(min);
			outputs.add(max);
		}
		return this.outputs;
	}

	/**
	 * @return the combined
	 */
	public boolean isCombined() {
		return combined;
	}

	/**
	 * @param combined the combined to set
	 */
	public void setCombined(boolean combined) {
		this.combined = combined;
	}
}
