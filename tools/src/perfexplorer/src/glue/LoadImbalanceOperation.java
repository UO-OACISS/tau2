/**
	This operation will split events from the trial(s) into two, putting
	communication & syncronization events into one trial, and application
	functions into another.
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class LoadImbalanceOperation extends	AbstractPerformanceOperation {
	
	public static String COMPUTATION = "Computation";
	public static String KERNEL_COMPUTATION = "Kernel Computation";
	public static String COMMUNICATION = "MPI";
	public static String COMMUNICATION_METHOD = "MPI_";
	public static String KERNEL_COMMUNICATION = "Kernel MPI";

	private static String MPI_INIT = "MPI_Init";
	private static String MPI_FINALIZE = "MPI_Finalize";
	public static int MEAN = 0;
	public static int MAX = 1;
	public static int MIN = 2;
	public static int STDDEV = 3;
	public static int COMMUNICATION_EFFICIENCY = 1; // same as max
	public static int LOAD_BALANCE = 4; // ratio of avg / max
	public static int COMPUTATION_SPLITS = 5; // for each thread, an aggregation 
				//of computation and communication
	private boolean percentage = true;

	/**
	 * @param input
	 */
	public LoadImbalanceOperation(PerformanceResult input) {
		super(input);
	}

	/**
	 * @param trial
	 */
	public LoadImbalanceOperation(Trial trial) {
		super(trial);
	}

	/**
	 * @param inputs
	 */
	public LoadImbalanceOperation(List<PerformanceResult> inputs) {
		super(inputs);
	}

	/* (non-Javadoc)
	 * @see edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		// iterate over inputs
		for (PerformanceResult input : inputs) {
			PerformanceResult split = new DefaultResult(input, false);

			// extract the non-MPI events from the trial
			for (Integer thread : input.getThreads()) {
				// initialize
				split.putCalls(thread, COMPUTATION, 1);
				split.putSubroutines(thread, COMPUTATION, 0);
				split.putCalls(thread, COMMUNICATION, 1);
				split.putSubroutines(thread, COMMUNICATION, 0);
				split.putCalls(thread, KERNEL_COMPUTATION, 1);
				split.putSubroutines(thread, KERNEL_COMPUTATION, 0);
				split.putCalls(thread, KERNEL_COMMUNICATION, 1);
				split.putSubroutines(thread, KERNEL_COMMUNICATION, 0);
				for (String metric : input.getMetrics()) {
					// initialize
					split.putExclusive(thread, COMPUTATION, metric, 0.0);
					split.putInclusive(thread, COMPUTATION, metric, 0.0);
					split.putInclusive(thread, COMMUNICATION, metric, 0.0);
					split.putExclusive(thread, COMMUNICATION, metric, 0.0);
					split.putExclusive(thread, KERNEL_COMPUTATION, metric, 0.0);
					split.putInclusive(thread, KERNEL_COMPUTATION, metric, 0.0);
					split.putInclusive(thread, KERNEL_COMMUNICATION, metric, 0.0);
					split.putExclusive(thread, KERNEL_COMMUNICATION, metric, 0.0);
					// get the total runtime for this thread
					double total = input.getInclusive(thread, input.getMainEvent(), metric);
					double totalKernel = total;
					for (String event : input.getEvents()) {
						// get the exclusive time for this event
						double value = input.getExclusive(thread, event, metric);
						if (event.contains(COMMUNICATION_METHOD)) {
							// if MPI, add to MPI running total
							double current = split.getExclusive(thread, COMMUNICATION, metric);
							split.putExclusive(thread, COMMUNICATION, metric, value + current);
							split.putInclusive(thread, COMMUNICATION, metric, value + current);
							if (event.contains(MPI_INIT) || event.contains(MPI_FINALIZE)) {
								totalKernel = totalKernel - value;
							} else {
								current = split.getExclusive(thread, KERNEL_COMMUNICATION, metric);
								split.putExclusive(thread, KERNEL_COMMUNICATION, metric, value + current);
								split.putInclusive(thread, KERNEL_COMMUNICATION, metric, value + current);
							}
						}
					}

					// save the values which include all fuctions
					double communication = split.getExclusive(thread, COMMUNICATION, metric);
					double computation = total - communication;
					if (this.percentage) {
						split.putInclusive(thread, COMPUTATION, metric, computation / total);
						split.putExclusive(thread, COMPUTATION, metric, computation / total);
						split.putInclusive(thread, COMMUNICATION, metric, communication / total);
						split.putExclusive(thread, COMMUNICATION, metric, communication / total);
					} else {
						split.putInclusive(thread, COMPUTATION, metric, computation);
						split.putExclusive(thread, COMPUTATION, metric, computation);
						split.putInclusive(thread, COMMUNICATION, metric, communication);
						split.putExclusive(thread, COMMUNICATION, metric, communication);
					}

					// save the values which ignore init, finalize
					communication = split.getExclusive(thread, KERNEL_COMMUNICATION, metric);
					computation = totalKernel - communication;
					if (this.percentage) {
						split.putInclusive(thread, KERNEL_COMPUTATION, metric, computation / totalKernel);
						split.putExclusive(thread, KERNEL_COMPUTATION, metric, computation / totalKernel );
						split.putInclusive(thread, KERNEL_COMMUNICATION, metric, communication / totalKernel );
						split.putExclusive(thread, KERNEL_COMMUNICATION, metric, communication / totalKernel );
					} else {
						split.putInclusive(thread, KERNEL_COMPUTATION, metric, computation);
						split.putExclusive(thread, KERNEL_COMPUTATION, metric, computation);
						split.putInclusive(thread, KERNEL_COMMUNICATION, metric, communication);
						split.putExclusive(thread, KERNEL_COMMUNICATION, metric, communication);
					}

				}
			}
		    PerformanceAnalysisOperation statMaker = new BasicStatisticsOperation(split, false);
		    List<PerformanceResult> stats = statMaker.processData();
		    outputs.add(stats.get(BasicStatisticsOperation.MEAN));
		    outputs.add(stats.get(BasicStatisticsOperation.MAX));
		    outputs.add(stats.get(BasicStatisticsOperation.MIN));
		    outputs.add(stats.get(BasicStatisticsOperation.STDDEV));

		    // get the ratio between average and max
			// This computes the CommEff term, T_avg/T_max
		    PerformanceAnalysisOperation ratioMaker = new RatioOperation(stats.get(BasicStatisticsOperation.MEAN), stats.get(BasicStatisticsOperation.MAX));
		    outputs.add(ratioMaker.processData().get(0));
		    outputs.add(split);
		}
		return outputs;
	}

	public void setPercentage(boolean percentage) {
		this.percentage = percentage;
	}

	public boolean getPercentage() {
		return this.percentage;
	}

}
