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
public class SplitCommunicationComputationOperation extends	AbstractPerformanceOperation {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7423086277837025258L;
	public final static int COMMUNICATION = 0;
	public final static int COMPUTATION = 1;

	/**
	 * @param input
	 */
	public SplitCommunicationComputationOperation(PerformanceResult input) {
		super(input);
	}

	/**
	 * @param trial
	 */
	public SplitCommunicationComputationOperation(Trial trial) {
		super(trial);
	}

	/**
	 * @param inputs
	 */
	public SplitCommunicationComputationOperation(List<PerformanceResult> inputs) {
		super(inputs);
	}

	/* (non-Javadoc)
	 * @see edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		// iterate over inputs
		for (PerformanceResult input : inputs) {
			PerformanceResult communication = new DefaultResult(input, false);
			PerformanceResult computation = new DefaultResult(input, false);
			PerformanceResult tmp = null;
	
			// iterate over all threads, and find min and max
			for (Integer thread : input.getThreads()) {
				for (String event : input.getEvents()) {
					if (event.startsWith("MPI_"))
						tmp = communication;
					else
						tmp = computation;
					for (String metric : input.getMetrics()) {
						tmp.putExclusive(thread, event, metric, input.getExclusive(thread, event, metric));
						tmp.putInclusive(thread, event, metric, input.getInclusive(thread, event, metric));
					}
					tmp.putCalls(thread, event, input.getCalls(thread, event));
					tmp.putSubroutines(thread, event, input.getSubroutines(thread, event));
				}
			}
			outputs.add(communication);
			outputs.add(computation);
		}
		return outputs;
	}

}
