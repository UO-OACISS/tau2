/**
 * 
 */
package glue;

import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class ExtractRankOperation extends AbstractPerformanceOperation {

	private Integer threadIndex;
	/**
	 * @param input
	 */
	public ExtractRankOperation(PerformanceResult input, int threadIndex) {
		super(input);
		this.threadIndex = threadIndex;
	}

	/**
	 * @param trial
	 */
	public ExtractRankOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public ExtractRankOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		this.outputs = new ArrayList<PerformanceResult>();
		for (PerformanceResult input : inputs) {
			PerformanceResult output = new DefaultResult(input.getTrial());
			outputs.add(output);
			for (String event : input.getEvents()) {
				for (String metric : input.getMetrics()) {
					output.putExclusive(0, event, metric, 
							input.getExclusive(threadIndex, event, metric));
					output.putInclusive(0, event, metric, 
							input.getInclusive(threadIndex, event, metric));
				}
				output.putCalls(0, event, input.getCalls(threadIndex, event));
				output.putSubroutines(0, event, input.getSubroutines(threadIndex, event));
			}
		}
		return outputs;
	}

	/**
	 * @return the threadIndex
	 */
	public Integer getThreadIndex() {
		return threadIndex;
	}

	/**
	 * @param threadIndex the threadIndex to set
	 */
	public void setThreadIndex(Integer threadIndex) {
		this.threadIndex = threadIndex;
	}

}
