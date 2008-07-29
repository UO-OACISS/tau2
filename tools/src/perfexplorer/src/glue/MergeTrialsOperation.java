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
public class MergeTrialsOperation extends AbstractPerformanceOperation {

	/**
	 * @param input
	 */
	public MergeTrialsOperation(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public MergeTrialsOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public MergeTrialsOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		this.outputs = new ArrayList<PerformanceResult>();
		PerformanceResult output = new DefaultResult(inputs.get(0).getTrial());
		for (PerformanceResult input : inputs) {
			for (String event : input.getEvents()) {
				for (Integer threadIndex : input.getThreads()) {
					for (String metric : input.getMetrics()) {
						output.putExclusive(threadIndex, event, metric, 
								input.getExclusive(threadIndex, event, metric));
						output.putInclusive(threadIndex, event, metric, 
								input.getInclusive(threadIndex, event, metric));
					}
					output.putCalls(threadIndex, event, input.getCalls(threadIndex, event));
					output.putSubroutines(threadIndex, event, input.getSubroutines(threadIndex, event));
				}
			}
		}
		outputs.add(output);
		return outputs;
	}

}
