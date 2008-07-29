/**
 * 
 */
package glue;

import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;
import java.lang.Math;

/**
 * @author khuck
 *
 */
public class LogarithmicOperation extends AbstractPerformanceOperation {

	private double base = 2;
	
	/**
	 * @param input
	 */
	public LogarithmicOperation(PerformanceResult input) {
		super(input);
	}

	/**
	 * @param trial
	 */
	public LogarithmicOperation(Trial trial) {
		super(trial);
	}

	/**
	 * @param inputs
	 */
	public LogarithmicOperation(List<PerformanceResult> inputs) {
		super(inputs);
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		List<PerformanceResult> outputs = new ArrayList<PerformanceResult>();
		
		for (PerformanceResult input : inputs) {
			PerformanceResult output = new DefaultResult(input.getTrial());
			outputs.add(output);
			for (Integer thread : input.getThreads()) {
				for (String event : input.getEvents()) {
					for (String metric : input.getMetrics()) {
						output.putExclusive(thread, event, metric,
								Math.log(input.getExclusive(thread, event, metric))/
								Math.log(this.base));
						output.putInclusive(thread, event, metric,
								Math.log(input.getInclusive(thread, event, metric))/
								Math.log(this.base));
					}
					output.putCalls(thread, event, 
							input.getCalls(thread, event));
					output.putSubroutines(thread, event, 
							input.getSubroutines(thread, event));
				}
			}
		}
		return outputs;
	}

	public double getBase() {
		return base;
	}

	public void setBase(double base) {
		this.base = base;
	}
}
