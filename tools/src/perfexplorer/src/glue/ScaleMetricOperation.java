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
public class ScaleMetricOperation extends AbstractPerformanceOperation {

	public static final String ADD = "+";
	public static final String SUBTRACT = "-";
	public static final String MULTIPLY = "*";
	public static final String DIVIDE = "/";

	private String metric = null;
	private Double value = 0.0;
	private String operation = ADD;
	
	/**
	 * @param input
	 */
	public ScaleMetricOperation(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public ScaleMetricOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public ScaleMetricOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public ScaleMetricOperation(PerformanceResult input, String metric, Double value, String operation) {
		super(input);
		this.metric = metric;
		this.value = value;
		this.operation = operation;
	}


	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		String newName = "(" + metric + operation + value.toString() + ")";
		for (PerformanceResult input : inputs) {
			PerformanceResult output = new DefaultResult(input, false);
			
			for (String event : input.getEvents()) {
				for (Integer thread : input.getThreads()) {
					double value1 = 0.0;
					double value2 = 0.0;
					if (operation.equals(ADD)) {
						value1 = input.getInclusive(thread, event, metric) + value;
						value2 = input.getExclusive(thread, event, metric) + value;
					} else if (operation.equals(SUBTRACT)) {
						value1 = input.getInclusive(thread, event, metric) - value;
						value2 = input.getExclusive(thread, event, metric) - value;
					} else if (operation.equals(MULTIPLY)) {
						value1 = input.getInclusive(thread, event, metric) * value;
						value2 = input.getExclusive(thread, event, metric) * value;
					} else if (operation.equals(DIVIDE)) {
						if (value == 0.0) {
							value1 = 0.0;
						} else {
							value1 = input.getInclusive(thread, event, metric) / value;
							value2 = input.getExclusive(thread, event, metric) / value;
						}
					}
					output.putInclusive(thread, event, newName, value1);
					output.putExclusive(thread, event, newName, value2);
					output.putCalls(thread, event, input.getCalls(thread, event));
					output.putSubroutines(thread, event, input.getSubroutines(thread, event));
				}
			}
			outputs.add(output);
		}
		return outputs;
	}

}
