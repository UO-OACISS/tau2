/**
 * 
 */
package glue;

import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class DeriveMetricOperation extends AbstractPerformanceOperation {

	public static final String ADD = "+";
	public static final String SUBTRACT = "-";
	public static final String MULTIPLY = "*";
	public static final String DIVIDE = "/";

	private String firstMetric = null;
	private String secondMetric = null;
	private String operation = ADD;
	private String newName = null;
	
	/**
	 * @param input
	 */
	public DeriveMetricOperation(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public DeriveMetricOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public DeriveMetricOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public DeriveMetricOperation(PerformanceResult input, String firstMetric, String secondMetric, String operation) {
		super(input);
		this.firstMetric = firstMetric;
		this.secondMetric = secondMetric;
		this.operation = operation;
		this.newName = "(" + firstMetric + operation + secondMetric + ")";
	}


	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		for (PerformanceResult input : inputs) {
			PerformanceResult output = new DefaultResult(input.getTrial());
			
			for (String event : input.getEvents()) {
				for (Integer thread : input.getThreads()) {
					double value1 = 0.0;
					double value2 = 0.0;
					if (operation.equals(ADD)) {
						value1 = input.getInclusive(thread, event, firstMetric) + input.getInclusive(thread, event, secondMetric);
						value2 = input.getExclusive(thread, event, firstMetric) + input.getExclusive(thread, event, secondMetric);
					} else if (operation.equals(SUBTRACT)) {
						value1 = input.getInclusive(thread, event, firstMetric) - input.getInclusive(thread, event, secondMetric);
						value2 = input.getExclusive(thread, event, firstMetric) - input.getExclusive(thread, event, secondMetric);
					} else if (operation.equals(MULTIPLY)) {
						value1 = input.getInclusive(thread, event, firstMetric) * input.getInclusive(thread, event, secondMetric);
						value2 = input.getExclusive(thread, event, firstMetric) * input.getExclusive(thread, event, secondMetric);
					} else if (operation.equals(DIVIDE)) {
						if (input.getInclusive(thread, event, secondMetric) == 0.0) {
							value1 = 0.0;
						} else {
							value1 = input.getInclusive(thread, event, firstMetric) / input.getInclusive(thread, event, secondMetric);
						}
						if (input.getExclusive(thread, event, secondMetric) == 0.0) {
							value2 = 0.0;
						} else {
							value2 = input.getExclusive(thread, event, firstMetric) / input.getExclusive(thread, event, secondMetric);
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

	/**
	 * @return the newName
	 */
	public String getNewName() {
		return newName;
	}

	/**
	 * @param newName the newName to set
	 */
	public void setNewName(String newName) {
		this.newName = newName;
	}
	
	/**
	 * Check if the derived metric already exists
	 * 
	 */
	public boolean exists() {
		return (inputs.get(0).getMetrics().contains(newName));
	}
}
