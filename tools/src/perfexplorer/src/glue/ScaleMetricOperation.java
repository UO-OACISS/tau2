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
	private String newName = null;
	private boolean rightValue = false;

	
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
		this.rightValue = true;
		newName = "(" + metric + operation + value.toString() + ")";
		if (!(input.getMetrics().contains(metric)))
			System.err.println("\n\n *** ERROR: Trial does not have a metric named: " + metric + " ***\n\n");
	}
	/**
	 * @param input
	 */
	public ScaleMetricOperation(PerformanceResult input, Double value,String metric,  String operation) {
		super(input);
		this.metric = metric;
		this.value = value;
		this.operation = operation;
		this.rightValue = false;
		newName = "(" + value.toString() + operation +metric+  ")";
		if (!(input.getMetrics().contains(metric)))
			System.err.println("\n\n *** ERROR: Trial does not have a metric named: " + metric + " ***\n\n");
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		for (PerformanceResult input : inputs) {
			PerformanceResult output = new DefaultResult(input, false);
			double leftInclusive,rightInclusive,leftExclusive,rightExclusive;

			for (String event : input.getEvents()) {
				for (Integer thread : input.getThreads()) {
					if(rightValue){
						 leftInclusive = input.getInclusive(thread, event, metric);			 
						 leftExclusive = input.getExclusive(thread, event, metric);
						 rightInclusive = value;
						 rightExclusive = value;
					}else{
						 leftInclusive = value;		 
						 leftExclusive = value;
						 rightInclusive = input.getInclusive(thread, event, metric);	
						 rightExclusive = input.getExclusive(thread, event, metric);
					}
					
					double value1 = 0.0;
					double value2 = 0.0;
					if (operation.equals(ADD)) {
						value1 = leftInclusive + rightInclusive;
						value2 = leftExclusive + rightExclusive;
					} else if (operation.equals(SUBTRACT)) {
						value1 = leftInclusive - rightInclusive;
						value2 = leftExclusive - rightExclusive;
					} else if (operation.equals(MULTIPLY)) {
						value1 = leftInclusive * rightInclusive;
						value2 = leftExclusive * rightExclusive;
					} else if (operation.equals(DIVIDE)) {
						if (rightInclusive==0|| rightExclusive==0) {
							value1 = 0.0;
							value2 = 0.0;
						} else {
							value1 = leftInclusive / rightInclusive;
							value2 = leftExclusive / rightExclusive;
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

	public String getNewName() {
		return newName;
	}

	public void setNewName(String newName) {
		this.newName = newName;
	}

}
