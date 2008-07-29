/**
 * 
 */
package glue;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.lang.Math;

import common.EngineType;
import common.PerfExplorerException;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class RatioOperation extends AbstractPerformanceOperation {

	private boolean straightRatio = true;
	private String metricSuffix = "";

	/**
	 * @param input
	 */
	public RatioOperation(PerformanceResult numerator, PerformanceResult denominator) {
		super(numerator);
		addInput(denominator);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public RatioOperation(PerformanceResult numerator, PerformanceResult denominator, String metricSuffix) {
		super(numerator);
		addInput(denominator);
		this.metricSuffix = metricSuffix;
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public RatioOperation(Trial numerator, Trial denominator) {
		super(numerator);
		addInput(denominator);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public RatioOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		PerformanceResult numerator = inputs.get(0);
		PerformanceResult denominator = inputs.get(1);
		
		// create a new output result matrix
		PerformanceResult output = new DefaultResult(inputs.get(0).getTrial());
		
		// get the set of threads
		Set<Integer> totalThreads = numerator.getThreads();
		totalThreads.addAll(denominator.getThreads());
		
		// get the set of events
		Set<String> totalEvents = numerator.getEvents();
		totalEvents.addAll(denominator.getEvents());
		
		// get the set of metrics
		Set<String> totalMetrics = numerator.getMetrics();
		totalMetrics.addAll(denominator.getMetrics());
		
		for (Integer thread : totalThreads) {
			for (String event : totalEvents) {
				for (String metric : totalMetrics) {
					// divide the numerator with the denominator
					if (straightRatio) {
						output.putExclusive(thread, event, metric + metricSuffix, 
								numerator.getExclusive(thread, event, metric) /
								denominator.getExclusive(thread, event, metric));
						output.putInclusive(thread, event, metric + metricSuffix, 
								numerator.getInclusive(thread, event, metric) /
								denominator.getInclusive(thread, event, metric));
					// or, divide the *difference* between them with
					// the denominator
					} else {
						output.putExclusive(thread, event, metric + metricSuffix, 
								Math.abs(numerator.getExclusive(thread, event, metric) -
								    denominator.getExclusive(thread, event, metric)) /
								denominator.getExclusive(thread, event, metric));
						output.putInclusive(thread, event, metric + metricSuffix, 
								Math.abs(numerator.getInclusive(thread, event, metric) -
								    denominator.getInclusive(thread, event, metric)) /
								denominator.getInclusive(thread, event, metric));
					}
					if (Double.isNaN(output.getExclusive(thread, event, metric + metricSuffix)))
						output.putExclusive(thread, event, metric + metricSuffix, 0.0);
					if (Double.isNaN(output.getInclusive(thread, event, metric + metricSuffix)))
						output.putInclusive(thread, event, metric + metricSuffix, 0.0);
				}
				// divide the numerator with the denominator
				if (straightRatio) {
					output.putCalls(thread, event, 
							numerator.getCalls(thread, event) /
							denominator.getCalls(thread, event));
					output.putSubroutines(thread, event, 
							numerator.getSubroutines(thread, event) /
							denominator.getSubroutines(thread, event));
				// or, divide the *difference* between them with
				// the denominator
				} else {
					output.putCalls(thread, event, 
							Math.abs(numerator.getCalls(thread, event) -
							     denominator.getCalls(thread, event)) /
							denominator.getCalls(thread, event));
					output.putSubroutines(thread, event, 
							Math.abs(numerator.getSubroutines(thread, event) -
							     denominator.getSubroutines(thread, event)) /
							denominator.getSubroutines(thread, event));
				}
				if (Double.isNaN(output.getCalls(thread, event)))
					output.putCalls(thread, event, 0.0);
				if (Double.isNaN(output.getSubroutines(thread, event)))
					output.putSubroutines(thread, event, 0.0);
			}
		}
		
		outputs.add(output);
		return this.outputs;
	}

	public PerformanceResult getNumerator() {
		return inputs.get(0);
	}
	
	public PerformanceResult getDenominator() {
		return inputs.get(1);
	}
	
	public String toString() {
		StringBuffer buf = new StringBuffer();
		// TODO
		return buf.toString();
	}

	/**
	 * @return whether the operation will perform a straight ratio or not
	 */
	public boolean getStraightRatio() {
		return straightRatio;
	}

	/**
	 * @param straightRatio whether to just divide the numerator with the denominator
	 */
	public void setStraightRatio(boolean straightRatio) {
		this.straightRatio = straightRatio;
	}
}
