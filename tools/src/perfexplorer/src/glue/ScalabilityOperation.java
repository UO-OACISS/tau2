/**
 * 
 */
package glue;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.analysis.Scalability;

import java.util.List;
import java.util.Set;

/**
 * @author khuck
 *
 */
public class ScalabilityOperation extends AbstractPerformanceOperation {

	// TODO: fix this, so that the RULES figure out what kind of scaling it is.
	private ScalabilityResult.Measure measure = ScalabilityResult.Measure.SPEEDUP;
	private ScalabilityResult.Scaling scaling = ScalabilityResult.Scaling.STRONG;
	
	/**
	 * @return the measure
	 */
	public ScalabilityResult.Measure getMeasure() {
		return measure;
	}

	/**
	 * @param measure the measure to set
	 */
	public void setMeasure(ScalabilityResult.Measure measure) {
		this.measure = measure;
	}

	/**
	 * @return the scaling
	 */
	public ScalabilityResult.Scaling getScaling() {
		return scaling;
	}

	/**
	 * @param scaling the scaling to set
	 */
	public void setScaling(ScalabilityResult.Scaling scaling) {
		this.scaling = scaling;
	}

	/**
	 * @param input
	 */
	public ScalabilityOperation(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public ScalabilityOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public ScalabilityOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		// validate the list of inputs
/*		for (PerformanceResult input : inputs) {
			if (!(input instanceof TrialMeanResult)) {
				throw (new IllegalArgumentException("Scalability analysis requires TrialMeanResult inputs."));
			}
		}
		TrialMeanResult baseline = (TrialMeanResult)inputs.get(0);
*/
		PerformanceResult baseline = inputs.get(0);
		
		// get the set of threads
		Set<Integer> totalThreads = baseline.getThreads();
		totalThreads.addAll(baseline.getThreads());
		
		// get the set of events
		Set<String> totalEvents = baseline.getEvents();
		totalEvents.addAll(baseline.getEvents());
		
		// get the set of metrics
		Set<String> totalMetrics = baseline.getMetrics();
		totalMetrics.addAll(baseline.getMetrics());
		
		for (PerformanceResult input : inputs) {
//			TrialMeanResult comparison = (TrialMeanResult)input;
			PerformanceResult comparison = input;
			
			// don't compare the baseline to itself
			if (baseline == comparison) {
				continue;
			}

			// create a new output result matrix
			ScalabilityResult output = new ScalabilityResult(input, false);

			// get the ratio of threads between the trials
			double ratio = comparison.getOriginalThreads() / baseline.getOriginalThreads();
			output.setIdealRatio(ratio);
			
			// divide the comparison with the baseline
			for (String event : totalEvents) {
				for (String metric : totalMetrics) {
					output.putExclusive(0, event, metric, 
							baseline.getExclusive(0, event, metric) /
							comparison.getExclusive(0, event, metric));
					output.putInclusive(0, event, metric, 
							baseline.getInclusive(0, event, metric) /
							comparison.getInclusive(0, event, metric));
				}
				output.putCalls(0, event, 
						baseline.getCalls(0, event) /
						comparison.getCalls(0, event));
				output.putSubroutines(0, event, 
						baseline.getSubroutines(0, event) /
						comparison.getSubroutines(0, event));
			}

			// for the main thread, find the main event, and the time metric
			// to get the main scalability measure
			Integer thread = 0;
			String event = baseline.getMainEvent();
			String metric = baseline.getTimeMetric();
			
			output.setActualRatio(baseline.getInclusive(thread, event, metric) / comparison.getInclusive(thread, event, metric));
			output.setMainEvent(event);
			output.setTimeMetric(metric);
			
			outputs.add(output);
		}
		return this.outputs;
	}

	public PerformanceResult getBaseline() {
		return inputs.get(0);
	}
	
	public PerformanceResult getComparison() {
		return inputs.get(1);
	}
	
	public String toString() {
		StringBuffer buf = new StringBuffer();
		buf.append("ScalabilityOperation.toString() not yet supported.");
		return buf.toString();
	}

}
