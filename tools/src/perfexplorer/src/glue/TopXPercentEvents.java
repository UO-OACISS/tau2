/**
 * 
 */
package glue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class TopXPercentEvents extends TopXEvents {

	/**
	 * @param input
	 * @param metric
	 * @param type
	 * @param threshold
	 */
	public TopXPercentEvents(PerformanceResult input, String metric, int type,
			double threshold) {
		super(input, metric, type, threshold);
		// convert the threshold to a percentage
		this.threshold = this.threshold / 100.0;
	}

	/**
	 * @param trial
	 * @param metric
	 * @param type
	 * @param threshold
	 */
	public TopXPercentEvents(Trial trial, String metric, int type, double threshold) {
		super(trial, metric, type, threshold);
		// convert the threshold to a percentage
		this.threshold = this.threshold / 100.0;
	}

	/**
	 * @param inputs
	 * @param threshold
	 */
	public TopXPercentEvents(List<PerformanceResult> inputs, String metric, int type, double threshold) {
		super(inputs, metric, type, threshold);
		// convert the threshold to a percentage
		this.threshold = this.threshold / 100.0;
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		// create a HashMap of values to store
		Map<String, Double> values = new HashMap<String, Double>();
		// for each input matrix in the set of inputs
		for (PerformanceResult input : this.inputs) {
			// create a new output result matrix
			PerformanceResult output = new DefaultResult();
			this.outputs.add(output);
			// get the list of event names
			Set<String> events = input.getEvents();
			// get the list of threads
			Set<Integer> threads = input.getThreads();
			// iterate through the threads
			for (Integer thread : threads) {
				// iterate through the events
				double total = 0.0;
				for (String event : events) {
					total += input.getDataPoint(thread, event, this.metric, this.type);
					values.put(event, input.getDataPoint(thread, event, this.metric, this.type));
				}
				// convert the threshold to a percentage
				double percentage = total * threshold;
				Map<String, Double> sorted = Utilities.sortHashMapByValues(values, false);
				for (String event : sorted.keySet()) {
					if (sorted.get(event) < percentage) {
						break;
					}
//					output.putDataPoint(thread, event, this.metric, this.type, sorted.get(event));
					output.putExclusive(thread, event, metric, 
							input.getExclusive(thread, event, metric));
					output.putInclusive(thread, event, metric, 
							input.getInclusive(thread, event, metric));
					output.putCalls(thread, event, input.getCalls(thread, event));
					output.putSubroutines(thread, event, input.getSubroutines(thread, event));
				}
			}
		}
		return outputs;
	}

}
