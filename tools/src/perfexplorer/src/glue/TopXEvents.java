/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * This is an implementation of the AbstractPerformanceOperation class which will perform
 * dimension reduction on the data.
 * 
 * <P>CVS $Id: TopXEvents.java,v 1.7 2009/02/24 00:53:40 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0 
 */
public class TopXEvents extends AbstractPerformanceOperation implements Serializable {

	protected Double threshold = 0.0;
	protected String metric = null;
	protected Integer type = 0;
	protected List<String> sortedEventNames = new ArrayList<String>();
	
	/**
	 * @param input
	 */
	public TopXEvents(PerformanceResult input, String metric, int type, double threshold) {
		super(input);
		this.threshold = threshold;
		this.metric = metric;
		this.type = type;
	}

	/**
	 * @param trial
	 */
	public TopXEvents(Trial trial, String metric, int type, double threshold) {
		super(trial);
		this.threshold = threshold;
		this.metric = metric;
		this.type = type;
	}

	/**
	 * @param inputs
	 */
	public TopXEvents(List<PerformanceResult> inputs, String metric, int type, double threshold) {
		super(inputs);
		this.threshold = threshold;
		this.metric = metric;
		this.type = type;
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		// create a HashMap of values to store
		Map<String, Double> values = new HashMap<String, Double>();
		// for each input matrix in the set of inputs
		
		for (PerformanceResult input : inputs) {
			// this is cheating, because we are only looking at the main thread...
			Integer thread = 0;
			// iterate through the events
			for (String event : input.getEvents()) {
				values.put(event, input.getDataPoint(thread, event, this.metric, this.type));
			}
			Map<String, Double> sorted = Utilities.sortHashMapByValues(values, false);
			int i = 1;
			for (String event : sorted.keySet()) {
				sortedEventNames.add(event);
				if (++i > threshold) {
					break;
				}
			}
		}
			
		for (PerformanceResult input : inputs) {
			// create a new output result matrix
			PerformanceResult output = new DefaultResult(input, false);
			outputs.add(output);

			for (String event : sortedEventNames) {
				for (String metric : input.getMetrics()) {
					for (Integer threadIndex : input.getThreads()) {
						output.putExclusive(threadIndex, event, metric, 
								input.getExclusive(threadIndex, event, metric));
						output.putInclusive(threadIndex, event, metric, 
								input.getInclusive(threadIndex, event, metric));
						output.putCalls(threadIndex, event, input.getCalls(threadIndex, event));
						output.putSubroutines(threadIndex, event, input.getSubroutines(threadIndex, event));
					}
				}
			}
			output.updateEventMap();
		}
		return outputs;
	}

	public List<String> getSortedEventNames() {
		return sortedEventNames;
	}
}
