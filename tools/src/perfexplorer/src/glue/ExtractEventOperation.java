/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class ExtractEventOperation extends AbstractPerformanceOperation {
	/**
	 * 
	 */
	private static final long serialVersionUID = -786694241087019821L;
	private List<String> events = null;

	/**
	 * @param input
	 */
	public ExtractEventOperation(PerformanceResult input, List<String> events) {
		super(input);
		this.events = events;
	}

	/**
	 * @param input
	 */
	public ExtractEventOperation(PerformanceResult input, String event) {
		super(input);
		this.events = new ArrayList<String>();
		this.events.add(event);
	}

	/**
	 * @param trial
	 */
	public ExtractEventOperation(Trial trial, List<String> events) {
		super(trial);
		this.events = events;
	}

	/**
	 * @param inputs
	 */
	public ExtractEventOperation(List<PerformanceResult> inputs, List<String> events) {
		super(inputs);
		this.events = events;
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		this.outputs = new ArrayList<PerformanceResult>();
		for (PerformanceResult input : inputs) {
			PerformanceResult output = new DefaultResult(input, false);
			outputs.add(output);
			for (String event : events) {
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

	/**
	 * @return the event
	 */
	public List<String> getEvent() {
		return events;
	}

	/**
	 * @param event the event to set
	 */
	public void setEvent(List<String> event) {
		this.events = event;
	}

}
