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
public class ExtractUserEventOperation extends AbstractPerformanceOperation {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2645125410181847065L;
	private List<String> events = null;

	/**
	 * @param input
	 */
	public ExtractUserEventOperation(PerformanceResult input, List<String> events) {
		super(input);
		this.events = events;
	}

	/**
	 * @param input
	 */
	public ExtractUserEventOperation(PerformanceResult input, String event) {
		super(input);
		this.events = new ArrayList<String>();
		this.events.add(event);
	}

	/**
	 * @param trial
	 */
	public ExtractUserEventOperation(Trial trial, List<String> events) {
		super(trial);
		this.events = events;
	}

	/**
	 * @param inputs
	 */
	public ExtractUserEventOperation(List<PerformanceResult> inputs, List<String> events) {
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
				for (Integer threadIndex : input.getThreads()) {
					output.putUsereventNumevents(threadIndex, event,
						input.getUsereventNumevents(threadIndex, event));
					output.putUsereventMax(threadIndex, event,
						input.getUsereventMax(threadIndex, event));
					output.putUsereventMin(threadIndex, event,
						input.getUsereventMin(threadIndex, event));
					output.putUsereventMean(threadIndex, event,
						input.getUsereventMean(threadIndex, event));
					output.putUsereventSumsqr(threadIndex, event,
						input.getUsereventSumsqr(threadIndex, event));
				}
			}
			output.updateEventMap();
		}
		return outputs;
	}

	/**
	 * @return the event
	 */
	public List<String> getUserEvent() {
		return events;
	}

	/**
	 * @param event the event to set
	 */
	public void setUserEvent(List<String> event) {
		this.events = event;
	}

}
