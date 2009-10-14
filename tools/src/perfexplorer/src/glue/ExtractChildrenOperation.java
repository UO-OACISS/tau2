/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class ExtractChildrenOperation extends AbstractPerformanceOperation {

	private String parentEvent = null;
	
	/**
	 * 
	 */
	public ExtractChildrenOperation() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public ExtractChildrenOperation(PerformanceResult input, String parentEvent) {
		super(input);
		this.parentEvent = parentEvent.trim();
	}

	/**
	 * @param trial
	 */
	public ExtractChildrenOperation(Trial trial, String parentEvent) {
		super(trial);
		this.parentEvent = parentEvent.trim();
	}

	/**
	 * @param inputs
	 */
	public ExtractChildrenOperation(List<PerformanceResult> inputs, String parentEvent) {
		super(inputs);
		this.parentEvent = parentEvent.trim();
	}

	/* (non-Javadoc)
	 * @see edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		this.outputs = new ArrayList<PerformanceResult>();
		for (PerformanceResult input : inputs) {
			input.setIgnoreWarnings(true);
			PerformanceResult output = new DefaultResult(input, false);
			outputs.add(output);
			for (String event : input.getEvents()) {
				StringTokenizer st = new StringTokenizer(event, "=>");
				boolean found = false;
				String child = null;
				while (st.hasMoreTokens()) {
					String tmp = st.nextToken().trim();
					// compare with both short and full name
					if (Utilities.shortenEventName(tmp).equalsIgnoreCase(this.parentEvent) ||
							tmp.equalsIgnoreCase(this.parentEvent)) {
						if (st.countTokens() == 1) {
							found = true;
							child = st.nextToken();
							break;
						}
					}
				}
				if (found) {
					for (String metric : input.getMetrics()) {
						for (Integer threadIndex : input.getThreads()) {
							output.putExclusive(threadIndex, child, metric, 
									input.getExclusive(threadIndex, child, metric));
							output.putInclusive(threadIndex, child, metric, 
									input.getInclusive(threadIndex, child, metric));
							output.putCalls(threadIndex, child, input.getCalls(threadIndex, child));
							output.putSubroutines(threadIndex, child, input.getSubroutines(threadIndex, child));
						}
					}
				}
			}
			output.updateEventMap();
		}
		return outputs;
	}

	public String getParentEvent() {
		return parentEvent;
	}

	public void setParentEvent(String parentEvent) {
		this.parentEvent = parentEvent;
	}


}
