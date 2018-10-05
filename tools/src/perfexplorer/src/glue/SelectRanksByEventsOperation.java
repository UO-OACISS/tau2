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
public class SelectRanksByEventsOperation extends AbstractPerformanceOperation {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8592649905701916890L;
	/**
	 * 
	 */

	private List<String> events;
	private boolean exclude;
	/**
	 * @param input
	 */
	public SelectRanksByEventsOperation(PerformanceResult input, List<String> events, boolean exclude) {
		super(input);
		this.exclude = exclude;
		this.events=events;
	}

	/**
	 * @param trial
	 */
	public SelectRanksByEventsOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public SelectRanksByEventsOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		this.outputs = new ArrayList<PerformanceResult>();
		for (PerformanceResult input : inputs) {
			PerformanceResult output = new DefaultResult(input, false);
			outputs.add(output);
			for (Integer thread : input.getThreads()) {
				boolean exit=false;
				for(String event:events) {
					Double numCalls = input.getCalls(thread,event);
					//If we're excluding and there is more than 0, we don't want this thread
					if(exclude && numCalls>0) {
						exit=true;
						break;
					}
					//If we're not excluding we're including. If there are no calls we don't want this thread.
					else if(!exclude && numCalls == 0) {
						exit=true;
						break;
					}
				}
				if(exit) {
					continue;
				}
				for (String event : input.getEvents()) {
					for (String metric : input.getMetrics()) {
						output.putExclusive(thread, event, metric, 
								input.getExclusive(thread, event, metric));
						output.putInclusive(thread, event, metric, 
								input.getInclusive(thread, event, metric));
					}
					output.putCalls(thread, event, input.getCalls(thread, event));
					output.putSubroutines(thread, event, input.getSubroutines(thread, event));
				}
				output.updateEventMap();
			}
		}
		return outputs;
	}



}

