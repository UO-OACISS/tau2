/**
	This operation will split events from the trial(s) into two, putting
	communication & syncronization events into one trial, and application
	functions into another.
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.List;

/**
 * @author khuck
 *
 */
public class SplitTrialClusters extends	AbstractPerformanceOperation {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7815535603874537215L;
	private List<PerformanceResult> clusterResults = null;
	private boolean includeNoisePoints = false;

	/**
	 * @param input
	 */
	public SplitTrialClusters(PerformanceResult input, List<PerformanceResult> clusterResults) {
		super(input);
		this.clusterResults = clusterResults;
	}

	/**
	 * @param inputs
	 */
	public SplitTrialClusters(List<PerformanceResult> inputs, List<PerformanceResult> clusterResults) {
		super(inputs);
		this.clusterResults = clusterResults;
	}

	/* (non-Javadoc)
	 * @see edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		// iterate over inputs
		int resultIndex = 0;
		for (PerformanceResult input : inputs) {
			int numClusters = clusterResults.get(resultIndex*5).getThreads().size();
			List<PerformanceResult> clusters = new ArrayList<PerformanceResult>(numClusters);
			for (int i = 0 ; i < numClusters ; i++) {
				clusters.add(i, new DefaultResult(input, false));
			}
			int noiseIndex = numClusters;
	
			// iterate over all threads
			for (Integer thread : input.getThreads()) {
				int clusterID = (int)clusterResults.get((resultIndex*5) + 4).getCalls(thread, "Cluster ID");
				// if we want noise points in clusters by themselves, 
				// make a cluster for this thread
				if (includeNoisePoints && clusterID < 0) {
					noiseIndex = clusters.size();
					if (clusters.size() == noiseIndex) {
						clusters.add(noiseIndex, new DefaultResult(input, false));
					}
				}
				for (String event : input.getEvents()) {
					// if this event was called by this thread, then save it 
					// in the correct cluster (if not a noise point)
					if ((clusterID >= 0) && (input.getCalls(thread, event) > 0)) {
						for (String metric : input.getMetrics()) {
							clusters.get(clusterID).putExclusive(thread, event, metric, input.getExclusive(thread, event, metric));
							clusters.get(clusterID).putInclusive(thread, event, metric, input.getInclusive(thread, event, metric));
						}
						clusters.get(clusterID).putCalls(thread, event, input.getCalls(thread, event));
						clusters.get(clusterID).putSubroutines(thread, event, input.getSubroutines(thread, event));
					}
					// if this event was called by this thread, then save it 
					// in the correct cluster (if a noise point)
					else if (includeNoisePoints && (input.getCalls(thread, event) > 0)) {
						for (String metric : input.getMetrics()) {
							clusters.get(noiseIndex).putExclusive(thread, event, metric, input.getExclusive(thread, event, metric));
							clusters.get(noiseIndex).putInclusive(thread, event, metric, input.getInclusive(thread, event, metric));
						}
						clusters.get(noiseIndex).putCalls(thread, event, input.getCalls(thread, event));
						clusters.get(noiseIndex).putSubroutines(thread, event, input.getSubroutines(thread, event));
					}
				}
			}
			for (int i = 0 ; i < clusters.size() ; i++) {
				outputs.add(clusters.get(i));
				System.out.println(clusters.get(i).getThreads());
			}
			resultIndex++;
		}
		return outputs;
	}

	public void setIncludeNoisePoints(boolean includeNoisePoints) {
		this.includeNoisePoints = includeNoisePoints;
	}

	public boolean getIncludeNoisePoints() {
		return this.includeNoisePoints;
	}

}
