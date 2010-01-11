/**
	This operation will split events from the trial(s) into two, putting
	communication & syncronization events into one trial, and application
	functions into another.
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class SplitTrialClusters extends	AbstractPerformanceOperation {

	private List<PerformanceResult> clusterResults = null;

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
			PerformanceResult clusters[] = new PerformanceResult[numClusters];
			for (int i = 0 ; i < numClusters ; i++) {
				clusters[i] = new DefaultResult(input, false);
			}
	
			// iterate over all threads
			for (Integer thread : input.getThreads()) {
				int clusterID = (int)clusterResults.get((resultIndex*5) + 4).getCalls(thread, "Cluster ID");
				for (String event : input.getEvents()) {
					if ((clusterID >= 0) && (input.getCalls(thread, event) > 0)) {
						for (String metric : input.getMetrics()) {
							clusters[clusterID].putExclusive(thread, event, metric, input.getExclusive(thread, event, metric));
							clusters[clusterID].putInclusive(thread, event, metric, input.getInclusive(thread, event, metric));
						}
						clusters[clusterID].putCalls(thread, event, input.getCalls(thread, event));
						clusters[clusterID].putSubroutines(thread, event, input.getSubroutines(thread, event));
					}
				}
			}
			for (int i = 0 ; i < numClusters ; i++) {
				outputs.add(clusters[i]);
			}
			resultIndex++;
		}
		return outputs;
	}

}
