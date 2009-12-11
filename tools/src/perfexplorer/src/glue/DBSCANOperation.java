/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.clustering.ClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.DBScanClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.KMeansClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.clustering.weka.AnalysisFactory;

/**
 * @author khuck
 *
 */
public class DBSCANOperation extends ClusterOperation {

	private double epsilon = 1.0;

	/**
	 * @param input
	 */
	public DBSCANOperation(PerformanceResult input, String metric, int type, double epsilon) {
		super(input);
		this.metric = metric;
		this.type = type;
		this.epsilon = epsilon;
	}

	/**
	 * @param trial
	 */
	public DBSCANOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public DBSCANOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {

        for (PerformanceResult input : inputs) {
        	System.out.println("instances: " + input.getThreads().size());
        	System.out.println("dimensions: " + input.getEvents().size());
        	this.clusterer = doClustering(input);
			PerformanceResult centroids = new ClusterOutputResult(clusterer.getClusterCentroids(), metric, type);
			PerformanceResult stddevs = new ClusterOutputResult(clusterer.getClusterStandardDeviations(), metric, type);
			PerformanceResult mins = new ClusterOutputResult(clusterer.getClusterMinimums(), metric, type);
			PerformanceResult maxs = new ClusterOutputResult(clusterer.getClusterMaximums(), metric, type); 
			outputs.add(stddevs);
			outputs.add(centroids);
			outputs.add(mins);
			outputs.add(maxs);
			
        	List<String> eventList = new ArrayList<String>(input.getEvents());
    		for(Integer thread : input.getThreads()) {
    			int clusterID = clusterer.clusterInstance(thread.intValue());
    			if (clusterID >= 0) {  // -1 means the point is noise, not clustered
	            	for (String event : eventList) {
	        			evaluation += Math.pow(input.getDataPoint(thread, event, metric, type) - centroids.getDataPoint(clusterID, event, metric, type), 2.0);
	            	}
    			}
        	}
    		evaluation = Math.sqrt(evaluation / this.maxClusters);
    		System.out.println(this.maxClusters + " clusters, Total squared distance from centroids: "+ evaluation);
    		double adjuster = (input.getThreads().size() - this.maxClusters);
    		adjuster = adjuster / input.getThreads().size();
    		adjuster = Math.sqrt(adjuster);
    		adjustedEvaluation = evaluation / adjuster;
    		System.out.println(this.maxClusters + " clusters, Total adjusted squared distance from centroids: "+ adjustedEvaluation);

    		if (computeGapStatistic) {
	    		computeGapStatistic(input, mins, maxs);
    		}

			// create new trials for each cluster
			PerformanceResult clusterIDs = new DefaultResult(input);
			
			for (Integer thread : input.getThreads()) {
    			int clusterID = clusterer.clusterInstance(thread.intValue());
				clusterIDs.putCalls(thread, "Cluster ID", clusterID);
			}	
			outputs.add(clusterIDs);
        }

		return outputs;
	}

	/**
	 * @param factory
	 * @param input
	 * @param eventList
	 */
	protected ClusterInterface doClustering(PerformanceResult input) {
    	List<String> eventList = new ArrayList<String>(input.getEvents());
		RawDataInterface data = AnalysisFactory.createRawData("Cluster Test", eventList, input.getThreads().size(), eventList.size(), null);
		for(Integer thread : input.getThreads()) {
			int eventIndex = 0;
			for (String event : eventList) {
				data.addValue(thread, eventIndex++, input.getDataPoint(thread, event, metric, type));
				if (event.equals(input.getMainEvent())) {
					data.addMainValue(thread, eventIndex-1, input.getDataPoint(thread, event, metric, type));
				}
			}
		}
		DBScanClusterInterface clusterer = AnalysisFactory.createDBScanEngine();
		clusterer.setInputData(data);
		clusterer.setError(epsilon);
		try {
			clusterer.findClusters();
		} catch (Exception e) {
			System.err.println("failure to cluster.");
			System.exit(0);
		}
		return clusterer;
	}

}
