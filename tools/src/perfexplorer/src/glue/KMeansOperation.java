/**
 * 
 */
package glue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.lang.Math;

import server.PerfExplorerServer;
import clustering.AnalysisFactory;
import clustering.KMeansClusterInterface;
import clustering.RawDataInterface;

import common.RMIPerfExplorerModel;

import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class KMeansOperation extends AbstractPerformanceOperation {

	private Integer maxClusters = 2;
	private String metric;
	private int type;
	private double evaluation = 0.0;
	private double adjustedEvaluation = 0.0;

	/**
	 * @param input
	 */
	public KMeansOperation(PerformanceResult input, String metric, int type, int maxClusters) {
		super(input);
		this.metric = metric;
		this.type = type;
		this.maxClusters = maxClusters;
	}

	/**
	 * @param trial
	 */
	public KMeansOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public KMeansOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
	    AnalysisFactory factory = null;
	    PerfExplorerServer server = null;
        server = PerfExplorerServer.getServer();
        factory = server.getAnalysisFactory();

        for (PerformanceResult input : inputs) {
        	List<String> eventList = new ArrayList<String>(input.getEvents());
        	System.out.println("instances: " + input.getThreads().size());
        	System.out.println("dimensions: " + input.getEvents().size());
        	RawDataInterface data = factory.createRawData("Cluster Test", eventList, input.getThreads().size(), eventList.size());
    		for(Integer thread : input.getThreads()) {
            	int eventIndex = 0;
            	for (String event : eventList) {
        			data.addValue(thread, eventIndex++, input.getDataPoint(thread, event, metric, type));
        			if (event.equals(input.getMainEvent())) {
        				data.addMainValue(thread, eventIndex-1, input.getDataPoint(thread, event, metric, type));
        			}
        		}
        	}
			KMeansClusterInterface clusterer = factory.createKMeansEngine();
			clusterer.setInputData(data);
			clusterer.setK(this.maxClusters);
			try {
				clusterer.findClusters();
			} catch (Exception e) {
				System.err.println("failure to cluster.");
				System.exit(0);
			}
			PerformanceResult centroids = new ClusterOutputResult(clusterer.getClusterCentroids(), metric, type);
			outputs.add(centroids);
			outputs.add(new ClusterOutputResult(clusterer.getClusterStandardDeviations(), metric, type));
			outputs.add(new ClusterOutputResult(clusterer.getClusterMinimums(), metric, type));
			outputs.add(new ClusterOutputResult(clusterer.getClusterMaximums(), metric, type));
			
    		for(Integer thread : input.getThreads()) {
    			int clusterID = clusterer.clusterInstance(thread.intValue());
            	for (String event : eventList) {
        			evaluation += Math.pow(input.getDataPoint(thread, event, metric, type) - centroids.getDataPoint(clusterID, event, metric, type), 2.0);
            	}
        	}
    		evaluation = Math.sqrt(evaluation / this.maxClusters);
    		System.out.println(this.maxClusters + " clusters, Total squared distance from centroids: "+ evaluation);
    		double adjuster = (input.getThreads().size() - this.maxClusters);
    		adjuster = adjuster / input.getThreads().size();
    		adjuster = Math.sqrt(adjuster);
    		adjustedEvaluation = evaluation / adjuster;
    		System.out.println(this.maxClusters + " clusters, Total adjusted squared distance from centroids: "+ adjustedEvaluation);
        }

		return outputs;
	}

	/**
	 * @return the maxClusters
	 */
	public Integer getMaxClusters() {
		return maxClusters;
	}

	/**
	 * @param maxClusters the maxClusters to set
	 */
	public void setMaxClusters(Integer maxClusters) {
		this.maxClusters = maxClusters;
	}

	/**
	 * @return the metric
	 */
	public String getMetric() {
		return metric;
	}

	/**
	 * @param metric the metric to set
	 */
	public void setMetric(String metric) {
		this.metric = metric;
	}

	/**
	 * @return the type
	 */
	public int getType() {
		return type;
	}

	/**
	 * @param type the type to set
	 */
	public void setType(int type) {
		this.type = type;
	}

}
