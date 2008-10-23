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
	private KMeansClusterInterface clusterer = null;
	private int B = 10;  // number of reference data sets to build for gap statistic computation
	private double gapStatistic = 0.0;
	private double gapStatisticError = 0.0;
	private boolean computeGapStatistic = false;
	
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
//        	System.out.println("instances: " + input.getThreads().size());
//        	System.out.println("dimensions: " + input.getEvents().size());
        	this.clusterer = doClustering(factory, input);
			PerformanceResult centroids = new ClusterOutputResult(clusterer.getClusterCentroids(), metric, type);
			outputs.add(centroids);
			outputs.add(new ClusterOutputResult(clusterer.getClusterStandardDeviations(), metric, type));
			PerformanceResult mins = new ClusterOutputResult(clusterer.getClusterMinimums(), metric, type);
			PerformanceResult maxs = new ClusterOutputResult(clusterer.getClusterMaximums(), metric, type); 
			outputs.add(mins);
			outputs.add(maxs);
			
        	List<String> eventList = new ArrayList<String>(input.getEvents());
    		for(Integer thread : input.getThreads()) {
    			int clusterID = clusterer.clusterInstance(thread.intValue());
            	for (String event : eventList) {
        			evaluation += Math.pow(input.getDataPoint(thread, event, metric, type) - centroids.getDataPoint(clusterID, event, metric, type), 2.0);
            	}
        	}
    		evaluation = Math.sqrt(evaluation / this.maxClusters);
//    		System.out.println(this.maxClusters + " clusters, Total squared distance from centroids: "+ evaluation);
    		double adjuster = (input.getThreads().size() - this.maxClusters);
    		adjuster = adjuster / input.getThreads().size();
    		adjuster = Math.sqrt(adjuster);
    		adjustedEvaluation = evaluation / adjuster;
//    		System.out.println(this.maxClusters + " clusters, Total adjusted squared distance from centroids: "+ adjustedEvaluation);

    		if (computeGapStatistic) {
//	    		System.out.println("Computing Gap Statistic");
	    		double w_k = computeErrorMeasure(input, this.clusterer, true);
	    		//System.out.println("Error Measure: " + w_k);
	    		double[] ref_w_k = new double[B];
	    		double l_bar = 0.0;
	    		// generate uniform distribution reference dataset
	    		for (int b = 0 ; b < B ; b++) {
	    			PerformanceResult reference = generateReferenceDataset(input, mins, maxs);
	    			KMeansClusterInterface tmpClusterer = doClustering(factory, reference);
	        		ref_w_k[b] = computeErrorMeasure(reference, tmpClusterer, true);
	        		// we are computing a sum, so sum
	        		l_bar += ref_w_k[b];
	    		}
	    		// the sum is divided by the number of reference data sets
	    		l_bar = l_bar / B;
	    		//System.out.println("Error Measure (reference): " + l_bar);
	    		// COMPUTE THE GAP STATISTIC!
	    		this.gapStatistic  = l_bar - w_k;
	    		// now, compute the sd_k term
	    		double sd_k = 0.0;
	    		for (int b = 0 ; b < B ; b++) {
	    			sd_k += Math.pow((ref_w_k[b] - l_bar), 2.0);
	    		}
	    		sd_k = sd_k / B;
	    		sd_k = Math.pow(sd_k, 0.5);
	    		// which is used to compute the gapStatisticError term, which is our error for this reference
	    		this.gapStatisticError  = sd_k * (Math.sqrt(1+(1/B)));
    		}
        }

		return outputs;
	}

	/**
	 * @param factory
	 * @param input
	 * @param eventList
	 */
	private KMeansClusterInterface doClustering(AnalysisFactory factory, PerformanceResult input) {
    	List<String> eventList = new ArrayList<String>(input.getEvents());
		RawDataInterface data = factory.createRawData("Cluster Test", eventList, input.getThreads().size(), eventList.size(), null);
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
		clusterer.doSmartInitialization(true);
		clusterer.setK(this.maxClusters);
		try {
			clusterer.findClusters();
		} catch (Exception e) {
			System.err.println("failure to cluster.");
			System.exit(0);
		}
		return clusterer;
	}

	private PerformanceResult generateReferenceDataset(PerformanceResult input, PerformanceResult mins, PerformanceResult maxs) {
		PerformanceResult reference = new DefaultResult();
		for (String event : input.getEvents()) {
			// get the min min
			double min = Double.MAX_VALUE;
			for (Integer thread : mins.getThreads()) {
				if (min > mins.getDataPoint(thread, event, metric, type)) {
					min = mins.getDataPoint(thread, event, metric, type);
				}
			}
			// get the max max
			double max = Double.MIN_VALUE;
			for (Integer thread : maxs.getThreads()) {
				if (max < maxs.getDataPoint(thread, event, metric, type)) {
					max = maxs.getDataPoint(thread, event, metric, type);
				}
			}
			double range = max - min;
//			System.out.println(event + " Range: " + min + " - " + max);
			// now that we have the range, we can generate random data in that range.
			for (Integer thread : input.getThreads()) {
				double value = (Math.random() * range) + min;
				reference.putDataPoint(thread, event, metric, type, value);
			}
		}
		return reference;
	}

	/**
	 * @param input
	 * @param eventList
	 */
	private double computeErrorMeasure(PerformanceResult input, KMeansClusterInterface clusterer, boolean print) {
		Set<String> eventList = input.getEvents();
		// compute the Gap Statistic!
		int numThreads = input.getThreads().size();
		double[] distance = new double[this.maxClusters];
		int[] sizes = clusterer.getClusterSizes();
		if (sizes.length < this.maxClusters)
			System.err.println("\n**************** not enough clusters! **************\n");
		double w_k = 0.0;
		// for each cluster...
//		System.out.print(this.maxClusters + " " + sizes.length + " ");
//		for(int x = 0 ; x < sizes.length ; x++)
//			System.out.print(sizes[x] + " ");
//		System.out.println("");
		for(int r = 0 ; r < sizes.length ; r++) {
			distance[r] = 0.0;
			// get the sum of the pairwise distances for all points in this cluster
			// question... all distances, twice?  The paper is ambiguous.
			for(int i = 0 ; i < numThreads ; i++) {
				int id1 = clusterer.clusterInstance(i);
				for(int iprime = i ; iprime < numThreads ; iprime++) {
					int id2 = clusterer.clusterInstance(iprime);
					// they are in the same cluster, so get the distance between them.
					if (r == id1 && id1 == id2) {
		            	for (String event : eventList) {
		        			distance[r] += Math.pow(input.getDataPoint(i, event, metric, type) - input.getDataPoint(iprime, event, metric, type), 2.0);
		            	}
					}
				}
			}
			// ambiguity in the paper... 
			w_k += distance[r] / (2 * sizes[r]);
			//w_k += distance[r] / sizes[r];
		}
   		//if (print) System.out.println("Error Measure: " + w_k);
		w_k = Math.log(w_k);
		return w_k;
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

	/**
	 * @return the gapStatistic
	 */
	public double getGapStatistic() {
		return gapStatistic;
	}

	/**
	 * @return the gapStatisticError
	 */
	public double getGapStatisticError() {
		return gapStatisticError;
	}

	public boolean isComputeGapStatistic() {
		return computeGapStatistic;
	}

	public void setComputeGapStatistic(boolean computeGapStatistic) {
		this.computeGapStatistic = computeGapStatistic;
	}

}
