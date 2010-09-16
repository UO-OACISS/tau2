/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.List;
import java.util.Set;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.clustering.ClusterInterface;

/**
 * @author khuck
 *
 */
public abstract class ClusterOperation extends AbstractPerformanceOperation {

	/**
	 * 
	 */
	private static final long serialVersionUID = 432943010443795674L;
	protected Integer maxClusters = 2;
	protected String metric;
	protected int type;
	protected double evaluation = 0.0;
	protected double adjustedEvaluation = 0.0;
	protected ClusterInterface clusterer = null;
	protected int B = 10;
	protected double gapStatistic = 0.0;
	protected double gapStatisticError = 0.0;
	protected boolean computeGapStatistic = false;
	protected int[] clusterIDs = null;

	/**
	 * 
	 */
	public ClusterOperation() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public ClusterOperation(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public ClusterOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public ClusterOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		// TODO Auto-generated method stub
		return null;
	}

	protected PerformanceResult generateReferenceDataset(PerformanceResult input, PerformanceResult mins,
			PerformanceResult maxs) {
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
	protected double computeErrorMeasure(PerformanceResult input, ClusterInterface tmpClusterer, boolean print) {
			Set<String> eventList = input.getEvents();
			// compute the Gap Statistic!
			int numThreads = input.getThreads().size();
			double[] distance = new double[this.maxClusters];
			int[] sizes = tmpClusterer.getClusterSizes();
			if (sizes.length < this.maxClusters)
				System.err.println("\n**************** not enough clusters! **************\n");
			double w_k = 0.0;
			// for each cluster...
	/*		System.out.print(this.maxClusters + " " + sizes.length + " ");
			for(int x = 0 ; x < sizes.length ; x++)
				System.out.print(sizes[x] + " ");
			System.out.println("");*/
			for(int r = 0 ; r < sizes.length ; r++) {
				distance[r] = 0.0;
				// get the sum of the pairwise distances for all points in this cluster
				// question... all distances, twice?  The paper is ambiguous.
				for(int i = 0 ; i < numThreads ; i++) {
					int id1 = tmpClusterer.clusterInstance(i);
					for(int iprime = i ; iprime < numThreads ; iprime++) {
						int id2 = tmpClusterer.clusterInstance(iprime);
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
	   		if (print) System.out.println("Error Measure: " + w_k);
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

	/**
	 * @param input
	 * @param mins
	 * @param maxs
	 */
	protected void computeGapStatistic(PerformanceResult input, PerformanceResult mins, PerformanceResult maxs) {
		System.out.println("Computing Gap Statistic");
		double w_k = computeErrorMeasure(input, this.clusterer, false);
		//System.out.println("Error Measure: " + w_k);
		double[] ref_w_k = new double[B];
		double l_bar = 0.0;
		// generate uniform distribution reference dataset
		for (int b = 0 ; b < B ; b++) {
			PerformanceResult reference = generateReferenceDataset(input, mins, maxs);
			ClusterInterface tmpClusterer = doClustering(reference);
			ref_w_k[b] = computeErrorMeasure(reference, tmpClusterer, false);
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

	protected abstract ClusterInterface doClustering(PerformanceResult reference);

}
