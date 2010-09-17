/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class SmartKMeansOperation extends AbstractPerformanceOperation {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4012041148374200416L;
	private Integer maxClusters = 2;
	private String metric;
	private int type;
//	private double evaluation = 0.0;
//	private double adjustedEvaluation = 0.0;

	/**
	 * @param input
	 */
	public SmartKMeansOperation(PerformanceResult input, String metric, int type, int maxClusters) {
		super(input);
		this.metric = metric;
		this.type = type;
		this.maxClusters = maxClusters;
	}

	/**
	 * @param input
	 */
	public SmartKMeansOperation(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public SmartKMeansOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public SmartKMeansOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		List<PerformanceResult> tmpOutputs = null;
		double previousGapStat = 0.0;
		double previousGapIndex = 0;
        for (PerformanceResult input : inputs) {
        	for (int i = 1 ; i <= this.maxClusters ; i++) {
//        		System.out.println("Clustering with k = " + i);
				ClusterOperation kmeans = new KMeansOperation(input, metric, type, i);
				kmeans.setComputeGapStatistic(true);
				tmpOutputs = kmeans.processData();
				// 0 - cluster centroids
				// 1 - cluster standard deviations
				// 2 - culster minimums
				// 3 - cluster maximums
				System.out.println("Gap Statistic for " + i + " clusters: " + kmeans.getGapStatistic() + " +/- " + kmeans.getGapStatisticError());
        		System.out.println("Previous Gap statistic: " + previousGapStat);
				double newGapStat = kmeans.getGapStatistic();
				double newGapError = kmeans.getGapStatisticError();
				if (i == 1) {
					outputs = tmpOutputs;
					previousGapStat = newGapStat;
					previousGapIndex = i;
				// make sure we are at least more accurate than noise!
				} else if (previousGapStat < 0.0 || 
					newGapStat-newGapError < 0.0 || 
					((newGapStat - (1*newGapError)) > previousGapStat)) {
					// we have a new winner!
					outputs = tmpOutputs;
					previousGapStat = newGapStat;
					previousGapIndex = i;
				} else if (newGapStat < 0.0 && previousGapStat > 0.0){
					// we have a new winner!  - we started going negative...
					outputs = tmpOutputs;
					previousGapStat = newGapStat;
					previousGapIndex = i;
					// early termination.
					break;
				} else if (previousGapIndex < i-1 || (newGapStat < 0.0 && previousGapStat > 0.0)){
					// early termination.
					break;
				}
        	}
        }
		return outputs;
	}

}
