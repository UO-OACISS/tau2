/**
 * 
 */
package glue;

import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class SmartKMeansOperation extends AbstractPerformanceOperation {

	private Integer maxClusters = 2;
	private String metric;
	private int type;
	private double evaluation = 0.0;
	private double adjustedEvaluation = 0.0;

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
        for (PerformanceResult input : inputs) {
        	for (int i = 1 ; i <= this.maxClusters ; i++) {
        		System.out.println("Clustering with k = " + i);
				KMeansOperation kmeans = new KMeansOperation(input, metric, type, i);
				tmpOutputs = kmeans.processData();
				// 0 - cluster centroids
				// 1 - cluster standard deviations
				// 2 - culster minimums
				// 3 - cluster maximums
				System.out.println("Gap Statistic for " + i + " clusters: " + kmeans.getGapStatistic() + " +/- " + kmeans.getGapStatisticError());
				if (i == 1) {
					outputs = tmpOutputs;
					previousGapStat = kmeans.getGapStatistic();
				} else if ((kmeans.getGapStatistic() - kmeans.getGapStatisticError()) > previousGapStat) {
					// we have a new winner!
					outputs = tmpOutputs;
					previousGapStat = kmeans.getGapStatistic();
				} else {
					// early termination.
					break;
				}
        	}
        }
		return outputs;
	}

}
