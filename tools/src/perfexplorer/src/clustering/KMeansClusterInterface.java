/*
 * Created on Mar 16, 2005
 *
 */
package clustering;

/**
 * @author khuck
 *
 */
public interface KMeansClusterInterface {
	/**
	 * This method performs the K means clustering
	 * @throws ClusterException
	 */
	public void findClusters() throws ClusterException;
	/**
	 * This method gets the ith ClusterDescription object
	 * @param i
	 * @return a ClusterDescription
	 * @throws ClusterException
	 */
	public ClusterDescription getClusterDescription(int i) throws ClusterException;
	/**
	 * Set the value of K (nuber of clusters)
	 * @param k
	 */
	public void setK(int k);
	/**
	 * Get the value of K (number of clusters)
	 * @return
	 */
	public int getK();
	/**
	 * Set the indices of the initial centers for K means.
	 * This method is necessary to get repeatable clustering results.
	 * @param indexes
	 */
	public void setInitialCenters(int[] indexes);
	
	/**
	 * Sets the input data for the clustering operation.
	 * @param inputData
	 */
	public void setInputData(RawDataInterface inputData);
	
	public RawDataInterface getClusterCentroids();
	
	public RawDataInterface getClusterStandardDeviations();
	/**
	 * Reset method, for resetting the cluster.  If a user loads
	 * this object with data, and then does several clusterings
	 * with several K values, then we need a reset method.
	 */
	public void reset();
	
	public void doPCA(boolean doPCA);
	
	public int[] getClusterSizes();
	
	public int clusterInstance(int i);
	
	public int getNumInstances();
}
