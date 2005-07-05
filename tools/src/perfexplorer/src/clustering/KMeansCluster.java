/*
 * Created on Mar 16, 2005
 *
 */
package clustering;

/**
 * @author khuck
 *
 */
public interface KMeansCluster {
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
	 * Set the indices of the initial centers for K means.
	 * This method is necessary to get repeatable clustering results.
	 * @param indexes
	 */
	public void setInitialCenters(int[] indexes);
}
