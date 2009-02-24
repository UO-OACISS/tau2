/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering;


 /**
  * This interface is used to define the methods to implement a hierarchical 
  * clustering class.
  *
  * <P>CVS $Id: KMeansClusterInterface.java,v 1.6 2009/02/24 00:53:35 khuck Exp $</P>
  * @author khuck
  * @version 0.1
  * @since   0.1
  *
  */
public interface KMeansClusterInterface {
    /**
     * This method performs the K means clustering
     * 
     * @throws ClusterException
     */
    public void findClusters() throws ClusterException;
    /**
     * This method gets the ith ClusterDescription object
     * 
     * @param i
     * @return a ClusterDescription
     * @throws ClusterException
     */
    public ClusterDescription getClusterDescription(int i) 
        throws ClusterException;
    /**
     * Set the value of K (nuber of clusters)
     * 
     * @param k
     */
    public void setK(int k);
    /**
     * Get the value of K (number of clusters)
     * 
     * @return
     */
    public int getK();
    /**
     * Set the indices of the initial centers for K means.
     * This method is necessary to get repeatable clustering results.
     * 
     * @param indexes
     */
    public void setInitialCenters(int[] indexes);
    
    /**
     * Sets the input data for the clustering operation.
     * 
     * @param inputData
     */
    public void setInputData(RawDataInterface inputData);

    /**
     * Method to get the cluster centroids (averages).
     * 
     * @return
     */
    public RawDataInterface getClusterCentroids();

    /**
     * Method to get the cluster minimum values.
     * 
     * @return
     */
    public RawDataInterface getClusterMinimums();

    /**
     * Method to get the cluster maximum values.
     * 
     * @return
     */
    public RawDataInterface getClusterMaximums();
    
    public RawDataInterface getClusterStandardDeviations();

    /**
     * Reset method, for resetting the cluster.  If a user loads
     * this object with data, and then does several clusterings
     * with several K values, then we need a reset method.
     *
     */
    public void reset();
    
    // TODO - remove this!
    public void doPCA(boolean doPCA);
    
    /**
     * Method to get the number of individuals in each cluster.
     * 
     * @return
     */
    public int[] getClusterSizes();
    
    /**
     * Method to get the cluster ID for the cluster that contains
     * individual "i".
     * 
     * @param i
     * @return
     */
    public int clusterInstance(int i);
    
    /**
     * Get the number of individuals that we are clustering.
     * 
     * @return
     */
    public int getNumInstances();
    
    /**
     * Initialize the K means with good initial centers, rather than random
     * This is slower, but more accurate.
     * 
     * @param b
     */
	public void doSmartInitialization(boolean b);
}
