/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering.weka;

import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import edu.uoregon.tau.perfexplorer.clustering.ClusterException;
import edu.uoregon.tau.perfexplorer.clustering.KMeansClusterInterface;

/**
 * This class is used as a list of names and values to describe 
 * a cluster created during some type of clustering operation.
 * 
 * <P>CVS $Id: WekaKMeansCluster.java,v 1.11 2009/11/18 17:45:36 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since 0.1
 *
 */
public class WekaKMeansCluster extends WekaAbstractCluster implements KMeansClusterInterface {

	// the number of clusters to find
	private int k = 0;

	private SimpleKMeans kmeans = null;
	
	/**
	 * Default constructor - package protected
	 */
	WekaKMeansCluster() {
		super();
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansCluster#findClusters()
	 */
	public void findClusters() throws ClusterException {
		//assert instances != null : instances;
		try {
			this.kmeans = new SimpleKMeans();
			kmeans.setNumClusters(k);
			Instances localInstances = null;

			if (this.doPCA) {
				localInstances = handlePCA(instances);
			} else localInstances = this.instances;

			kmeans.buildClusterer(localInstances);
			this.clusterCentroids = kmeans.getClusterCentroids();
			this.clusterStandardDeviations = kmeans.getClusterStandardDevs();
			evaluateCluster();
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getMessage());
		}
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansCluster#setK(int)
	 */
	public void setK(int k) {
		this.k = k;
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansCluster#getK()
	 */
	public int getK() {
		return this.k;
	}

	public int[] getClusterSizes() {
		return this.kmeans.getClusterSizes();
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#clusterInstance(int)
	 */
	public int clusterInstance(int i) {
		//assert kmeans != null : kmeans;
		int retval = 0;
		try {
			retval = kmeans.clusterInstance(instances.instance(i));
		} catch (Exception e) {
		}
		return retval;
	}
	
	protected double getSquaredError() {
		return kmeans.getSquaredError();
	}

	public int clusterInstance(Instance instance) throws Exception {
		return kmeans.clusterInstance(instance);
	}

}
