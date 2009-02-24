/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering.weka;

import edu.uoregon.tau.perfexplorer.clustering.ClusterDescription;
import edu.uoregon.tau.perfexplorer.clustering.ClusterException;
import edu.uoregon.tau.perfexplorer.clustering.DendrogramTree;
import edu.uoregon.tau.perfexplorer.clustering.DistanceMatrix;
import edu.uoregon.tau.perfexplorer.clustering.KMeansClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;
import weka.core.Instances;
import weka.attributeSelection.PrincipalComponents;

/**
 * This class is used as a list of names and values to describe 
 * a cluster created during some type of clustering operation.
 * 
 * <P>CVS $Id: WekaKMeansCluster.java,v 1.8 2009/02/24 00:53:36 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since 0.1
 *
 */
public class WekaKMeansCluster implements KMeansClusterInterface {

	// dimension reduction possibilities
	private boolean doPCA = false;
	// the number of clusters to find
	private int k = 0;
	// the cluster descriptions
	private Instances instances = null;
	private Instances clusterCentroids = null;
	private Instances clusterMaximums = null;
	private Instances clusterMinimums = null;
	private Instances clusterStandardDeviations = null;
	private ImprovedSimpleKMeans kmeans = null;
	private RawDataInterface inputData = null;
	private boolean hierarchicalInitialize = true;
	
	/**
	 * Default constructor - package protected
	 */
	WekaKMeansCluster() {
		super();
		reset();
	}

	/**
	 * Reset method, for resetting the cluster.  If a user loads
	 * this object with data, and then does several clusterings
	 * with several K values, then we need a reset method.
	 */
	public void reset() {
		this.clusterCentroids = null;
		this.clusterMaximums = null;
		this.clusterMinimums = null;
		this.clusterStandardDeviations = null;
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#setInputData(RawDataInterface)
	 */
	public void setInputData(RawDataInterface inputData) {
		this.instances = (Instances) inputData.getData();
		this.inputData = inputData;
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansCluster#findClusters()
	 */
	public void findClusters() throws ClusterException {
		//assert instances != null : instances;
		try {
			this.kmeans = new ImprovedSimpleKMeans();
			kmeans.setNumClusters(k);
			Instances localInstances = null;
			if (this.doPCA) {
				PrincipalComponents pca = new PrincipalComponents();
				pca.setMaximumAttributeNames(1);
				pca.setNormalize(true);
				pca.setTransformBackToOriginal(true);
				pca.buildEvaluator(instances);
				localInstances = pca.transformedData();
			} else localInstances = this.instances;
			// get the initial centers
			if (hierarchicalInitialize) {
				DistanceMatrix distances = new DistanceMatrix(localInstances.numInstances());
				distances.solveManhattanDistances(inputData);
				JavaHierarchicalCluster hclust = new JavaHierarchicalCluster(distances);
				DendrogramTree newTree = hclust.buildDendrogramTree();
				kmeans.setInitialCenters(newTree.findCenters(k));
			}

			kmeans.buildClusterer(localInstances);
			this.clusterCentroids = kmeans.getClusterCentroids();
			this.clusterMaximums = kmeans.getClusterMaximums();
			this.clusterMinimums = kmeans.getClusterMinimums();
			this.clusterStandardDeviations = kmeans.getClusterStandardDevs();
			evaluateCluster();
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getMessage());
		}
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansCluster#getClusterDescription(int)
	 */
	public ClusterDescription getClusterDescription(int i)
			throws ClusterException {
		return null;
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

	/* (non-Javadoc)
	 * @see clustering.KMeansCluster#setInitialCenters(int[])
	 */
	public void setInitialCenters(int[] indexes) {
		// TODO Auto-generated method stub
		
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#getClusterCentroids()
	 */
	public RawDataInterface getClusterCentroids() {
		WekaRawData centroids = new WekaRawData(clusterCentroids);
		return centroids;
	}

	public RawDataInterface getClusterMaximums() {
		WekaRawData maximums = new WekaRawData(clusterMaximums);
		return maximums;
	}

	public RawDataInterface getClusterMinimums() {
		WekaRawData minimums = new WekaRawData(clusterMinimums);
		return minimums;
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#getClusterStandardDeviations()
	 */
	public RawDataInterface getClusterStandardDeviations() {
		WekaRawData deviations = new WekaRawData(clusterStandardDeviations);
		return deviations;
	}
	
	public int[] getClusterSizes() {
		return this.kmeans.getClusterSizes();
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#doPCA(boolean)
	 */
	public void doPCA(boolean doPCA) {
		this.doPCA = doPCA;
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

	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#getNumInstances()
	 */
	public int getNumInstances() {
		return instances.numInstances();
	}

	// this method is by Calinski & Harabasz(1974)
	private void evaluateCluster() {
		try {
			double betweenError = kmeans.getBetweenError();
			//PerfExplorerOutput.println("Between Squared Error: " + betweenError);
			double withinError = kmeans.getSquaredError();
			//PerfExplorerOutput.println("Within Squared Error: " + withinError);
			//PerfExplorerOutput.println("k-1: " + (k-1));
			//PerfExplorerOutput.println("n-k: " + (instances.numInstances()-k));
			double maximizeMe = (betweenError * (k-1)) / 
				(withinError * (instances.numInstances() - k));
			//PerfExplorerOutput.println("Maximize Me: " + maximizeMe);
		} catch (Exception e) {
			PerfExplorerOutput.println ("EXCEPTION: " + e.getMessage());
			e.printStackTrace();
		}
	}

	public void doSmartInitialization(boolean b) {
		this.hierarchicalInitialize = b;
	}
}
