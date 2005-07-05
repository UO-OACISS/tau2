/*
 * Created on Mar 16, 2005
 *
 */
package clustering;

import java.util.List;
import weka.core.Instances;
import weka.attributeSelection.PrincipalComponents;

/**
 * This class is used as a list of names and values to describe 
 * a cluster created during some type of clustering operation.
 * 
 * <P>CVS $Id: WekaKMeansCluster.java,v 1.1 2005/07/05 22:29:51 amorris Exp $</P>
 * @author khuck
 *
 */
public class WekaKMeansCluster implements KMeansClusterInterface {

	// dimension reduction possibilities
	private boolean doPCA = false;
	// the number of clusters to find
	private int k = 0;
	// the cluster descriptions
	private List clusterDescriptions = null;
	private RawDataInterface inputData = null;
	private Instances instances = null;
	private Instances clusterCentroids = null;
	private Instances clusterStandardDeviations = null;
	private ImprovedSimpleKMeans kmeans = null;
	
	/**
	 * Default constructor
	 */
	public WekaKMeansCluster() {
		super();
		reset();
	}

	/**
	 * Reset method, for resetting the cluster.  If a user loads
	 * this object with data, and then does several clusterings
	 * with several K values, then we need a reset method.
	 */
	public void reset() {
		this.clusterDescriptions = null;
		this.clusterCentroids = null;
		this.clusterStandardDeviations = null;
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#setInputData(RawDataInterface)
	 */
	public void setInputData(RawDataInterface inputData) {
		this.inputData = inputData;
		this.instances = (Instances) inputData.getData();		
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansCluster#findClusters()
	 */
	public void findClusters() throws ClusterException {
		assert instances != null : instances;
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
			/*
			DistanceMatrix distances = new DistanceMatrix(inputData.numVectors());
			distances.solveManhattanDistances(inputData);
			JavaHierarchicalCluster hclust = new JavaHierarchicalCluster(distances);
			DendrogramTree newTree = hclust.buildDendrogramTree();
			kmeans.setInitialCenters(newTree.findCenters(k));
			*/

			kmeans.buildClusterer(localInstances);
			this.clusterCentroids = kmeans.getClusterCentroids();
			this.clusterStandardDeviations = kmeans.getClusterStandardDevs();
		} catch (Exception e) {
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
		assert kmeans != null : kmeans;
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

}
