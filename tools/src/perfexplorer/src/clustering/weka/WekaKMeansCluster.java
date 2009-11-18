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
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.attributeSelection.PrincipalComponents;

/**
 * This class is used as a list of names and values to describe 
 * a cluster created during some type of clustering operation.
 * 
 * <P>CVS $Id: WekaKMeansCluster.java,v 1.10 2009/11/18 10:17:35 khuck Exp $</P>
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
	private SimpleKMeans kmeans = null;
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
			this.kmeans = new SimpleKMeans();
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
//				kmeans.setInitialCenters(newTree.findCenters(k));
			}

			kmeans.buildClusterer(localInstances);
			this.clusterCentroids = kmeans.getClusterCentroids();
			this.clusterStandardDeviations = kmeans.getClusterStandardDevs();
			evaluateCluster();
/*			for (int x = 0 ; x < instances.numInstances() ; x++) {
				Instance inst = instances.instance(x);
				System.out.println(x + ": " + kmeans.clusterInstance(inst));
			}*/
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
			double betweenError = getBetweenError();
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

	public int[] clusterInstances() {
		int[] clusterIDs = new int[instances.numInstances()];
		for (int i = 0 ; i < instances.numInstances(); i++)
			clusterIDs[i] = clusterInstance(i);		
		return clusterIDs;
	}
	
	public RawDataInterface getClusterMaximums() {
	    if (this.clusterMaximums == null) {
			try {
				Instances instances = (Instances)this.inputData.getData();
				this.clusterMaximums = new Instances(instances, 0);
				Instances [] temp = new Instances[this.k];
				for (int i = 0; i < this.k; i++) {
					temp[i] = new Instances(instances, 0);
				}
				for (int i = 0; i < instances.numInstances(); i++) {
					temp[kmeans.clusterInstance(instances.instance(i))].add(new Instance(1.0, instances.instance(i).toDoubleArray()));
				}
				// iterate over the clusters
				for (int i = 0; i < this.k; i++) {
					double [] vals = new double[instances.numAttributes()];
					// iterate over the dimensions
					for (int j = 0; j < instances.numAttributes(); j++) {
						// note - calling this method changes the order of the data!
						vals[j] = temp[i].kthSmallestValue(j,temp[i].numInstances()-1);
					}
					// assign the new centroid value
					this.clusterMaximums.add(new Instance(1.0, vals));
				}
			} catch (Exception e) {
				System.err.println("getServer exception: " + e.getMessage());
				e.printStackTrace();
			}
	    }
		WekaRawData maximums = new WekaRawData(clusterMaximums);
		return maximums;
	}
	
	public RawDataInterface getClusterMinimums() {
		if (this.clusterMinimums == null) {
			try {
				Instances instances = (Instances)this.inputData.getData();
		        this.clusterMinimums = new Instances(instances, 0);
		        Instances [] temp = new Instances[this.k];
		        for (int i = 0; i < this.k; i++) {
		        	temp[i] = new Instances(instances, 0);
		        }
		        for (int i = 0; i < instances.numInstances(); i++) {
		        	temp[kmeans.clusterInstance(instances.instance(i))].add(new Instance(1.0, instances.instance(i).toDoubleArray()));
		        }
			    // iterate over the clusters
		        for (int i = 0; i < this.k; i++) {
		        	double [] vals = new double[instances.numAttributes()];
		        	// iterate over the dimensions
		        	for (int j = 0; j < instances.numAttributes(); j++) {
		        		// note - calling this method changes the order of the data!
		        		vals[j] = temp[i].kthSmallestValue(j,0);
		        	}
		        	// assign the new centroid value
		        	this.clusterMinimums.add(new Instance(1.0, vals));
		        }
			} catch (Exception e) {
				System.err.println("getServer exception: " + e.getMessage());
		        e.printStackTrace();
			}
		}
		WekaRawData minimums = new WekaRawData(clusterMinimums);
		return minimums;
	}
	
	public double getBetweenError() {
		// get the mean of the centroids
		Instance centroidMean = new
		Instance(this.clusterCentroids.numAttributes());
		for (int x = 0 ; x < this.clusterCentroids.numInstances() ; x++) {
			Instance tmpInst = this.clusterCentroids.instance(x);
			for (int y = 0 ; y < tmpInst.numAttributes() ; y++) {
				double tmp = centroidMean.value(y) + tmpInst.value(y);
				centroidMean.setValue(y,tmp);
			}
		}
		// get the squared error for the centroids
		double betweenError = 0.0;
		for (int x = 0 ; x < this.clusterCentroids.numInstances() ; x++) {
			betweenError += distance(centroidMean, this.clusterCentroids.instance(x));
    	}
		return betweenError;
	}

	/**
	 * Calculates the distance between two instances
	 *
	 * @param test the first instance
	 * @param train the second instance
	 * @return the distance between the two given instances, between 0 and 1
	 */          
	private double distance(Instance first, Instance second) {  

		double distance = 0;
	    int firstI, secondI;

	    for (int p1 = 0, p2 = 0; p1 < first.numValues() || p2 < second.numValues();) {
	    	if (p1 >= first.numValues()) {
	    		firstI = this.clusterCentroids.numAttributes();
	    	} else {
	    		firstI = first.index(p1); 
	    	}
	    	if (p2 >= second.numValues()) {
	    		secondI = this.clusterCentroids.numAttributes();
	    	} else {
	    		secondI = second.index(p2);
	    	}
	    	if (firstI == this.clusterCentroids.classIndex()) {
	    		p1++; continue;
	    	} 
	    	if (secondI == this.clusterCentroids.classIndex()) {
	    		p2++; continue;
	    	} 
	    	double diff;
	    	if (firstI == secondI) {
	    		diff = difference(firstI, first.valueSparse(p1), second.valueSparse(p2));
	    		p1++; p2++;
	    	} else if (firstI > secondI) {
	    		diff = difference(secondI, 0, second.valueSparse(p2));
	    		p2++;
	    	} else {
	    		diff = difference(firstI, first.valueSparse(p1), 0);
	    		p1++;
	    	}
	    	distance += diff * diff;
	    }
	    
	    //return Math.sqrt(distance / m_ClusterCentroids.numAttributes());
	    return distance;
	  }

	  /**
	   * Computes the difference between two given attribute
	   * values.
	   */
	  private double difference(int index, double val1, double val2) {

	    switch (this.clusterCentroids.attribute(index).type()) {
	    case Attribute.NOMINAL:
	      
	      // If attribute is nominal
	      if (Instance.isMissingValue(val1) || 
	          Instance.isMissingValue(val2) ||
	          ((int)val1 != (int)val2)) {
	        return 1;
	      } else {
	        return 0;
	      }
	    case Attribute.NUMERIC:

	      // If attribute is numeric
	      if (Instance.isMissingValue(val1) || 
	          Instance.isMissingValue(val2)) {
	        if (Instance.isMissingValue(val1) && 
	            Instance.isMissingValue(val2)) {
	          return 1;
	        } else {
	          double diff;
	          if (Instance.isMissingValue(val2)) {
	            diff = val1;
	          } else {
	            diff = val2;
	          }
	          if (diff < 0.5) {
	            diff = 1.0 - diff;
	          }
	          return diff;
	        }
	      } else {
	        return val1 - val2;
	      }
	    default:
	      return 0;
	    }
	  }

}
