/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering.weka;


import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;
import edu.uoregon.tau.perfexplorer.clustering.ClusterDescription;
import edu.uoregon.tau.perfexplorer.clustering.ClusterException;
import edu.uoregon.tau.perfexplorer.clustering.DendrogramTree;
import edu.uoregon.tau.perfexplorer.clustering.DistanceMatrix;
import edu.uoregon.tau.perfexplorer.clustering.HierarchicalCluster;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;

/**
 * This class solves a hierarchical clustering for a RawData object.
 * 
 * @author khuck
 *
 */
public class JavaHierarchicalCluster implements HierarchicalCluster {
	
	private DistanceMatrix distances = null;
	private int dimension = 0;
	private LinkedHashSet<Integer> remainingIndices = null;
	private DendrogramTree[] trees = null;
	private DendrogramTree root = null;
	private DendrogramTree[] clusters = null;
	private List<Integer>[] clusterIndexes = null;
	private int k = 0; // number of clusters
	private Instances instances = null;
	private Instances clusterCentroids = null;
	private Instances clusterMaximums = null;
	private Instances clusterMinimums = null;
	private Instances clusterStandardDeviations = null;
	private int[] clusterSizes = null;
	private RawDataInterface inputData = null;

	/**
	 * Package-private constructor.
	 * 
	 * @param distances
	 */
	JavaHierarchicalCluster(DistanceMatrix distances) {
		this.distances = new DistanceMatrix(distances);
	}
	
	public JavaHierarchicalCluster() {
		
	}

	/**
	 * Build the dendrogram tree.
	 * 
	 * Using the distance matric passed in, perform a hierarchical cluster
	 * on the data.
	 */
	public DendrogramTree buildDendrogramTree() {
		if (distances == null) {
			distances = new DistanceMatrix(inputData.numVectors());
			//distances.solveManhattanDistances(inputData);
			distances.solveCartesianDistances(inputData);
		}
		
		this.dimension = distances.getDimension();
		
		// create a set of the remaining indices
		remainingIndices = new LinkedHashSet<Integer>(dimension);
		// create an initial array of leaf trees
		trees = new DendrogramTree[dimension];
		
		// initialize the HashSet and the tree array
		for (int i = 0 ; i < dimension ; i++) {
			remainingIndices.add(new Integer(i));
			trees[i] = new DendrogramTree(((-1)-i), 0);
		}
		
		DendrogramTree newTree = null;
		//int newline = 0;
		while (remainingIndices.size() > 1) {
			boolean first = true;
			double min = 0.0;
			int[] location = {0,0}; 
			for (int i = 0 ; i < dimension ; i++) {
				// skip the ones we have merged already
				if (!remainingIndices.contains(new Integer(i)))
					continue;
				
				for (int j = 0 ; j < i ; j++) {
					// skip the ones we have merged already
					if (!remainingIndices.contains(new Integer(j)))
						continue;

					// if this is the first pass, save the first distance
					if (first) {
						first = false;
						min = distances.elementAt(i,j);
						location[0] = i;
						location[1] = j;
					} else {
						// find the two closest vectors
						if (min > distances.elementAt(i,j)) {
							min = distances.elementAt(i,j);
							location[0] = i;
							location[1] = j;
						} //if
					} //else
				} //for
			} //for
			
			//ok, we found the two closest.  Now what?
			// remove the second index from the hash set
			remainingIndices.remove(new Integer(location[0]));
			//PerfExplorerOutput.print(" " + location[0]);
			//if (++newline % 20 == 0) PerfExplorerOutput.println(" : " + newline / 20);
			
			// create a new tree node, with the left and right leaves
			newTree = new DendrogramTree(location[1], min);
			newTree.setLeftAndRight(trees[location[1]], trees[location[0]]);
			trees[location[1]] = newTree;
			
			// merge the two vectors into one in the distance matrix
			distances.mergeDistances(location[1], location[0]);
			//PerfExplorerOutput.println(distances.toString());
			// lather, rinse, repeat
		}
		this.root = newTree;
		return newTree;
	}
	
	public int clusterInstance(int index) {
		int clusterID = 0;
		if (this.clusters == null) {
			try {
				findClusters();
			} catch (ClusterException e) {
				System.err.println("Error clustering");
				e.printStackTrace();
			}
		}
		// remember, the instances are 1 indexed, not 0 indexed, so add one here
		index++;

		// iterate over the array, until we have as many subtrees as we have desired clusters
		for (int i = 0 ; i < this.k ; i++) {
			if (isInTree(index, this.clusters[i])) {
				clusterID = i;
				break;
			}
		}
		
		return clusterID;
	}

	private boolean isInTree(int i, DendrogramTree tree) {
		if (tree.isLeaf()) {
			if (Math.abs(tree.getID()) == i) {
				return true;
			}
			return false;
		} else if (isInTree(i, tree.getLeft()) || isInTree(i, tree.getRight()))
			return true;
		return false;
	}

	public int[] clusterInstances() {
		if (this.clusters == null)
			try {
				findClusters();
			} catch (ClusterException e) {
				System.err.print("Error clustering data");
				e.printStackTrace();
			}
		
		// there is a WAY faster way to this...
		int[] clusterIDs = new int[this.dimension];
		for (int i = 0 ; i < this.dimension ; i++) { 
			clusterIDs[i] = clusterInstance(i);
		}
		return clusterIDs;
	}

	public void findClusters() throws ClusterException {
		// check to make sure the tree is there
		if (this.root == null)
			this.buildDendrogramTree();
		
//		// cut the tree at the level which gives us the number of clusters we want (subtrees)
//		this.distances = new DistanceMatrix(inputData.numVectors());
//		this.distances.solveManhattanDistances(inputData);
//		this.root = buildDendrogramTree();
		
		// first, the root of the tree is put at the first position of the array
		this.clusters = new DendrogramTree[this.k];
		this.clusters[0] = this.root;
		
		// iterate over the array, until we have as many subtrees as we have desired clusters
		for (int i = 1 ; i < this.k ; i++) {
			// find the subtree from all the current subtrees with the largest height
			double max = this.clusters[0].getHeight();
			int maxIndex = 0;
			for (int j = 0 ; j < i ; j++) {
				if (this.clusters[j].getHeight() > max) {
					max = this.clusters[j].getHeight();
					maxIndex = j;
				}
			}
			// now, split that subtree, put the left where the parent is,
			// and put the right at the end of the working array
			this.clusters[i] = this.clusters[maxIndex].getRight();
			this.clusters[maxIndex] = this.clusters[maxIndex].getLeft();
		}
	}

	public RawDataInterface getClusterCentroids() {
		if (clusterCentroids == null)
			generateStats();

		WekaRawData centroids = new WekaRawData(clusterCentroids);
		return centroids;
	}

	private void generateStats() {
		// first, make sure we have the instances for each subtree
		if (this.clusterSizes == null) {
			getClusterSizes();
		}
		if (this.instances == null) {
			this.instances = (Instances)this.inputData.getData();
		}
		clusterCentroids = new Instances(instances, clusters.length);
		clusterMaximums = new Instances(instances, clusters.length);
		clusterMinimums = new Instances(instances, clusters.length);
		clusterStandardDeviations = new Instances(instances, clusters.length);

		for (int i = 0 ; i < this.clusters.length ; i++) {
			int numAttr = this.instances.firstInstance().numAttributes();
			double[] wekasucks = new double[numAttr];
			Instance total = new Instance(1.0, wekasucks);
			Instance min = new Instance(1.0, wekasucks);
			Instance max = new Instance(1.0, wekasucks);
			Instance avg = new Instance(1.0, wekasucks);
			Instance stddev = new Instance(1.0, wekasucks);
			// iterate over the instances in the cluster
			boolean first = true;
			for (Integer index : this.clusterIndexes[i]) {
				Instance tmp = this.instances.instance(index.intValue());
				// iterate over the attributes for each dimension
				for (int attr = 0 ; attr < numAttr ; attr++) {
					double tmpVal = tmp.value(attr);
					total.setValue(attr, total.value(attr) + tmpVal);
					max.setValue(attr, Math.max(max.value(attr), tmpVal));
					if (first) {
						min.setValue(attr, tmpVal);
					} else {
						min.setValue(attr, Math.min(min.value(attr), tmpVal));
					}
				}
				first = false;
			}
			for (int attr = 0 ; attr < numAttr ; attr++) {
				avg.setValue(attr, total.value(attr) / this.clusterSizes[i]);
			}
			clusterCentroids.add(avg);
			clusterMaximums.add(max);
			clusterMinimums.add(min);
			// iterate over the instances in the cluster for stddev
			for (Integer index : this.clusterIndexes[i]) {
				Instance tmp = this.instances.instance(index.intValue());
				// iterate over the attributes for each dimension
				for (int attr = 0 ; attr < numAttr ; attr++) {
					double tmpVal = tmp.value(attr);
					stddev.setValue(attr, (stddev.value(attr) + (Math.pow(avg.value(attr) - tmpVal, 2.0))));
				}
			}
			for (int attr = 0 ; attr < numAttr ; attr++) {
				stddev.setValue(attr, Math.sqrt(stddev.value(attr)));
			}
			clusterStandardDeviations.add(stddev);
		}
	}

	public ClusterDescription getClusterDescription(int i)
			throws ClusterException {
		// TODO Auto-generated method stub
		return null;
	}

	public RawDataInterface getClusterMaximums() {
		if (clusterMaximums == null)
			generateStats();

		WekaRawData maximums = new WekaRawData(clusterMaximums);
		return maximums;
	}

	public RawDataInterface getClusterMinimums() {
		if (clusterMinimums == null)
			generateStats();

		WekaRawData minimums = new WekaRawData(clusterMinimums);
		return minimums;
	}

	@SuppressWarnings("unchecked")
	public int[] getClusterSizes() {
		if (this.clusterSizes == null) {
			clusterSizes = new int[this.clusters.length];
			if (clusterIndexes == null) {
				clusterIndexes = new List[this.clusters.length];
			}
			for (int i = 0 ; i < this.clusters.length ; i++) {
				if (clusterIndexes[i] == null) {
					clusterIndexes[i] = this.clusters[i].getIndexes();
				}
				clusterSizes[i] = clusterIndexes[i].size();
			}
		}
		return clusterSizes;
	}

	public RawDataInterface getClusterStandardDeviations() {
		if (clusterStandardDeviations == null)
			generateStats();

		WekaRawData deviations = new WekaRawData(clusterStandardDeviations);
		return deviations;
	}

	public int getK() {
		return k;
	}

	public int getNumInstances() {
		return trees.length;
	}

	public void reset() {
//		this.remainingIndices = null;
//		this.trees = null;
//		this.root = null;
		this.clusters = null;
		this.clusterCentroids = null;
		this.clusterMaximums = null;
		this.clusterMinimums = null;
		this.clusterStandardDeviations = null;
		this.clusterSizes = null;
		this.clusterIndexes = null;
	}

	public void setInputData(RawDataInterface inputData) {
		this.inputData = inputData;
	}

	public void setK(int k) {
		this.k = k;
	}
	
	/**
	 * Testing method...
	 * @param args
	 */
	public static void main(String[] args) {
		int dimensions = 2;
		int k = 4;
		int vectors = 3*k;
		
		// generate some raw data
		List<String> attrs = new ArrayList<String>(vectors);
		attrs.add("x");
		attrs.add("y");
//		attrs.add("z");
		RawDataInterface data = new WekaRawData("test", attrs, vectors, dimensions, null);
		for (int i = 0 ; i < vectors ; i++) {
			int modval = (i % k);
			System.out.print("modval: " + modval);
			for (int j = 0 ; j < dimensions ; j++) {
				double val = (0.5 + (Math.random()/10.0) + modval);
				System.out.print(" val[" + j + "]: " + val);
				data.addValue(i, j, val);
			}
			System.out.println("");
		}
		
		// get the distances
		DistanceMatrix distances = new DistanceMatrix(vectors);
		distances.solveManhattanDistances(data);
		PerfExplorerOutput.println("Got Distances...");
		
		// do the hierarchical clustering
		JavaHierarchicalCluster hclust = new JavaHierarchicalCluster(distances);
		DendrogramTree newTree = hclust.buildDendrogramTree();
		PerfExplorerOutput.println("\n\n" + newTree.toString());
		
		// do it again, the other way
		hclust = new JavaHierarchicalCluster();
		hclust.setInputData(data);
		hclust.setK(k);
		int[] clusters = hclust.clusterInstances();
		for (int i = 0 ; i < vectors ; i++) {
			System.out.println("Instance " + i + " is in cluster: " + clusters[i]);
		}
	}


}
