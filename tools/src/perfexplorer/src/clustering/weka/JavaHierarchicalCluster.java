/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering.weka;


import java.util.LinkedHashSet;
import java.util.List;
import java.util.ArrayList;

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
	private int k = 0; // number of clusters
	private Instances instances = null;
	private Instances clusterCentroids = null;
	private Instances clusterMaximums = null;
	private Instances clusterMinimums = null;
	private Instances clusterStandardDeviations = null;
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
			distances.solveManhattanDistances(inputData);
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
		int newline = 0;
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

		// iterate over the array, until we have as many subtrees as we have desired clusters
		for (int i = 1 ; i < this.k ; i++) {
			if (isInTree(i, this.clusters[i])) {
				clusterID = i;
				break;
			}
		}
		
		return clusterID;
	}

	private boolean isInTree(int i, DendrogramTree tree) {
		if (tree.getID() == i)
			return true;
		else if (tree.isLeaf())
			return false;
		else if (isInTree(i, tree.getLeft()) || isInTree(i, tree.getRight()))
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
		
		// cut the tree at the level which gives us the number of clusters we want (subtrees)
		this.distances = new DistanceMatrix(inputData.numVectors());
		this.distances.solveManhattanDistances(inputData);
		this.root = buildDendrogramTree();
		
		// first, the root of the tree is put at the first position of the array
		this.clusters = new DendrogramTree[this.k];
		this.clusters[0] = this.root;
		
		// iterate over the array, until we have as many subtrees as we have desired clusters
		for (int i = 1 ; i < this.k ; i++) {
			// find the subtree from all the current subtrees with the smallest height
			double min = this.clusters[0].getHeight();
			int minIndex = 0;
			for (int j = 0 ; j < i ; j++) {
				if (this.clusters[j].getHeight() < min) {
					min = this.clusters[j].getHeight();
					minIndex = 0;
				}
			}
			// now, split that subtree, put the left where the parent is,
			// and put the right at the end of the working array
			this.clusters[minIndex] = this.clusters[minIndex].getLeft();
			this.clusters[i] = this.clusters[minIndex].getRight();
		}
	}

	public RawDataInterface getClusterCentroids() {
		// TODO Auto-generated method stub
		return null;
	}

	public ClusterDescription getClusterDescription(int i)
			throws ClusterException {
		// TODO Auto-generated method stub
		return null;
	}

	public RawDataInterface getClusterMaximums() {
		// TODO Auto-generated method stub
		return null;
	}

	public RawDataInterface getClusterMinimums() {
		// TODO Auto-generated method stub
		return null;
	}

	public int[] getClusterSizes() {
		// TODO Auto-generated method stub
		return null;
	}

	public RawDataInterface getClusterStandardDeviations() {
		// TODO Auto-generated method stub
		return null;
	}

	public int getK() {
		return k;
	}

	public int getNumInstances() {
		return trees.length;
	}

	public void reset() {
		this.remainingIndices = null;
		this.trees = null;
		this.root = null;
		this.clusters = null;
		this.clusterCentroids = null;
		this.clusterMaximums = null;
		this.clusterMinimums = null;
		this.clusterStandardDeviations = null;
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
		int vectors = 10;
		int dimensions = 3;
		
		// generate some raw data
		List<String> attrs = new ArrayList<String>(vectors);
		attrs.add("x");
		attrs.add("y");
		attrs.add("z");
		RawDataInterface data = new WekaRawData("test", attrs, vectors, dimensions, null);
		for (int i = 0 ; i < vectors ; i++) {
			int modval = (i % 2) + 1;
			for (int j = 0 ; j < dimensions ; j++) {
				data.addValue(i, j, (Math.random() + modval));
			}
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
		hclust.setK(2);
		int[] clusters = hclust.clusterInstances();
		for (int i = 0 ; i < vectors ; i++) {
			System.out.println("Instance " + i + " is in cluster: " + clusters[i]);
		}
	}


}
