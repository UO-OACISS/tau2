/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering.r;


import java.util.Hashtable;
import org.omegahat.R.Java.REvaluator;

import edu.uoregon.tau.perfexplorer.clustering.ClusterDescription;
import edu.uoregon.tau.perfexplorer.clustering.ClusterException;
import edu.uoregon.tau.perfexplorer.clustering.DendrogramTree;
import edu.uoregon.tau.perfexplorer.clustering.KMeansClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;

/**
 * This class is the R implementation of the k-means clustering operation.
 * This class is package private - it should only be accessed from the
 * clustering class.  To access these methods, create an AnalysisFactory,
 * and the factory will be able to create a k-means cluster object.
 *
 * <P>CVS $Id: RKMeansCluster.java,v 1.8 2009/03/02 19:23:49 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 */
public class RKMeansCluster implements KMeansClusterInterface {

	// the number of clusters to find
	private int k = 0;
	// the cluster descriptions
	private RawDataInterface inputData = null;
	private RawDataInterface clusterCentroids = null;
	private RawDataInterface clusterMaximums = null;
	private RawDataInterface clusterMinimums = null;
	private RawDataInterface clusterStandardDeviations = null;
	private REvaluator rEvaluator = null;
	private DendrogramTree dendrogramTree = null;
	private int[] clusterSizes = null;
	private int[] clusters = null;
	private boolean doHierarchical = true;
	
	/**
	 * Default constructor
	 */
	public RKMeansCluster() {
		super();
		this.rEvaluator = RSingletons.getREvaluator();
		reset();
	}

	/**
	 * Reset method, for resetting the cluster.  If a user loads
	 * this object with data, and then does several clusterings
	 * with several K values, then we need a reset method.
	 */
	public void reset() {
		this.clusterCentroids = null;
		this.clusterStandardDeviations = null;
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#setInputData(RawDataInterface)
	 */
	public void setInputData(RawDataInterface inputData) {
		this.inputData = inputData;
		// put the data into R
		PerfExplorerOutput.print("Copying data...");
		rEvaluator.voidEval("raw <- matrix(0, nrow=" +
		inputData.numVectors() + ", ncol=" + inputData.numDimensions() + ")");
		StringBuilder buf = new StringBuilder();
		buf.append("alldata <- matrix(c(");
		for (int i = 0 ; i < inputData.numVectors() ; i++) {
			for (int j = 0 ; j < inputData.numDimensions() ; j++) {
				rEvaluator.voidEval("raw[" + (i+1) + "," + (j+1) +
					"] <- " + inputData.getValue(i,j));
				buf.append(inputData.getValue(i,j) + ",");
			}
		}
		buf.append("), nrow=" + inputData.numVectors() + 
			", ncol=" + inputData.numDimensions() + ", byrow=TRUE)");
		//PerfExplorerOutput.println(buf.toString());
		PerfExplorerOutput.println(" Done!");
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansCluster#findClusters()
	 */
	public void findClusters() throws ClusterException {
		boolean haveCenters = findInitialCenters();

		rEvaluator.voidEval("traw <- t(raw)");
		if (haveCenters) {
			PerfExplorerOutput.print("Making " + k + " centers...");
			int[]centerIndexes = dendrogramTree.findCenters(k);
			rEvaluator.voidEval("centers <- matrix(0, nrow=" + k + 
				", ncol=" + inputData.numDimensions() + ")");
			PerfExplorerOutput.print("centers: ");
			for (int j = 1 ; j <= k ; j++) {
				PerfExplorerOutput.print(centerIndexes[j-1]);
				rEvaluator.voidEval("centers[" + j + 
					",] <- raw[" + centerIndexes[j-1] + ",]");
				if (j != k)
					PerfExplorerOutput.print(",");
			}
			PerfExplorerOutput.println(" Done!");
		}

		PerfExplorerOutput.print("Doing k-means clustering...");
		if (haveCenters)
			rEvaluator.voidEval("cl <- kmeans(raw, centers, 20)");
		else
			rEvaluator.voidEval("cl <- kmeans(raw, " + k + ", 20)");
		PerfExplorerOutput.println(" Done!");

		PerfExplorerOutput.print("Getting averages, mins, maxes...");
		// get the averages for each cluster, and graph them
		rEvaluator.voidEval("maxes <- matrix(0.0, nrow=" + k + 
			", ncol=" + inputData.numDimensions() + ")");
		rEvaluator.voidEval("mins <- matrix(" + inputData.getMaximum() + 
			", nrow=" + k + ", ncol=" + inputData.numDimensions() + ")");
		rEvaluator.voidEval("totals <- matrix(0.0, nrow=" + k + 
			", ncol=" + inputData.numDimensions() + ")");
		rEvaluator.voidEval("counts <- matrix(0.0, nrow=" + k + 
			", ncol=" + inputData.numDimensions() + ")");
		rEvaluator.voidEval("ci <- 0");
		for (int m = 1 ; m <= inputData.numVectors() ; m++) {
			rEvaluator.voidEval("ci <- cl$cluster[" + m + "]");
			for (int n = 1 ; n <= inputData.numDimensions() ; n++) {
				rEvaluator.voidEval("if (raw[" + m + "," + n + "] > maxes[ci," + 
					n + "]) maxes[ci," + n + "] <- raw[" + m + "," + n + "]");
				rEvaluator.voidEval("if (raw[" + m + "," + n + "] < mins[ci," + 
					n + "]) mins[ci," + n + "] <- raw[" + m + "," + n + "]");
				rEvaluator.voidEval("totals[ci," + n + "] = totals[ci," + 
					n + "] + raw[" + m + "," + n + "]");
				rEvaluator.voidEval("counts[ci," + n + "] = counts[ci," + 
					n + "] + 1");
			}
		}
		rEvaluator.voidEval("avgs <- totals / counts");
		PerfExplorerOutput.println(" Done!");

		clusterSizes = (int[])rEvaluator.eval("cl$size");
		double[] centers = (double[])rEvaluator.eval("cl$centers");
		double[] avgs = (double[])rEvaluator.eval("avgs");
		clusterCentroids = new RRawData(k, inputData.numDimensions(), avgs);
		double[] mins = (double[])rEvaluator.eval("mins");
		clusterMinimums = new RRawData(k, inputData.numDimensions(), mins);
		double[] maxes = (double[])rEvaluator.eval("maxes");
		clusterMaximums = new RRawData(k, inputData.numDimensions(), maxes);
		clusters = (int[])rEvaluator.eval("cl$cluster");
		evaluateCluster();
		return;
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
		return clusterCentroids;
	}

	public RawDataInterface getClusterMinimums() {
		return clusterMinimums;
	}

	public RawDataInterface getClusterMaximums() {
		return clusterMaximums;
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#getClusterStandardDeviations()
	 */
	public RawDataInterface getClusterStandardDeviations() {
		return clusterStandardDeviations;
	}
	
	public int[] getClusterSizes() {
		return clusterSizes;
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#doPCA(boolean)
	 */
	public void doPCA(boolean doPCA) {
	}
	
	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#clusterInstance(int)
	 */
	public int clusterInstance(int i) {
		//assert kmeans != null : kmeans;
		// these values are 1 indexed, instead of 0 indexed...
		return clusters[i] - 1;
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansClusterInterface#getNumInstances()
	 */
	public int getNumInstances() {
		return inputData.numVectors();
	}

	// this method is by Calinski & Harabasz(1974)
	private void evaluateCluster() {
		return;
	}

	private boolean findInitialCenters() {
		boolean retval = false;
 		// arbitrary choice of 4098 to prevent crashes...
		if (this.doHierarchical && inputData.numVectors() < 4098) {
			PerfExplorerOutput.print("Getting distances...");
			rEvaluator.voidEval("threads <- dist(raw, method=\"manhattan\")");
			PerfExplorerOutput.println(" Done!");
			PerfExplorerOutput.print("Hierarchical clustering...");
			rEvaluator.voidEval("hcgtr <- hclust(threads, method=\"average\")");
			int[] merge = (int[])rEvaluator.eval("t(hcgtr$merge)");
			double[] height = (double[])rEvaluator.eval("hcgtr$height");
			dendrogramTree = createDendrogramTree(merge, height);
			rEvaluator.voidEval("dend <- as.dendrogram(hcgtr)");
			PerfExplorerOutput.println(" Done!");
			retval = true;
		}
		return (retval);
	}

	public DendrogramTree createDendrogramTree (int[] merge, double[] height) {
		DendrogramTree leftLeaf = null;
		DendrogramTree rightLeaf = null;
		DendrogramTree newTree = null;
		Hashtable finder = new Hashtable(merge.length);
		int j = 0;
		for (int i = 0 ; i < height.length ; i++) {
			if (merge[j] < 0)
				leftLeaf = new DendrogramTree(merge[j], 0.0);
			else
				leftLeaf = (DendrogramTree)finder.get(new Integer(merge[j]));
			j++;
			if (merge[j] < 0)
				rightLeaf = new DendrogramTree(merge[j], 0.0);
			else
				rightLeaf = (DendrogramTree)finder.get(new Integer(merge[j]));
			j++;
			newTree = new DendrogramTree(i+1, height[i]);
			newTree.setLeftAndRight(leftLeaf, rightLeaf);
			finder.put(new Integer(i+1), newTree);
		}
		return newTree;
	}

	public void doSmartInitialization(boolean b) {
		// TODO Auto-generated method stub
		
	}

}
