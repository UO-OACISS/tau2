/*
 * Created on Mar 16, 2005
 *
 */
package clustering.r;

import clustering.*;
import java.util.Hashtable;
import org.omegahat.R.Java.REvaluator;
import org.omegahat.R.Java.ROmegahatInterpreter;

/**
 * This class is used as a list of names and values to describe 
 * a cluster created during some type of clustering operation.
 * 
 * <P>CVS $Id: RKMeansCluster.java,v 1.1 2005/09/27 19:49:29 khuck Exp $</P>
 * @author khuck
 *
 */
public class RKMeansCluster implements KMeansClusterInterface {

	// dimension reduction possibilities
	private boolean doPCA = false;
	// the number of clusters to find
	private int k = 0;
	// the cluster descriptions
	private RawDataInterface inputData = null;
	private RawDataInterface clusterCentroids = null;
	private RawDataInterface clusterMaximums = null;
	private RawDataInterface clusterMinimums = null;
	private RawDataInterface clusterStandardDeviations = null;
	private ROmegahatInterpreter rInterpreter = null;
	private REvaluator rEvaluator = null;
	private DendrogramTree dendrogramTree = null;
	private int[] clusterSizes = null;
	private int[] clusters = null;
	
	/**
	 * Default constructor
	 */
	public RKMeansCluster() {
		super();
		this.rInterpreter = RSingletons.getRInterpreter();
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
		System.out.print("Copying data...");
		rEvaluator.voidEval("raw <- matrix(0, nrow=" +
		inputData.numVectors() + ", ncol=" + inputData.numDimensions() + ")");
		StringBuffer buf = new StringBuffer();
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
		//System.out.println(buf.toString());
		System.out.println(" Done!");
	}

	/* (non-Javadoc)
	 * @see clustering.KMeansCluster#findClusters()
	 */
	public void findClusters() throws ClusterException {
		boolean haveCenters = findInitialCenters();

		rEvaluator.voidEval("traw <- t(raw)");
		if (haveCenters) {
			System.out.print("Making " + k + " centers...");
			int[]centerIndexes = dendrogramTree.findCenters(k);
			rEvaluator.voidEval("centers <- matrix(0, nrow=" + k + 
				", ncol=" + inputData.numDimensions() + ")");
			System.out.print("centers: ");
			for (int j = 1 ; j <= k ; j++) {
				System.out.print(centerIndexes[j-1]);
				rEvaluator.voidEval("centers[" + j + 
					",] <- raw[" + centerIndexes[j-1] + ",]");
				if (j != k)
					System.out.print(",");
			}
			System.out.println(" Done!");
		}

		System.out.print("Doing k-means clustering...");
		if (haveCenters)
			rEvaluator.voidEval("cl <- kmeans(raw, centers, 20)");
		else
			rEvaluator.voidEval("cl <- kmeans(raw, " + k + ", 20)");
		System.out.println(" Done!");

		System.out.print("Getting averages, mins, maxes...");
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
		System.out.println(" Done!");

		clusterSizes = (int[])rEvaluator.eval("cl$size");
		double[] centers = (double[])rEvaluator.eval("cl$centers");
		double[] avgs = (double[])rEvaluator.eval("centers");
		clusterCentroids = new RRawData(k, inputData.numDimensions(), avgs);
		double[] mins = (double[])rEvaluator.eval("mins");
		clusterMinimums = new RRawData(k, inputData.numDimensions(), mins);
		double[] maxes = (double[])rEvaluator.eval("maxes");
		clusterMaximums = new RRawData(k, inputData.numDimensions(), maxes);
		clusters = (int[])rEvaluator.eval("cl$cluster");

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
		this.doPCA = doPCA;
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
	}

	private boolean findInitialCenters() {
		boolean retval = false;
 		// arbitrary choice of 4098 to prevent crashes...
		if (inputData.numVectors() < 4098) {
			System.out.print("Getting distances...");
			rEvaluator.voidEval("threads <- dist(raw, method=\"manhattan\")");
			System.out.println(" Done!");
			System.out.print("Hierarchical clustering...");
			rEvaluator.voidEval("hcgtr <- hclust(threads, method=\"average\")");
			int[] merge = (int[])rEvaluator.eval("t(hcgtr$merge)");
			double[] height = (double[])rEvaluator.eval("hcgtr$height");
			dendrogramTree = createDendrogramTree(merge, height);
			rEvaluator.voidEval("dend <- as.dendrogram(hcgtr)");
			System.out.println(" Done!");
/*
			System.out.print("Making png image...");
			String description = "dendrogram." + modelData.toString();
			description = description.replaceAll(":",".");
			String shortDescription = modelData.toShortString();
			String fileName = "/tmp/dendrogram." + shortDescription + ".png";
			String thumbnail = "/tmp/dendrogram.thumb." + shortDescription + ".jpg";
			rEvaluator.voidEval("png(\"" + fileName + "\",width=800, height=400)");
			rEvaluator.voidEval("plot (dend, main=\"" + description + "\", edge.root=FALSE,horiz=FALSE,axes=TRUE)");
			rEvaluator.voidEval("dev.off()");
			System.out.println(" Done!");
			saveAnalysisResult(description, fileName, thumbnail, true);
*/
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

}
