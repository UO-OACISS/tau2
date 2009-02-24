/*
 * Created on Apr 1, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */

package edu.uoregon.tau.perfexplorer.clustering.r;

import edu.uoregon.tau.perfexplorer.clustering.ClusterDescription;
import edu.uoregon.tau.perfexplorer.clustering.ClusterException;
import edu.uoregon.tau.perfexplorer.clustering.KMeansClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.PrincipalComponentsAnalysisInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.common.RMICubeData;

/**
 * This class is the R implementation of the k-means clustering operation.
 * This class is package private - it should only be accessed from the
 * clustering class.  To access these methods, create an AnalysisFactory,
 * and the factory will be able to create a k-means cluster object.
 *
 * <P>CVS $Id: RPrincipalComponents.java,v 1.6 2009/02/24 00:53:35 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 */
public class RPrincipalComponents implements PrincipalComponentsAnalysisInterface {

	// the cluster descriptions
	private RawDataInterface inputData = null;
	private int[] maxIndex = null;
	private RMICubeData cubeData = null;
	private RawDataInterface transformed = null;
	private KMeansClusterInterface clusterer = null;
	private RawDataInterface[] clusters = null;
	
    /**
     * Default constructor is private to prevent instantiation this way
     */
    private RPrincipalComponents() {}

    /**
	 * Default constructor
	 */
	public RPrincipalComponents (RMICubeData cubeData) {
		super();
		this.cubeData = cubeData;
		maxIndex = new int[2];
		maxIndex[0] = maxIndex[1] = 0;
	}

	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#doPCA()
	 */
	public void doPCA() throws ClusterException {
		// call the princomp() method in R
		// get the pc$sdev values
		/*
		rEvaluator.voidEval("pc <- prcomp(raw, scale=TRUE)");
		sdevs = (double[])rEvaluator.eval("pc$sdev");
		if (sdevs != null) {
			for (int i = 1 ; i < inputData.numDimensions() ; i++) {
				if (sdevs[maxIndex[0]] < sdevs[i]) {
					maxIndex[1] = maxIndex[0];
					maxIndex[0] = i;
				} else if (sdevs[maxIndex[1]] < sdevs[i]) {
					maxIndex[1] = i;
				} 
			}
		}
		*/
		// there are 4 values for each thread of execution in the Cube
		// data.  We want 2 rows with all threads in them.
		transformed = new RRawData(2,inputData.numVectors(),cubeData.getNames());
		for (int i = 0 ; i < inputData.numVectors() ; i++) {
			float[] values = cubeData.getValues(i);
			for (int j = 0 ; j < 2 ; j++) {
				transformed.addValue(j,i,(double)(values[j]));
			}
		}
		return;
	}

	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#getComponentDescription(int)
	 */
	public ClusterDescription getComponentDescription(int i)
			throws ClusterException {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#setInputData(clustering.RawDataInterface)
	 */
	public void setInputData(RawDataInterface inputData) {
		this.inputData = inputData;
	}

	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#getResults()
	 */
	public RawDataInterface getResults() {
		//assert components != null : components;
		return transformed;
	}

	public void setClusterer (KMeansClusterInterface clusterer) {
		this.clusterer = clusterer;
	}

	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#getResults()
	 */
	public RawDataInterface[] getClusters() {
		if (this.clusterer != null) {
			int[] clusterSizes = clusterer.getClusterSizes();
			clusters = new RawDataInterface[clusterer.getK()];
			int[] counters = new int[clusterer.getK()];
			for (int i = 0 ; i < clusterer.getK() ; i++) {
				clusters[i] = new RRawData(clusterSizes[i], 2);
				counters[i] = 0;
			}

			for (int i = 0 ; i < inputData.numVectors() ; i++) {
				int location = clusterer.clusterInstance(i);
				double value = transformed.getValue(0,i);
				clusters[location].addValue(counters[location], 0, value);
				value = transformed.getValue(1,i);
				clusters[location].addValue(counters[location], 1, value);
				counters[location]++;
			}
			
		}
		return clusters;
	}

	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#reset()
	 */
	public void reset() {
		// TODO Auto-generated method stub

	}

	public void setMaxComponents(int maxComponents) {
		// TODO Auto-generated method stub
		System.err.println("NOT IMPLEMENTED");
	}

}
