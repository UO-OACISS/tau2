/*
 * Created on Apr 1, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */

package clustering.r;

import clustering.*;
import common.RMICubeData;
import org.omegahat.R.Java.REvaluator;
import org.omegahat.R.Java.ROmegahatInterpreter;

/**
 * @author khuck
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public class RPrincipalComponents implements PrincipalComponentsAnalysisInterface {

	// the number of components to keep
	private int k = 0;
	// the cluster descriptions
	private RawDataInterface inputData = null;
	private ROmegahatInterpreter rInterpreter = null;
	private REvaluator rEvaluator = null;
	private double[] sdevs = null;
	private int[] maxIndex = null;
	private RMICubeData cubeData = null;
	private RawDataInterface transformed = null;
	private KMeansClusterInterface clusterer = null;
	private RawDataInterface[] clusters = null;
	
	/**
	 * Default constructor
	 */
	public RPrincipalComponents (RMICubeData cubeData) {
		super();
		this.cubeData = cubeData;
		this.rInterpreter = RSingletons.getRInterpreter();
		this.rEvaluator = RSingletons.getREvaluator();
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
	 * @see clustering.PrincipalComponentsAnalysisInterface#setK(int)
	 */
	public void setK(int k) {
		this.k = k;
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

}
