/*
 * Created on Apr 1, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package clustering.weka;

import clustering.*;
import common.RMICubeData;
import weka.core.Instances;
import weka.core.Instance;
import java.util.ArrayList;
import weka.attributeSelection.PrincipalComponents;

/**
 * @author khuck
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public class WekaPrincipalComponents implements PrincipalComponentsAnalysisInterface {

	// the number of components to keep
	private int k = 0;
	// the cluster descriptions
	private RawDataInterface inputData = null;
	private Instances instances = null;
	private Instances components = null;
	private PrincipalComponents pca = null;
	private int numAttributes = 0;
	private double[][] correlationCoefficients = null;
	private RMICubeData cubeData = null;
	private KMeansClusterInterface clusterer = null;
	private RawDataInterface[] clusters = null;
	private RawDataInterface transformed = null;
	private AnalysisFactory factory = null;
	
	public WekaPrincipalComponents (RMICubeData cubeData,
	AnalysisFactory factory) {
		this.cubeData = cubeData;
		this.factory = factory;
	}

	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#doPCA()
	 */
	public void doPCA() throws ClusterException {
		//assert instances != null : instances;

/*
		try {
			this.pca = new PrincipalComponents();
			if (k > 0)
				pca.setMaximumAttributeNames(k);
			pca.setNormalize(true);
			//pca.setTransformBackToOriginal(true);
			pca.buildEvaluator(instances);
			components = pca.transformedData();
			transformed = new WekaRawData(components);
		} catch (Exception e) {
		}
*/
		ArrayList names = new ArrayList();
		for (int i = 0 ; i < 2 ; i++) {
			names.add(cubeData.getNames()[i]);
		}

		transformed = factory.createRawData("Scatterplot Data",
			names, 2,inputData.numVectors());
		for (int i = 0 ; i < inputData.numVectors() ; i++) {
			float[] values = cubeData.getValues(i);
			for (int j = 0 ; j < 2 ; j++) {
				transformed.addValue(j,i,(double)(values[j]));
			}
		}
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
		this.instances = (Instances) inputData.getData();
	}

	public double[][] getCorrelationCoefficients() {
		if (this.correlationCoefficients == null) {
			this.numAttributes = this.instances.numAttributes(); 
			this.correlationCoefficients = new double[numAttributes][numAttributes];
			for (int i = 0 ; i < numAttributes ; i++) {
				double[] event1 = instances.attributeToDoubleArray(i);
				for (int j = 0 ; j < i ; j++) {
					double[] event2 = instances.attributeToDoubleArray(j);
					this.correlationCoefficients[i][j] = weka.core.Utils.correlation(event1, event2, numAttributes);
				}
			}
		}
		return correlationCoefficients;
	}

	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#getResults()
	 */
	public RawDataInterface getResults() {
		//assert components != null : components;
		return transformed;
	}

	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#reset()
	 */
	public void reset() {
		// TODO Auto-generated method stub

	}

	public RawDataInterface[] getClusters () {
		if (this.clusterer != null) {
			int[] clusterSizes = clusterer.getClusterSizes();
			int k = clusterer.getK();
			clusters = new RawDataInterface[k];
			Instances[] instances = new Instances[k];
			int[] counters = new int[k];
			
			for (int i = 0 ; i < k ; i++) {
				instances[i] = new
				Instances((Instances)(transformed.getData()), clusterSizes[i]);
				counters[i] = 0;
			}

			double values[] = new double[2];
			//int x = transformed.numDimensions() - 1;
			//int y = transformed.numDimensions() - 2;
			//int x = 0;
			//int y = 1;
			for (int i = 0 ; i < inputData.numVectors() ; i++) {
				int location = clusterer.clusterInstance(i);
				values[0] = transformed.getValue(0, i);
				values[1] = transformed.getValue(1, i);
				instances[location].add(new Instance(1.0, values));
				counters[location]++;
			}
			
			for (int i = 0 ; i < k ; i++) {
				clusters[i] = new WekaRawData(instances[i]);
			}
		}
		return clusters;
	}

	public void setClusterer (KMeansClusterInterface clusterer) {
		this.clusterer = clusterer;
	}

}
