/*
 * Created on Apr 1, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package clustering;

import weka.core.Instances;
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
	
	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#doPCA()
	 */
	public void doPCA() throws ClusterException {
		//assert instances != null : instances;

/*
		getCorrelationCoefficients();
		for (int i = 0 ; i < numAttributes ; i++) {
			for (int j = 0 ; j < i ; j++) {
				if (correlationCoefficients[i][j] > 0.8 || correlationCoefficients[i][j] < -0.8) {
					StringBuffer buf = new StringBuffer();
					buf.append(instances.attribute(i).name());
					buf.append(" / ");
					buf.append(instances.attribute(j).name());
					buf.append(" = ");
					buf.append(correlationCoefficients[i][j]);
					System.out.println(buf.toString());
				}
			}
		}
		*/
		
		try {
			this.pca = new PrincipalComponents();
			if (k > 0)
				pca.setMaximumAttributeNames(k);
			pca.setNormalize(true);
			//pca.setTransformBackToOriginal(true);
			pca.buildEvaluator(instances);
			components = pca.transformedData();
		} catch (Exception e) {
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
		WekaRawData transformed = new WekaRawData(components);
		return transformed;
	}

	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#reset()
	 */
	public void reset() {
		// TODO Auto-generated method stub

	}

}
