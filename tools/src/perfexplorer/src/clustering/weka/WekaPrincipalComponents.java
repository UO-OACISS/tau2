/*
 * Created on Apr 1, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package edu.uoregon.tau.perfexplorer.clustering.weka;

import java.util.ArrayList;

import weka.attributeSelection.PrincipalComponents;
import weka.core.Instance;
import weka.core.Instances;
import edu.uoregon.tau.perfexplorer.clustering.ClusterDescription;
import edu.uoregon.tau.perfexplorer.clustering.ClusterException;
import edu.uoregon.tau.perfexplorer.clustering.ClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.DBScanClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.PrincipalComponentsAnalysisInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.common.RMICubeData;

/**
 * This class will perform PCA on weka data.
 * TODO - make this class immutable?
 * 
 * @author khuck
 * <P>CVS $Id: WekaPrincipalComponents.java,v 1.15 2009/11/25 09:15:34 khuck Exp $</P>
 * @version 0.1
 * @since   0.1
 */
public class WekaPrincipalComponents implements PrincipalComponentsAnalysisInterface {

	// the cluster descriptions
	private RawDataInterface inputData = null;
	private Instances instances = null;
	private Instances components = null;
	private PrincipalComponents pca = null;
	private int numAttributes = 0;
	private double[][] correlationCoefficients = null;
	private RMICubeData cubeData = null;
	private RawDataInterface rawData = null;
	private ClusterInterface clusterer = null;
	private RawDataInterface[] clusters = null;
	private RawDataInterface transformed = null;
	private int maxComponents = 2;
	
	/**
	 * Package-protected constructor.
	 * 
	 * @param cubeData
	 * @param factory
	 */
	WekaPrincipalComponents (RMICubeData cubeData) {
		this.cubeData = cubeData;
	}

	/**
	 * Package-protected constructor.
	 * 
	 * @param cubeData
	 * @param factory
	 */
	WekaPrincipalComponents (RawDataInterface rawData) {
		this.rawData = rawData;
	}

	/* (non-Javadoc)
	 * @see clustering.PrincipalComponentsAnalysisInterface#doPCA()
	 */
	public void doPCA() throws ClusterException {
		
		if (this.rawData != null) {
			// this code is for performing PCA on the full data set
			try {
				this.pca = new PrincipalComponents();
				if (this.maxComponents > 0)
					pca.setMaximumAttributeNames(this.maxComponents);
				pca.setNormalize(false);
				pca.setTransformBackToOriginal(false);
				Instances instances = (Instances)rawData.getData(); 
				pca.buildEvaluator(instances);
				System.out.println("variance covered: " + pca.getVarianceCovered());
				for (int i = 0 ; i < ((Instances)rawData.getData()).numAttributes() ; i++) {
					System.out.println("merit["+i+"]: " + pca.evaluateAttribute(i));
				}
				components = pca.transformedData(instances);
				transformed = new WekaRawData(components);
			} catch (Exception e) {
				System.err.println("Error performing PCA on dataset");
				e.printStackTrace(System.err);
			}
		} else {
			// this code is for performing correlation analysis on two components.
			ArrayList<String> names = new ArrayList<String>();
			for (int i = 0 ; i < maxComponents ; i++) {
				names.add(cubeData.getNames()[i]);
			}
	
			transformed = AnalysisFactory.createRawData("Scatterplot Data",
				names, 2,inputData.numVectors(), null);
			for (int i = 0 ; i < inputData.numVectors() ; i++) {
				float[] values = cubeData.getValues(i);
				for (int j = 0 ; j < maxComponents ; j++) {
					transformed.addValue(j,i,(double)(values[j]));
				}
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
	 * @see clustering.PrincipalComponentsAnalysisInterface#setInputData(clustering.RawDataInterface)
	 */
	public void setInputData(RawDataInterface inputData) {
		this.inputData = inputData;
		this.instances = (Instances) inputData.getData();
	}

	/* (non-Javadoc)
     * @see clustering.PrincipalComponentsAnalysisInterface#getCorrelationCoefficients()
     */
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

	/* (non-Javadoc)
     * @see clustering.PrincipalComponentsAnalysisInterface#getClusters()
     */
	public RawDataInterface[] getClusters () {
		if (this.clusterer != null) {
			int[] clusterSizes = clusterer.getClusterSizes();
			int k = clusterer.getClusterSizes().length;
			if (k == 0) {
				clusters = null;
				return clusters;
			}
			if (clusterer instanceof DBScanClusterInterface) {
				k++; // add one for noise
			}
			clusters = new RawDataInterface[k];
			Instances[] instances = new Instances[k];
			int[] counters = new int[k];
			
			for (int i = 0 ; i < k ; i++) {
				// we have to do this check, because sometimes Weka creates
				// empty clusters, and removes them.
				if (i >= clusterSizes.length)
					instances[i] = new Instances((Instances)(transformed.getData()), 0);
				else
					instances[i] = new Instances((Instances)(transformed.getData()), clusterSizes[i]);
				counters[i] = 0;
			}

			//int x = transformed.numDimensions() - 1;
			//int y = transformed.numDimensions() - 2;
			//int x = 0;
			//int y = 1;
			for (int i = 0 ; i < inputData.numVectors() ; i++) {
				double values[] = new double[2];
				int location = clusterer.clusterInstance(i);
				if (location < 0) {
					location = instances.length-1; // put the noise in the last cluster
				}
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

	/* (non-Javadoc)
     * @see clustering.PrincipalComponentsAnalysisInterface#setClusterer()
     */
	public void setClusterer (ClusterInterface clusterer) {
		this.clusterer = clusterer;
	}

	/**
	 * @return the maxComponents
	 */
	public int getMaxComponents() {
		return maxComponents;
	}

	/**
	 * @param maxComponents the maxComponents to set
	 */
	public void setMaxComponents(int maxComponents) {
		this.maxComponents = maxComponents;
	}

}
