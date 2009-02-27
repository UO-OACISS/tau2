/*
 * Created on Mar 18, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering.weka;


import java.util.List;

import edu.uoregon.tau.perfexplorer.clustering.AnalysisFactory;
import edu.uoregon.tau.perfexplorer.clustering.ClassifierInterface;
import edu.uoregon.tau.perfexplorer.clustering.DataNormalizer;
import edu.uoregon.tau.perfexplorer.clustering.KMeansClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.LinearRegressionInterface;
import edu.uoregon.tau.perfexplorer.clustering.PrincipalComponentsAnalysisInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.clustering.Utilities;
import edu.uoregon.tau.perfexplorer.common.RMICubeData;

/**
 * This class is an extention of the AnalysisFactory class.  This class
 * should never be directly created - use the static method in the
 * AnalysisFactory class.
 *
 * <P>CVS $Id: WekaAnalysisFactory.java,v 1.10 2009/02/27 00:45:08 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class WekaAnalysisFactory extends AnalysisFactory {

	/* (non-Javadoc)
	 * @see clustering.AnalysisFactory#createRawData(java.lang.String, java.util.List, int, int)
	 */
	public RawDataInterface createRawData(String name, List<String> attributes,
			int vectors, int dimensions, List<String> classAttributes) {
		return new WekaRawData(name, attributes, vectors, dimensions, classAttributes);
	}

	/* (non-Javadoc)
	 * @see clustering.AnalysisFactory#createKMeansEngine()
	 */
	public KMeansClusterInterface createKMeansEngine() {
		return new WekaKMeansCluster();
	}
	
	/* (non-Javadoc)
	 * @see clustering.AnalysisFactory#createPCAEngine()
	 */
	public PrincipalComponentsAnalysisInterface createPCAEngine(RMICubeData cubeData) {
		return new WekaPrincipalComponents(cubeData, this);
	}

	/* (non-Javadoc)
	 * @see clustering.AnalysisFactory#createPCAEngine()
	 */
	public PrincipalComponentsAnalysisInterface createPCAEngine(RawDataInterface rawData) {
		return new WekaPrincipalComponents(rawData, this);
	}

	/* (non-Javadoc)
     * @see clustering.AnalysisFactory#createDataNormalizer()
     */
    public DataNormalizer createDataNormalizer(RawDataInterface inputData) {
        return new WekaDataNormalizer(inputData);
    }

	public void closeFactory() {
		// do nothing
		return;
	}

	@Override
	public LinearRegressionInterface createLinearRegressionEngine() {
		// TODO Auto-generated method stub
		return new WekaLinearRegression();
	}

	public Utilities getUtilities() {
		return new WekaUtilities();
	}

	@Override
	public ClassifierInterface createNaiveBayesClassifier(
			RawDataInterface inputData) {
		return new WekaNaiveBayesClassifier(inputData);
	}

	@Override
	public ClassifierInterface createSupportVectorClassifier(RawDataInterface inputData) {
		return new WekaSupportVectorClassifier(inputData);
	}
}
