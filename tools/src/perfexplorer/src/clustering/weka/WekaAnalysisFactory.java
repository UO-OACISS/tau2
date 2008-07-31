/*
 * Created on Mar 18, 2005
 *
 */
package clustering.weka;

import clustering.AnalysisFactory;
import clustering.ClassifierInterface;
import clustering.KMeansClusterInterface;
import clustering.LinearRegressionInterface;
import clustering.PrincipalComponentsAnalysisInterface;
import clustering.DataNormalizer;
import clustering.RawDataInterface;
import clustering.Utilities;
import common.RMICubeData;

import java.util.List;

/**
 * This class is an extention of the AnalysisFactory class.  This class
 * should never be directly created - use the static method in the
 * AnalysisFactory class.
 *
 * <P>CVS $Id: WekaAnalysisFactory.java,v 1.6 2008/07/31 05:34:55 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class WekaAnalysisFactory extends AnalysisFactory {

	/* (non-Javadoc)
	 * @see clustering.AnalysisFactory#createRawData(java.lang.String, java.util.List, int, int)
	 */
	public RawDataInterface createRawData(String name, List attributes,
			int vectors, int dimensions) {
		return new WekaRawData(name, attributes, vectors, dimensions);
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
}
