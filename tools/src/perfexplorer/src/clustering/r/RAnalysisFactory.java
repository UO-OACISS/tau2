/*
 * Created on Mar 18, 2005
 *
 */
package clustering.r;

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
 * <P>CVS $Id: RAnalysisFactory.java,v 1.10 2008/08/05 00:18:17 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class RAnalysisFactory extends AnalysisFactory {

	/**
	 * Protected constructor.
	 *
	 */
	protected RAnalysisFactory () {
	}

	/* (non-Javadoc)
	 * @see clustering.AnalysisFactory#createRawData(java.lang.String, java.util.List, int, int)
	 */
	public RawDataInterface createRawData(String name, List attributes,
			int vectors, int dimensions, List<String> classAttributes) {
		Object[] objects = attributes.toArray();
		//String[] eventNames = (String[])(objects);
		// do something with the name and attributes?
		String[] eventNames = new String[attributes.size()];
		for (int i = 0 ; i < attributes.size() ; i++) {
			eventNames[i] = (String) attributes.get(i);
		}

		return new RRawData(vectors, dimensions, eventNames);
	}

	/* (non-Javadoc)
	 * @see clustering.AnalysisFactory#createKMeansEngine()
	 */
	public KMeansClusterInterface createKMeansEngine() {
		return new RKMeansCluster();
	}
	
	/* (non-Javadoc)
	 * @see clustering.AnalysisFactory#createPCAEngine()
	 */
	public PrincipalComponentsAnalysisInterface createPCAEngine(RMICubeData cubeData) {
		return new RPrincipalComponents(cubeData);
	}

    /* (non-Javadoc)
     * @see clustering.AnalysisFactory#createDataNormalizer()
     */
    public DataNormalizer createDataNormalizer
        (RawDataInterface inputData) {
        return new RDataNormalizer(inputData);
    }

	public void closeFactory() {
		RSingletons.endRSession();
	}

	@Override
	public LinearRegressionInterface createLinearRegressionEngine() {
		// TODO Auto-generated method stub
		System.out.println("linear regression for R is UNIMPLEMENTED");
		return null;
	}

	@Override
	public PrincipalComponentsAnalysisInterface createPCAEngine(RawDataInterface rawData) {
		// TODO Auto-generated method stub
		System.out.println("this PCA for R is UNIMPLEMENTED");
		return null;
	}

 	/* (non-Javadoc)
     * @see clustering.AnalysisFactory#createDataNormalizer()
     */
    public Utilities getUtilities () {
        return new RUtilities();
    }

	@Override
	public ClassifierInterface createNaiveBayesClassifier(
			RawDataInterface inputData) {
		// TODO Auto-generated method stub
		System.out.println("this Naive Bayes Classification for R is UNIMPLEMENTED");
		return null;
	}

	@Override
	public ClassifierInterface createSupportVectorClassifier(RawDataInterface inputData) {
		// TODO Auto-generated method stub
		System.out.println("this Support Vector Classification for R is UNIMPLEMENTED");
		return null;
	}

}
