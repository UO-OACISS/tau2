/*
 * Created on Mar 18, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering.r;


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
 * <P>CVS $Id: RAnalysisFactory.java,v 1.12 2009/03/29 21:47:28 khuck Exp $</P>
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
	public RawDataInterface createRawData(String name, List<String> attributes,
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
