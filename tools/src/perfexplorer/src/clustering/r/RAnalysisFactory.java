/*
 * Created on Mar 18, 2005
 *
 */
package clustering.r;

import clustering.AnalysisFactory;
import clustering.RawDataInterface;
import clustering.KMeansClusterInterface;
import clustering.PrincipalComponentsAnalysisInterface;
import clustering.DataNormalizer;
import common.RMICubeData;
import java.util.List;

/**
 * This class is an extention of the AnalysisFactory class.  This class
 * should never be directly created - use the static method in the
 * AnalysisFactory class.
 *
 * <P>CVS $Id: RAnalysisFactory.java,v 1.5 2007/01/04 21:20:02 khuck Exp $</P>
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
			int vectors, int dimensions) {
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
}
