/*
 * Created on Mar 18, 2005
 *
 */
package clustering.weka;

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
 * <P>CVS $Id: WekaAnalysisFactory.java,v 1.3 2007/01/04 21:20:02 khuck Exp $</P>
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
     * @see clustering.AnalysisFactory#createDataNormalizer()
     */
    public DataNormalizer createDataNormalizer(RawDataInterface inputData) {
        return new WekaDataNormalizer(inputData);
    }

	public void closeFactory() {
		// do nothing
		return;
	}
}
