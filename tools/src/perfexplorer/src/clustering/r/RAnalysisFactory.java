/*
 * Created on Mar 18, 2005
 *
 */
package clustering.r;

import clustering.*;
import common.RMICubeData;
import java.util.List;

/**
 * @author khuck
 *
 */
public class RAnalysisFactory extends AnalysisFactory {

	private static RAnalysisFactory theFactory = null;
	
	public static RAnalysisFactory getFactory() {
		if (theFactory == null) {
			theFactory = new RAnalysisFactory();
		}
		return theFactory;
	}
	/* (non-Javadoc)
	 * @see clustering.AnalysisFactory#createRawData(java.lang.String, java.util.List, int, int)
	 */
	public RawDataInterface createRawData(String name, List attributes,
			int vectors, int dimensions) {
		// do something with the name and attributes?
		return new RRawData(vectors, dimensions);
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

}
