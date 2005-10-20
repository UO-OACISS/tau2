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

	public void closeFactory() {
		RSingletons.endRSession();
	}
}
