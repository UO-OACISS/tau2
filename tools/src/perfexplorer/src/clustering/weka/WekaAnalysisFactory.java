/*
 * Created on Mar 18, 2005
 *
 */
package clustering.weka;

import clustering.*;
import common.RMICubeData;
import java.util.List;

/**
 * @author khuck
 *
 */
public class WekaAnalysisFactory extends AnalysisFactory {

	private static WekaAnalysisFactory theFactory = null;
	
	public static WekaAnalysisFactory getFactory() {
		if (theFactory == null) {
			theFactory = new WekaAnalysisFactory();
		}
		return theFactory;
	}
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

	public void closeFactory() {
		// do nothing
		return;
	}
}
