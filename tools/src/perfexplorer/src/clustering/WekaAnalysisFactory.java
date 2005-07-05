/*
 * Created on Mar 18, 2005
 *
 */
package clustering;

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
	 * @see clustering.AnalysisFactory#CreateRawData(java.lang.String, java.util.List, int, int)
	 */
	public RawDataInterface CreateRawData(String name, List attributes,
			int vectors, int dimensions) {
		return new WekaRawData(name, attributes, vectors, dimensions);
	}

	/* (non-Javadoc)
	 * @see clustering.AnalysisFactory#CreateKMeansEngine()
	 */
	public KMeansClusterInterface CreateKMeansEngine() {
		return new WekaKMeansCluster();
	}
	
	/* (non-Javadoc)
	 * @see clustering.AnalysisFactory#CreatePCAEngine()
	 */
	public PrincipalComponentsAnalysisInterface CreatePCAEngine() {
		return new WekaPrincipalComponents();
	}

}
