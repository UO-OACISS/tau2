/*
 * Created on Mar 18, 2005
 *
 */
package clustering;

import java.util.List;
import common.RMICubeData;

/**
 * @author khuck
 *
 */
public abstract class AnalysisFactory {
	public abstract RawDataInterface createRawData(String name, List attributes, int vectors, int dimensions);
	public abstract KMeansClusterInterface createKMeansEngine();
	public abstract PrincipalComponentsAnalysisInterface createPCAEngine(RMICubeData cubeData);
	public abstract void closeFactory();
}
