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
public abstract class AnalysisFactory {
	public abstract RawDataInterface CreateRawData(String name, List attributes, int vectors, int dimensions);
	public abstract KMeansClusterInterface CreateKMeansEngine();
	public abstract PrincipalComponentsAnalysisInterface CreatePCAEngine();
}
