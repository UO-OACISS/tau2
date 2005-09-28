/*
 * Created on Apr 1, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package clustering;

/**
 * @author khuck
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public interface PrincipalComponentsAnalysisInterface {
	/**
	 * This method performs the Principal Components Analysis
	 * @throws ClusterException
	 */
	public void doPCA() throws ClusterException;

	/**
	 * This method gets the ith Principal Comonent object
	 * @param i
	 * @return a ClusterDescription
	 * @throws ClusterException
	 */
	public ClusterDescription getComponentDescription(int i) throws ClusterException;

	/**
	 * Sets the input data for the clustering operation.
	 * @param inputData
	 */
	public void setInputData(RawDataInterface inputData);
	
	public RawDataInterface getResults();

	public void setClusterer(KMeansClusterInterface clusterer);

	public RawDataInterface[] getClusters();

	public void reset();
	
}
