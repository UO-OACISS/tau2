/*
 * Created on Apr 1, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering;


 /**
  * This interface is used to define the methods to implement a class
  * which performs PCA, or Principle Components Analysis.
  *
  * <P>CVS $Id: PrincipalComponentsAnalysisInterface.java,v 1.8 2009/11/17 16:31:00 khuck Exp $</P>
  * @author khuck
  * @version 0.1
  * @since   0.1
  *
  */
public interface PrincipalComponentsAnalysisInterface {
    /**
     * This method performs the Principal Components Analysis
     * 
     * @throws ClusterException
     */
    public void doPCA() throws ClusterException;

    /**
     * This method gets the ith Principal Comonent object
     * 
     * @param i
     * @return a ClusterDescription
     * @throws ClusterException
     */
    public ClusterDescription getComponentDescription(int i) 
        throws ClusterException;

    /**
     * Sets the input data for the clustering operation.
     * 
     * @param inputData
     */
    public void setInputData(RawDataInterface inputData);
    
    /**
     * Returns the results of the PCA analysis.
     * @return
     */
    public RawDataInterface getResults();

    /**
     * Specifies the clusterer to use if you wish to perform PCA after the
     * clustering has been done.  This is used for linear projection of the
     * results.
     * 
     * @param clusterer
     */
    public void setClusterer(ClusterInterface clusterer);

    /**
     * Used to return the PCA reduced data.
     * 
     * @return
     */
    public RawDataInterface[] getClusters();

    /**
     * Used to reset the PCA.  If the user wishes to rerun the analysis
     * with different parameters, they don't have to create a new
     * PCA class to do it.
     */
    public void reset();
    
    /**
     * Sets the maximum number of components returned.
     * 
     * @param maxComponents
     */
    public void setMaxComponents(int maxComponents);
}
