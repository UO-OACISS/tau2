/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering;


 /**
  * This interface is used to define the methods to implement a hierarchical 
  * clustering class.
  *
  * <P>CVS $Id: KMeansClusterInterface.java,v 1.7 2009/10/23 16:26:16 khuck Exp $</P>
  * @author khuck
  * @version 0.1
  * @since   0.1
  *
  */
public interface KMeansClusterInterface extends ClusterInterface {

    /**
     * Set the indices of the initial centers for K means.
     * This method is necessary to get repeatable clustering results.
     * 
     * @param indexes
     */
    public void setInitialCenters(int[] indexes);
    
    // TODO - remove this!
    public void doPCA(boolean doPCA);
    
    /**
     * Initialize the K means with good initial centers, rather than random
     * This is slower, but more accurate.
     * 
     * @param b
     */
	public void doSmartInitialization(boolean b);
}
