/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering;


 /**
  * This interface is used to define the methods to implement a hierarchical 
  * clustering class.
  *
  * <P>CVS $Id: KMeansClusterInterface.java,v 1.8 2009/11/18 17:45:20 khuck Exp $</P>
  * @author khuck
  * @version 0.1
  * @since   0.1
  *
  */
public interface KMeansClusterInterface extends ClusterInterface {

    // TODO - remove this!
    public void doPCA(boolean doPCA);
    
    /**
     * Set the value of K (number of clusters)
     * 
     * @param k
     */
    public void setK(int k);

    /**
     * Get the value of K (number of clusters)
     * 
     * @return
     */
    public int getK();

}
