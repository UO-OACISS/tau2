/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering;


 /**
  * This interface is used to define the methods to implement a hierarchical 
  * clustering class.
  *
  * <P>CVS $Id: DBScanClusterInterface.java,v 1.1 2009/11/18 17:45:17 khuck Exp $</P>
  * @author khuck
  * @version 0.2
  * @since   0.2
  *
  */
public interface DBScanClusterInterface extends ClusterInterface {

    /**
     * Set the value of error (determines the maximum distance between 
	 * points in the same cluster)
     * 
     * @param k
     */
    public void setError(double e);

    /**
     * Get the value of error (determines the maximum distance between 
	 * points in the same cluster)
     * 
     * @return
     */
    public double getError();
       
}
