/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering;


 /**
  * This interface is used to define the methods to implement a hierarchical 
  * clustering class.
  *
  * <P>CVS $Id: DBScanClusterInterface.java,v 1.2 2009/11/25 09:15:33 khuck Exp $</P>
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
     * @param e
     */
    public void setError(double e);

    /**
     * Get the value of error (determines the maximum distance between 
	 * points in the same cluster)
     * 
     * @return
     */
    public double getError();

    /**
     * Set the value of minPoints (determines the minimum number of
	 * points required to be considered a cluster)
     * 
     * @param minPoints
     */
    public void setMinPoints(int minPoints);

    /**
     * Gets the value of minPoints (determines the minimum number of
	 * points required to be considered a cluster)
     * 
     * @return
     */
    public int getMinPoints();

    /**
     * Guestimate the value of error (determines the maximum distance between 
	 * points in the same cluster)
     * 
     * @return
     */

    public double guessEpsilon();

    /**
	 * Guestimate the index into the k-distances which contains the value of
	 * error (determines the maximum distance between points in the same
	 * cluster)
     * 
     * @return
     */

    public int guessEpsilonIndex();

   /**
     * Return the k-distance values for the current value of k, which
	 * is the minPoints 
     * 
     * @return
     */
    public double[] getKDistances();
       

}
