/*
 * Created on Mar 18, 2005
 *
 */

package edu.uoregon.tau.perfexplorer.clustering;

 /**
  * This interface is used to define the methods to implement a hierarchical 
  * clustering class.
  *
  * <P>CVS $Id: HierarchicalCluster.java,v 1.4 2009/10/23 16:26:16 khuck Exp $</P>
  * @author khuck
  * @version 0.1
  * @since   0.1
  *
  */
public interface HierarchicalCluster extends ClusterInterface {

    /**
     * Method to build the resulting dendrogram tree
     * 
     * @return
     */
    public abstract DendrogramTree buildDendrogramTree();
}