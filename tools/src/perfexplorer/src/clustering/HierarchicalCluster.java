/*
 * Created on Mar 18, 2005
 *
 */

package edu.uoregon.tau.perfexplorer.clustering;

 /**
  * This interface is used to define the methods to implement a hierarchical 
  * clustering class.
  *
  * <P>CVS $Id: HierarchicalCluster.java,v 1.3 2009/02/24 00:53:35 khuck Exp $</P>
  * @author khuck
  * @version 0.1
  * @since   0.1
  *
  */
public interface HierarchicalCluster {

    /**
     * Method to build the resulting dendrogram tree
     * 
     * @return
     */
    public abstract DendrogramTree buildDendrogramTree();
}