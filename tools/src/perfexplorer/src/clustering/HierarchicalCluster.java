/*
 * Created on Mar 18, 2005
 *
 */

package clustering;

 /**
  * This interface is used to define the methods to implement a hierarchical 
  * clustering class.
  *
  * <P>CVS $Id: HierarchicalCluster.java,v 1.2 2007/01/04 21:20:01 khuck Exp $</P>
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