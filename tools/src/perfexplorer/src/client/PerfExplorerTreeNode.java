/**
 * Created February 14, 2006
 * 
 */
package edu.uoregon.tau.perfexplorer.client;

import javax.swing.tree.DefaultMutableTreeNode;

import edu.uoregon.tau.perfexplorer.common.RMIView;


public class PerfExplorerTreeNode extends DefaultMutableTreeNode {
    /**
	 * 
	 */
	private static final long serialVersionUID = -4747248940980600950L;
	private boolean _allowsChildren = true;

    public PerfExplorerTreeNode(Object nodeObject) {
        super(nodeObject);
		if (nodeObject instanceof RMIView) {
			RMIView view = (RMIView)nodeObject;
			view.setDMTN(this);
		}
    }
    
    public PerfExplorerTreeNode(Object nodeObject, boolean allowsChildren) {
        super(nodeObject);
        this._allowsChildren = allowsChildren;
    }
    
    public boolean getAllowsChildren() {
        return this._allowsChildren;
    }
    
}

