/**
 * Created February 14, 2006
 * 
 */
package client;

import java.awt.*;
import javax.swing.*;
import javax.swing.tree.*;


public class PerfExplorerTreeNode extends DefaultMutableTreeNode {
    private boolean _allowsChildren = true;

    public PerfExplorerTreeNode(Object nodeObject) {
        super(nodeObject);
    }
    
    public PerfExplorerTreeNode(Object nodeObject, boolean allowsChildren) {
        super(nodeObject);
        this._allowsChildren = allowsChildren;
    }
    
    public boolean getAllowsChildren() {
        return this._allowsChildren;
    }
}

