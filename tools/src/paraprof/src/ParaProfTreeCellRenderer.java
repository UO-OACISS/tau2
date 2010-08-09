/* 
 ParaProfTreeCellRenderer.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:
 */

/*
 To do: Class is complete.
 */

package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.net.URL;

import javax.swing.ImageIcon;
import javax.swing.JTree;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeCellRenderer;

import edu.uoregon.tau.common.Utility;

public class ParaProfTreeCellRenderer extends DefaultTreeCellRenderer {
    /**
	 * 
	 */
	private static final long serialVersionUID = -903882363241486249L;

	public Component getTreeCellRendererComponent(JTree tree, Object value, boolean selected, boolean expanded,
            boolean leaf, int row, boolean hasFocus) {
        super.getTreeCellRendererComponent(tree, value, selected, expanded, leaf, row, hasFocus);
        DefaultMutableTreeNode node = (DefaultMutableTreeNode) value;
        //DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) node.getParent();
        Object userObject = node.getUserObject();

        
        if (node.isRoot()) {
            URL url = Utility.getResource("red-ball.gif");
            if (url != null) {
              this.setIcon(new ImageIcon(url));
            }
        }
        //else if(parentNode.isRoot()){
        //URL url = ParaProfTreeCellRenderer.class.getResource("red-ball.gif");
        //this.setIcon(new ImageIcon(url));
        //}
        else if (userObject instanceof ParaProfMetric) {
            URL url = Utility.getResource("green-ball.gif");
            if (url != null) {
              this.setIcon(new ImageIcon(url));
            }
        }
        return this;
    }
}
