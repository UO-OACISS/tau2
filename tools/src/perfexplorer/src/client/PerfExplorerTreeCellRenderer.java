package client;

import java.awt.*;
import javax.swing.*;
import javax.swing.tree.*;
import edu.uoregon.tau.perfdmf.*;

public class PerfExplorerTreeCellRenderer extends DefaultTreeCellRenderer{
	public Component getTreeCellRendererComponent(JTree tree,
						  Object value,
						  boolean selected,
						  boolean expanded,
						  boolean leaf,
						  int row,
						  boolean hasFocus){
	super.getTreeCellRendererComponent(tree,value,selected,expanded,leaf,row,hasFocus);
	DefaultMutableTreeNode node = (DefaultMutableTreeNode) value;
	Object userObject = node.getUserObject();
	
	if(node.isRoot()) {
		ImageIcon icon = null;
		this.setIcon(icon);
	}
	if(userObject instanceof Metric){
		ImageIcon icon = createImageIcon("green-ball.gif");
		this.setIcon(icon);
	}
	//else if (expanded) {
		//this.setIcon(openIcon);
	//}
	//else {
		//this.setIcon(closedIcon);
	//}
	return this;
	}

	/** Returns an ImageIcon, or null if the path was invalid. */
	protected static ImageIcon createImageIcon(String path) {
		java.net.URL imgURL = PerfExplorerTreeCellRenderer.class.getResource(path);
		if (imgURL != null) {
			return new ImageIcon(imgURL);
		} else {
			System.err.println("Couldn't find file: " + path);
			return null;
		}
	}
}
