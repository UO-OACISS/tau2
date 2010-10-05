package edu.uoregon.tau.perfexplorer.client;

import java.awt.Component;

import javax.swing.ImageIcon;
import javax.swing.JTree;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeCellRenderer;

import edu.uoregon.tau.perfdmf.IntervalEvent;
import edu.uoregon.tau.perfdmf.Metric;

public class PerfExplorerTreeCellRenderer extends DefaultTreeCellRenderer{
	/**
	 * 
	 */
	private static final long serialVersionUID = -7301783435132823652L;

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
	if (userObject instanceof Metric){
		ImageIcon icon = createImageIcon("green-ball.gif");
		this.setIcon(icon);
	}
	if (userObject instanceof IntervalEvent){
		ImageIcon icon = createImageIcon("blue-ball.gif");
		this.setIcon(icon);
	}
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
