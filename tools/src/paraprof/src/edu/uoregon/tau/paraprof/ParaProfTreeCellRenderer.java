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

import java.io.*;
import java.awt.*;
import javax.swing.*;
import javax.swing.tree.*;
import java.net.*;

public class ParaProfTreeCellRenderer extends DefaultTreeCellRenderer{
    public Component getTreeCellRendererComponent(JTree tree,
						  Object value,
						  boolean selected,
						  boolean expanded,
						  boolean leaf,
						  int row,
						  boolean hasFocus){
	super.getTreeCellRendererComponent(tree,value,selected,expanded,leaf,row,hasFocus);
	DefaultMutableTreeNode node = (DefaultMutableTreeNode) value;
	DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) node.getParent();
	Object userObject = node.getUserObject();
	
	if(node.isRoot()){
	    URL url = ParaProfTreeCellRenderer.class.getResource("red-ball.gif");
	    this.setIcon(new ImageIcon(url)); 
	}
	//else if(parentNode.isRoot()){
	//URL url = ParaProfTreeCellRenderer.class.getResource("red-ball.gif");
	//this.setIcon(new ImageIcon(url));
	//}
	else if(userObject instanceof Metric){
	    URL url = ParaProfTreeCellRenderer.class.getResource("green-ball.gif");
	    this.setIcon(new ImageIcon(url));
	}
	return this;
    }
}
