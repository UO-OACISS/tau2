package edu.uoregon.tau.paraprof.treetable;

import java.awt.datatransfer.*;
import java.io.IOException;

import javax.swing.tree.*;

public //TransferableTreeNode.java
//A Transferable TreePath to be used with Drag & Drop applications.
//

class TransferableTreeNode implements Transferable {

  //  public static DataFlavor TREE_PATH_FLAVOR = new DataFlavor(TreePath.class,"Tree Path");
    public  DataFlavor TREE_PATH_FLAVOR; 
	
    DataFlavor flavors[]  = new DataFlavor[1];

    DefaultMutableTreeNode node;

    public TransferableTreeNode(DefaultMutableTreeNode tp) {
	node = tp;
	try {
	    TREE_PATH_FLAVOR = new DataFlavor(DataFlavor.javaJVMLocalObjectMimeType +";class="+javax.swing.tree.DefaultMutableTreeNode.class.getName());
	    flavors[0] = TREE_PATH_FLAVOR;
	} catch (ClassNotFoundException e) {
	    // TODO Auto-generated catch block
	    e.printStackTrace();
	}
    }

    public synchronized DataFlavor[] getTransferDataFlavors() {
	return flavors;
    }

    public boolean isDataFlavorSupported(DataFlavor flavor) {
	return (flavor.equals(flavors[0]));
    }

    public synchronized Object getTransferData(DataFlavor flavor)
    throws UnsupportedFlavorException, IOException {

	if (isDataFlavorSupported(flavor)) {
	    return (Object) node;
	} else {
	    throw new UnsupportedFlavorException(flavor);
	}
    }
}

