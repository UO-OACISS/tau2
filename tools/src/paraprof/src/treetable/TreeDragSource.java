package edu.uoregon.tau.paraprof.treetable;

import java.awt.dnd.*;

import javax.swing.tree.*;
import javax.swing.*;

import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.paraprof.ParaProfTrial;

public class TreeDragSource implements DragSourceListener, DragGestureListener {

    DragSource source;
    DragGestureRecognizer recognizer;
    TransferableTreeNode transferable;
    DefaultMutableTreeNode oldNode;
    JTree sourceTree;

    public TreeDragSource(JTree tree, int actions) {
	sourceTree = tree;
	source = new DragSource();
	recognizer = source.createDefaultDragGestureRecognizer(sourceTree,
		actions, this);
    }

    /*
     *   * Drag Gesture Handler
     *     */
     public void dragGestureRecognized(DragGestureEvent dge) {
	TreePath path = sourceTree.getSelectionPath();
	if ((path == null) || (path.getPathCount() <= 1)) {
	    //We can't move the root node or an empty selection
	    return;
	}
	
	oldNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	if(oldNode.getUserObject() instanceof ParaProfTrial){
	 ParaProf.paraProfManagerWindow.expand(oldNode);
	}
	transferable = new TransferableTreeNode(oldNode);
	source.startDrag(dge, DragSource.DefaultMoveDrop, transferable, this);

	// If you support dropping the node anywhere, you should probably
	// start with a valid move cursor:
	//source.startDrag(dge, DragSource.DefaultMoveDrop, transferable,
	// this);
     }

     /*
      * Drag Event Handlers
      */
     public void dragEnter(DragSourceDragEvent dsde) {
     }

     public void dragExit(DragSourceEvent dse) {
     }

     public void dragOver(DragSourceDragEvent dsde) {
	
     }

     public void dropActionChanged(DragSourceDragEvent dsde) {
	
     }

     public void dragDropEnd(DragSourceDropEvent dsde) {
	 /*
	  * to support move or copy, we have to check which occurred:
	  */
	 if (dsde.getDropSuccess()
		 && (dsde.getDropAction() == DnDConstants.ACTION_MOVE)) {
	    // ((DefaultTreeModel) sourceTree.getModel())
	     //.removeNodeFromParent(oldNode);
	 }

	 /*
	  * to support move only... if (dsde.getDropSuccess()) {
	  * ((DefaultTreeModel)sourceTree.getModel()).removeNodeFromParent(oldNode); }
	  */
     }
}

