package client;

import javax.swing.*;
import javax.swing.tree.*;
import javax.swing.event.*;
import java.util.List;
import java.util.ArrayList;

public class PerfExplorerTreeSelectionListener implements TreeSelectionListener {

	private JTree tree;
	private TreePath[] oldPaths = null;
	public PerfExplorerTreeSelectionListener(JTree tree) {
		super();
		this.tree = tree;
	}

	public void valueChanged(TreeSelectionEvent e) {
		TreePath[] paths = tree.getSelectionPaths();
		if (paths == null)
			return;
		if (paths.length == 1) {
			DefaultMutableTreeNode node = (DefaultMutableTreeNode)
					   		tree.getLastSelectedPathComponent();
			if (node == null) return;
	
			Object currentSelection = node.getUserObject();
			Object[] objectPath = node.getUserObjectPath();
			PerfExplorerModel.getModel().setCurrentSelection(objectPath);
			oldPaths = paths;
			updateRightPanel(currentSelection);
		}
		else {
        	List multiSelection = new ArrayList();
        	for (int i = 0 ; i < paths.length ; i++) {
            	DefaultMutableTreeNode node = (DefaultMutableTreeNode)(paths[i].getLastPathComponent());
            	multiSelection.add(node.getUserObject());
        	}
			// don't allow heterogeneous selections
			if (!PerfExplorerModel.getModel().setMultiSelection(multiSelection)) {
				JOptionPane.showMessageDialog(null, 
					"Please select only one type (Application, Experiment, Trial, Metric, Event) of level.",
					"Selection Error", JOptionPane.ERROR_MESSAGE);
				// un-select the new ones
				tree.clearSelection();
				// select the old ones
				if (oldPaths != null)
					tree.setSelectionPaths(oldPaths);
			} else {
				oldPaths = paths;
			}
		}
	}

	public void updateRightPanel(Object object) {
		PerfExplorerJTabbedPane pane = PerfExplorerJTabbedPane.getPane();
		int index = pane.getSelectedIndex();
		if (index == 0) {
			PerfExplorerTableModel model = (PerfExplorerTableModel)AnalysisManagementPane.getPane().getTable().getModel();
			model.updateObject(object);
			// update the managment view
		} else if (index == 3) {
			ChartPane.getPane().refreshDynamicControls(true, true, true);
			// update the results view
		} else {
			PerformanceExplorerPane.getPane().updateImagePanel();
			// update the results view
		}
		pane.update();
	}

}
