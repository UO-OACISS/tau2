package edu.uoregon.tau.perfexplorer.client;

import java.awt.EventQueue;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.swing.JOptionPane;
import javax.swing.JTree;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;

import edu.uoregon.tau.perfdmf.Trial;

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
			final DefaultMutableTreeNode node = (DefaultMutableTreeNode)
					   		tree.getLastSelectedPathComponent();
			if (node == null) return;
			PerfExplorerJTree.setConnectionIndex(node);
	
			final Object currentSelection = node.getUserObject();
			Object[] objectPath = node.getUserObjectPath();
			PerfExplorerModel.getModel().setCurrentSelection(objectPath);
			oldPaths = paths;
			// change to string
			if (currentSelection instanceof Trial) {
				node.setUserObject(new String("Loading data, please wait..."));
				PerfExplorerJTree.nodeChanged(node);
				// update the right side
				Thread thread = new Thread() { 
					public void run() {
						threadWork(node, currentSelection);
					}
				};
				thread.start();
			} else {
				updateRightPanel(currentSelection);
			}
		}
		else {
        	List<Object> multiSelection = new ArrayList<Object>();
        	Set<Integer> connections = new HashSet<Integer>();
        	DefaultMutableTreeNode node = null;
        	for (int i = 0 ; i < paths.length ; i++) {
            	node = (DefaultMutableTreeNode)(paths[i].getLastPathComponent());
            	multiSelection.add(node.getUserObject());
            	connections.add(new Integer(PerfExplorerJTree.getConnectionIndex(node)));
        	}
        	if (connections.size() > 1) {
				JOptionPane.showMessageDialog(null, 
						"Please select only one type (Application, Experiment, Trial, Metric, Event) of level from one database.",
						"Selection Error", JOptionPane.ERROR_MESSAGE);
					// un-select the new ones
					tree.clearSelection();
					// select the old ones
					if (oldPaths != null)
						tree.setSelectionPaths(oldPaths);
        	}
			// don't allow heterogeneous selections
        	else if (!PerfExplorerModel.getModel().setMultiSelection(multiSelection)) {
				JOptionPane.showMessageDialog(null, 
					"Please select only one type (Application, Experiment, Trial, Metric, Event) of level.",
					"Selection Error", JOptionPane.ERROR_MESSAGE);
				// un-select the new ones
				tree.clearSelection();
				// select the old ones
				if (oldPaths != null)
					tree.setSelectionPaths(oldPaths);
			} else {
				PerfExplorerJTree.setConnectionIndex(node);
				oldPaths = paths;
			}
		}
	}

	private void threadWork(final DefaultMutableTreeNode node, final Object currentSelection) {
		EventQueue.invokeLater( new Runnable() { 
			public void run() {
				updateRightPanel(currentSelection);
				// change it back to Trial
				node.setUserObject(currentSelection);
				PerfExplorerJTree.nodeChanged(node);
			}
		});
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
			ChartPane.getPane().drawChart();
			// update the results view
		} else if (index == 4) {
			DeriveMetricsPane.getPane().refreshDynamicControls(true, true, true);
		//	DeriveMetricsPane.getPane().drawChart();
			// update the results view
			
		} else {
			PerformanceExplorerPane.getPane().updateImagePanel();
			// update the results view
		}
		// update the chart control center, if it is open
		ChartGUI chartGui = ChartGUI.getInstance(false);
		if (chartGui != null) {
			chartGui.refresh();
		}
		pane.update();
	}

}
