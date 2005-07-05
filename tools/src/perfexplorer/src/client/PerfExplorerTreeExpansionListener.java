package client;

import javax.swing.*;
import javax.swing.event.*;

public class PerfExplorerTreeExpansionListener implements TreeExpansionListener, TreeWillExpandListener {

	private JTree tree;
	public PerfExplorerTreeExpansionListener(JTree tree) {
		super();
		this.tree = tree;
	}

	public void treeWillExpand (TreeExpansionEvent e) {
	/*
		TreePath path = e.getPath();
		if (path == null)
			return;
		DefaultMutableTreeNode node = (DefaultMutableTreeNode) path.getLastPathComponent();
		if (node.isRoot()) {
			// do nothing
		} else if (node.toString().equals("Database Profiles")) {
			PerfExplorerJTree.addApplicationNodes(node, false);
		} else if (node.toString().equals("Views")) {
			PerfExplorerJTree.addViewNodes(node, "0");
		} else {
			Object object = node.getUserObject();
			if (object instanceof Application) {
				Application app = (Application)object;
				PerfExplorerJTree.addExperimentNodes (node, app, true);
			} else if (object instanceof Experiment) {
				Experiment exp = (Experiment)object;
				PerfExplorerJTree.addTrialNodes (node, exp);
			} else if (object instanceof Trial) {
				Trial trial = (Trial)object;
				PerfExplorerJTree.addMetricNodes (node, trial);
			} else if (object instanceof Metric) {
				// do nothing
			} else if (object instanceof RMIView) {
				RMIView view = (RMIView) object;
				PerfExplorerJTree.addViewNodes(node, view.getField("id"));
			} else {
				System.out.println("unknown!");
			}
		}
		*/
	}

	public void treeExpanded(TreeExpansionEvent e) {
	}

	public void treeWillCollapse (TreeExpansionEvent e) {
	/*
		TreePath path = e.getPath();
		if (path == null)
			return;
		DefaultMutableTreeNode node = (DefaultMutableTreeNode) path.getLastPathComponent();
		if (!node.isRoot()) {
			node.removeAllChildren();
		}
		*/
	}

	public void treeCollapsed(TreeExpansionEvent e) {
	}
}
