package client;

import common.RMIView;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.IntervalEvent;
import java.util.List;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;
import javax.swing.tree.TreeModel;
import javax.swing.tree.DefaultTreeModel;


public class PerfExplorerTreeExpansionListener implements TreeExpansionListener, TreeWillExpandListener {

	private JTree tree;
	public PerfExplorerTreeExpansionListener(JTree tree) {
		super();
		this.tree = tree;
	}

	public void treeWillExpand (TreeExpansionEvent e) {
		TreePath path = e.getPath();
		if (path == null)
			return;
		DefaultMutableTreeNode node = (DefaultMutableTreeNode) path.getLastPathComponent();
		if (!node.isRoot()) {
			node.removeAllChildren();
			TreeModel model = tree.getModel();
			if (model instanceof DefaultTreeModel) {
				DefaultTreeModel dModel = (DefaultTreeModel)model;
				dModel.reload(node);
            }
		}

		if (node.isRoot()) {
			// do nothing
		} else if (node.toString() != null && node.toString().startsWith("jdbc:")) {
			PerfExplorerJTree.addApplicationNodes(node, false);
		} else if (node.toString() != null && node.toString().equals("Views")) {
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
				Metric metric = (Metric)object;
				// get the trial
				DefaultMutableTreeNode pNode = (DefaultMutableTreeNode)node.getParent();
				object = pNode.getUserObject();
				Trial trial = (Trial)object;
				// find the metric index
				List metrics = trial.getMetrics();
				for (int i = 0; i < metrics.size() ; i++) {
					Metric m = (Metric)metrics.get(i);
					if (m.getID() == metric.getID()) {
						PerfExplorerJTree.addEventNodes (node, trial, i);
						break;
					}
				}
			} else if (object instanceof IntervalEvent) {
				// do nothing
			} else if (object instanceof RMIView) {
				RMIView view = (RMIView) object;
				PerfExplorerJTree.addViewNodes(node, view.getField("id"));
			} else {
				System.out.println("unknown!");
			}
		}
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
