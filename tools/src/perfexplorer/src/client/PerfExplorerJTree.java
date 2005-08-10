package client;

import java.util.ListIterator;
import javax.swing.*;
import javax.swing.tree.*;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;
import edu.uoregon.tau.dms.dss.*;
import common.RMIView;;

public class PerfExplorerJTree extends JTree {

	private static PerfExplorerJTree theTree = null;
	private static List leafViews = null;

	private PerfExplorerJTree (DefaultTreeModel model) {
		super(model);
		putClientProperty("JTree.lineStyle", "Angled");
		//getSelectionModel().setSelectionMode (
			//TreeSelectionModel.SINGLE_TREE_SELECTION);
		PerfExplorerTreeCellRenderer renderer = 
			new PerfExplorerTreeCellRenderer();
		setCellRenderer(renderer);
		addTreeSelectionListener (
			new PerfExplorerTreeSelectionListener(this));
		addTreeWillExpandListener (
			new PerfExplorerTreeExpansionListener(this));
	}

	public static PerfExplorerJTree getTree() {
		if (theTree == null) {
			DefaultTreeModel model = new DefaultTreeModel(createNodes());
			model.setAsksAllowsChildren(true);
			theTree = new PerfExplorerJTree(model);
			//theTree.setShowsRootHandles(true);
			addTrialsForViews();
		}
		return theTree;
	}

	private static DefaultMutableTreeNode createNodes () {
		DefaultMutableTreeNode root = new DefaultMutableTreeNode("Performance Data");
		DefaultMutableTreeNode top = new DefaultMutableTreeNode("Database Profiles");
		DefaultMutableTreeNode viewTop = new DefaultMutableTreeNode("Views");
		addApplicationNodes(top, true);
		leafViews = new ArrayList();
		addViewNodes(viewTop, "0");
		root.add(top);
		root.add(viewTop);

		return root;
	}

	public static void addViewNodes (DefaultMutableTreeNode parentNode, String parent) {
		// get the top level views
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		List viewVector = server.getViews(Integer.parseInt(parent));
		Iterator views = viewVector.iterator();
		while (views.hasNext()) {
			RMIView view = (RMIView) views.next();
			DefaultMutableTreeNode node = new DefaultMutableTreeNode (view);
			addViewNodes(node, view.getField("ID"));
			parentNode.add(node);
		}
		if (viewVector.size() == 0) {
			leafViews.add(parentNode);
			//addTrialsForView(parentNode);
		}
	}

	public static void addApplicationNodes (DefaultMutableTreeNode parent, boolean getExperiments) {
		DefaultMutableTreeNode node = null;
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		if (server != null) {
			// get the applications
			ListIterator applications = server.getApplicationList();
			if (applications != null) {
				Application app = null;

				// loop through all the applications, and print out some info
				while(applications.hasNext())
				{
					app = (Application) applications.next();
					node = new DefaultMutableTreeNode (app);
					addExperimentNodes(node, app, true);
					parent.add(node);
				}
			}
		}

	}

	public static void addExperimentNodes (DefaultMutableTreeNode node, Application app, boolean getTrials) {
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the experiments
		ListIterator experiments = server.getExperimentList(app.getID());
		Experiment exp = null;
		DefaultMutableTreeNode expNode = null;
		// loop through all the experiments, and print out some info
		while(experiments.hasNext())
		{
			exp = (Experiment) experiments.next();
			expNode = new DefaultMutableTreeNode (exp);
			if (getTrials) addTrialNodes(expNode, exp);
			node.add(expNode);
		}
	}

	public static void addTrialNodes (DefaultMutableTreeNode node, Experiment exp) {
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the trials
		ListIterator trials = server.getTrialList(exp.getID());
		Trial trial = null;
		DefaultMutableTreeNode trialNode = null;
		// loop through all the trials, and print out some info
		while(trials.hasNext())
		{
			trial = (Trial) trials.next();
			trialNode = new DefaultMutableTreeNode (trial);
			addMetricNodes(trialNode, trial);
			node.add(trialNode);
		}
	}

	public static void addTrialsForViews () {
		Iterator e = leafViews.iterator();
		while (e.hasNext()) {
			DefaultMutableTreeNode node = (DefaultMutableTreeNode) e.next();
			addTrialsForView(node);
		}
	}

	public static void addTrialsForView (DefaultMutableTreeNode node) {
		Object[] objects = node.getUserObjectPath();
		List views = new ArrayList();
		for (int i = 0 ; i < objects.length ; i++) {
			if (objects[i] instanceof RMIView) {
				views.add(objects[i]);
			}
		}
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the trials
		if (views.size() > 0) {
			ListIterator trials = server.getTrialsForView(views);
			Trial trial = null;
			DefaultMutableTreeNode trialNode = null;
			// loop through all the trials, and print out some info
			while(trials.hasNext())
			{
				trial = (Trial) trials.next();
				trialNode = new DefaultMutableTreeNode (trial);
				addMetricNodes(trialNode, trial);
				node.add(trialNode);
			}
		}
	}


	public static void addMetricNodes (DefaultMutableTreeNode node, Trial trial) {
		// get the metrics
		List metricVector = trial.getMetrics();
		if (metricVector != null) {
			ListIterator metrics = metricVector.listIterator();
			Metric metric = null;
			DefaultMutableTreeNode metricNode = null;
			// loop through all the metrics, and print out some info
			while(metrics.hasNext())
			{
				metric = (Metric) metrics.next();
				metricNode = new DefaultMutableTreeNode (metric);
				node.add(metricNode);
			}
		}
	}
}
