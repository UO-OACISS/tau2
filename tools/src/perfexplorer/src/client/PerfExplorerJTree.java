package edu.uoregon.tau.perfexplorer.client;

import java.util.ListIterator;
import javax.swing.*;
import javax.swing.tree.*;

import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.perfexplorer.common.RMISortableIntervalEvent;
import edu.uoregon.tau.perfexplorer.common.RMIView;

public class PerfExplorerJTree extends JTree {

	private static PerfExplorerJTree theTree = null;
	private static List<DefaultMutableTreeNode> leafViews = null;
	private static DefaultMutableTreeNode root = null;
	private static DefaultTreeModel theModel = null;

	private PerfExplorerJTree (DefaultTreeModel model) {
		super(model);
		putClientProperty("JTree.lineStyle", "Angled");
		//getSelectionModel().setSelectionMode (
			//TreeSelectionModel.SINGLE_TREE_SELECTION);
		setShowsRootHandles(true);

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
			theModel = model;
			//addTrialsForViews();
		}
		return theTree;
	}

	public static void refreshDatabases () {
		DefaultMutableTreeNode root = PerfExplorerJTree.root;
        DefaultMutableTreeNode treeNode;
        
		root.removeAllChildren();

/*        // make a list of the nodes to remove
        List toRemove = new ArrayList();
        Enumeration nodes = root.children();
        while (nodes.hasMoreElements()) {
            treeNode = (DefaultMutableTreeNode) nodes.nextElement();
            toRemove.add(treeNode);
        }
*/
        // reset the server to get all new configurations
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		server.resetServer();
		List<String> strings = server.getConnectionStrings();

		// add new nodes
		for (int i = 0 ; i < strings.size() ; i++ ) {
			String tmp = (String)strings.get(i);
			DefaultMutableTreeNode top = new PerfExplorerTreeNode(new ConnectionNodeObject(tmp, i));
			root.add(top);
		}

/*		// remove the original nodes
        for (int i = 0; i < toRemove.size(); i++) {
            treeNode = (DefaultMutableTreeNode) toRemove.get(i);
            treeNode.removeFromParent();
            theModel.nodeStructureChanged((TreeNode)treeNode);
        }
*/
		theModel.nodeStructureChanged((TreeNode)root);

		theTree.setVisible(false);
		theTree.validate();
		theModel.reload();
		theTree.setVisible(true);
	}

	private static DefaultMutableTreeNode createNodes () {
		DefaultMutableTreeNode root = new PerfExplorerTreeNode("Performance Data");
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		List<String> strings = server.getConnectionStrings();
		for (int i = 0 ; i < strings.size() ; i++ ) {
			String tmp = (String)strings.get(i);
			DefaultMutableTreeNode top = new PerfExplorerTreeNode(new ConnectionNodeObject(tmp, i));
			root.add(top);
		}
		PerfExplorerJTree.root = root;
		return root;
	}
	
	public void addViewNode(DefaultMutableTreeNode node) {
		DefaultMutableTreeNode viewTop = new PerfExplorerTreeNode("Views");
		//addApplicationNodes(top, true);
		leafViews = new ArrayList<DefaultMutableTreeNode>();
		//addViewNodes(viewTop, "0");
		node.add(viewTop);	
	}

	public static void addViewNodes (DefaultMutableTreeNode parentNode, String parent) {
		setConnectionIndex(parentNode);
		// get the top level views
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		List<RMIView> viewVector = server.getViews(Integer.parseInt(parent));
		Iterator<RMIView> views = viewVector.iterator();
		while (views.hasNext()) {
			RMIView view = (RMIView) views.next();
			DefaultMutableTreeNode node = new PerfExplorerTreeNode (view);
			addViewNodes(node, view.getField("ID"));
			parentNode.add(node);
		}
		if (viewVector.size() == 0) {
			leafViews.add(parentNode);
			addTrialsForView(parentNode);
		}
	}

	public static void addApplicationNodes (DefaultMutableTreeNode parent, boolean getExperiments) {
		setConnectionIndex(parent);
		//System.out.println("application nodes...");
		DefaultMutableTreeNode node = null;
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		if (server != null) {
			// get the applications
			ListIterator<Application> applications = server.getApplicationList();
			if (applications != null) {
				Application app = null;

				// loop through all the applications, and print out some info
				while(applications.hasNext())
				{
					app = (Application) applications.next();
					node = new PerfExplorerTreeNode (app);
					//addExperimentNodes(node, app, true);
					parent.add(node);
				}
			}
		}

	}

	public static void addExperimentNodes (DefaultMutableTreeNode node, Application app, boolean getTrials) {
		setConnectionIndex(node);
		//System.out.println("experiment nodes...");
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the experiments
		ListIterator<Experiment> experiments = server.getExperimentList(app.getID());
		Experiment exp = null;
		DefaultMutableTreeNode expNode = null;
		// loop through all the experiments, and print out some info
		while(experiments.hasNext())
		{
			exp = (Experiment) experiments.next();
			expNode = new PerfExplorerTreeNode (exp);
			//if (getTrials) addTrialNodes(expNode, exp);
			node.add(expNode);
		}
	}

	public static void addTrialNodes (DefaultMutableTreeNode node, Experiment exp) {
		setConnectionIndex(node);
		//System.out.println("trial nodes...");
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the trials
		ListIterator<Trial> trials = server.getTrialList(exp.getID());
		Trial trial = null;
		DefaultMutableTreeNode trialNode = null;
		// loop through all the trials, and print out some info
		while(trials.hasNext())
		{
			trial = (Trial) trials.next();
			trialNode = new PerfExplorerTreeNode (trial);
			//addMetricNodes(trialNode, trial);
			node.add(trialNode);
		}
	}

	public static void addTrialsForViews () {
		Iterator<DefaultMutableTreeNode> e = leafViews.iterator();
		while (e.hasNext()) {
			PerfExplorerTreeNode node = (PerfExplorerTreeNode) e.next();
			addTrialsForView(node);
		}
	}

	public static void addTrialsForView (DefaultMutableTreeNode node) {
		setConnectionIndex(node);
		//System.out.println("trial nodes...");
		Object[] objects = node.getUserObjectPath();
		List<RMIView> views = new ArrayList<RMIView>();
		for (int i = 0 ; i < objects.length ; i++) {
			if (objects[i] instanceof RMIView) {
				views.add((RMIView)objects[i]);
			}
		}
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the trials
		if (views.size() > 0) {
			ListIterator<Trial> trials = server.getTrialsForView(views);
			Trial trial = null;
			DefaultMutableTreeNode trialNode = null;
			// loop through all the trials, and print out some info
			while(trials.hasNext())
			{
				trial = (Trial) trials.next();
				trialNode = new PerfExplorerTreeNode (trial);
				//addMetricNodes(trialNode, trial);
				node.add(trialNode);
			}
		}
	}


	@SuppressWarnings("unchecked") // for trial.getMetrics() call
	public static void addMetricNodes (DefaultMutableTreeNode node, Trial trial) {
		setConnectionIndex(node);
		//System.out.println("metric nodes...");
		// get the metrics
		List<Metric> metricVector = trial.getMetrics();
		int metricIndex = 0;
		if (metricVector != null) {
			ListIterator<Metric> metrics = metricVector.listIterator();
			Metric metric = null;
			DefaultMutableTreeNode metricNode = null;
			// loop through all the metrics, and print out some info
			while(metrics.hasNext())
			{
				metric = metrics.next();
				metricNode = new PerfExplorerTreeNode (metric, true);
				//addEventNodes(metricNode, trial, metricIndex++);
				node.add(metricNode);
			}
		}
	}

	public static void addEventNodes (DefaultMutableTreeNode node, Trial trial, int metricIndex) {
		setConnectionIndex(node);
		//System.out.println("event nodes...");
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the events
		ListIterator<RMISortableIntervalEvent> events = server.getEventList(trial.getID(), metricIndex);
		IntervalEvent event = null;
		DefaultMutableTreeNode eventNode = null;
		// loop through all the events, and print out some info
		while(events.hasNext())
		{
			event = (IntervalEvent) events.next();
			eventNode = new PerfExplorerTreeNode (event, false);
			node.add(eventNode);
		}
	}
	
	public static int getConnectionIndex(DefaultMutableTreeNode node) {
		int index = 0;
		// find the connection node for this subtree
		DefaultMutableTreeNode parent = node;
		Object obj = parent.getUserObject();
		while (parent != null && !(obj instanceof ConnectionNodeObject)) {
			parent = (DefaultMutableTreeNode)parent.getParent();
			if (parent != null) {
				obj = parent.getUserObject();
			}
		}
		
		if (obj != null && obj instanceof ConnectionNodeObject) {
			ConnectionNodeObject conn = (ConnectionNodeObject)obj;
			index = conn.index;
		}
		return index;
	}
	
	public static void setConnectionIndex(DefaultMutableTreeNode node) {
		int index = getConnectionIndex(node);
		PerfExplorerModel.getModel().setConnectionIndex(index);
		PerfExplorerConnection.getConnection().setConnectionIndex(index);
	}
}
