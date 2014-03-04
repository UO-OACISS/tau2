package edu.uoregon.tau.perfexplorer.client;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import javax.swing.JTree;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.MutableTreeNode;
import javax.swing.tree.TreeNode;
import javax.swing.tree.TreePath;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfexplorer.common.RMISortableIntervalEvent;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;

public class PerfExplorerJTree extends JTree{

    /**
     * 
     */
    private static final long serialVersionUID = -7184526063985636881L;
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
	addMouseListener(
		new PerfExplorerTreeMouseListener(this));
    }

    public static PerfExplorerJTree getTree() {
	if (theTree == null) {
	    DefaultTreeModel model = new DefaultTreeModel(createNodes()){

			private static final long serialVersionUID = 1L;

			public void valueForPathChanged(TreePath path, Object newValue) {
				MutableTreeNode aNode = (MutableTreeNode) path
				.getLastPathComponent();
				handleRename((DefaultMutableTreeNode) aNode, newValue);
				PerfExplorerJTree.nodeChanged((DefaultMutableTreeNode) aNode);

			}
		};
	    
	    
	    model.setAsksAllowsChildren(true);
	    theTree = new PerfExplorerJTree(model);
	    theModel = model;
	    theTree.setEditable(true);
	    //addTrialsForViews();
	}
	return theTree;
    }
    
    protected static void handleRename(DefaultMutableTreeNode aNode, Object newValue) {
		if (newValue instanceof String) {
			String name = (String) newValue;
			if (aNode.getUserObject() instanceof Application) {
				Application application = (Application) aNode
				.getUserObject();
				application.setName(name);

				//if (application.dBApplication()) {
					DatabaseAPI databaseAPI = PerfExplorerServer.getServer().getSession(application
							.getDatabase());
					if (databaseAPI != null) {
						databaseAPI.saveApplication(application);
						//databaseAPI.terminate();
					}
				//}

			} else if (aNode.getUserObject() instanceof Experiment) {
				Experiment experiment = (Experiment) aNode
				.getUserObject();
				experiment.setName(name);

				//if (experiment.dBExperiment()) {
					DatabaseAPI databaseAPI = PerfExplorerServer.getServer().getSession(experiment
							.getDatabase());
					if (databaseAPI != null) {
						databaseAPI.saveExperiment(experiment);
						//databaseAPI.terminate();
					}
				//}

			} else if (aNode.getUserObject() instanceof Trial) {
				Trial ppTrial = (Trial) aNode.getUserObject();
				
				//if (ppTrial.dBTrial()) {
					DatabaseAPI databaseAPI = PerfExplorerServer.getServer().getSession(ppTrial
							.getDatabase());
					//ppTrial.setDatabaseAPI(databaseAPI);
				//}
				ppTrial.rename(databaseAPI.db(),name);

//			} else if (aNode.getUserObject() instanceof Metric) {
//				Metric metric = (Metric) aNode.getUserObject();
//				//if (metric.dbMetric()) {
//					DatabaseAPI databaseAPI = PerfExplorerServer.getServer().getSession(metric..getParaProfTrial()
//							.getDatabase());
//					metric.rename(databaseAPI.db(), name);
//				//}
				
			} else if (aNode.getUserObject() instanceof View) {
				View view = (View) aNode.getUserObject();
				DatabaseAPI databaseAPI = PerfExplorerServer.getServer().getSession(view.getDatabase());

				view.rename(databaseAPI.db(), name);
				

			}

		}

	}

    public static void nodeChanged(DefaultMutableTreeNode node) {
	if (theModel != null && theTree != null) {
	    theTree.setVisible(false);
	    theModel.nodeChanged(node);
	    theModel.reload(node);
	    theTree.repaint();
	    theTree.setVisible(true);
	}
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

	public static void addViewNodes(DefaultMutableTreeNode parentNode,
			int parent) {
		setConnectionIndex(parentNode);
		// get the top level views
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		List<View> viewVector = server.getViews(parent);
		Iterator<View> views = viewVector.iterator();
		while (views.hasNext()) {
			View view = views.next();
			DefaultMutableTreeNode node = new PerfExplorerTreeNode(view);
			parentNode.add(node);
			//addViewNodes(node, view.getID());
		}
		leafViews.add(parentNode);
		
		if (viewVector.size() == 0) {
			addTrialsForView(parentNode);
		}
		else{
			View parentView = (View)parentNode.getUserObject();
			View view = View.VirtualView(parentView);
			
			DefaultMutableTreeNode node = new PerfExplorerTreeNode(view);
			parentNode.add(node);
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
		    app = applications.next();
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
	    exp = experiments.next();
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
	ListIterator<Trial> trials = server.getTrialList(exp.getID(),false);
	Trial trial = null;
	DefaultMutableTreeNode trialNode = null;
	// loop through all the trials, and print out some info
	while(trials.hasNext())
	{
	    trial = trials.next();
	    trialNode = new PerfExplorerTreeNode (trial);
	    //addMetricNodes(trialNode, trial);
	    node.add(trialNode);
	}
    }

    public static void addTAUdbViewNodes (DefaultMutableTreeNode parentNode, int parent) {
	setConnectionIndex(parentNode);
	if (parentNode.getUserObject() instanceof ConnectionNodeObject) {
		leafViews = new ArrayList<DefaultMutableTreeNode>();
	}
	// get the top level views
	PerfExplorerConnection server = PerfExplorerConnection.getConnection();
	List<View> viewVector = server.getViews(parent);
	Iterator<View> views = viewVector.iterator();
	while (views.hasNext()) {
	    View view = views.next();
	    DefaultMutableTreeNode node = new PerfExplorerTreeNode (view);
	    parentNode.add(node);

	    // DO NOT iterate over the children of this view. They will get loaded if they are expanded.
	    //addTAUdbViewNodes(node, view.getID());
	}
//	if (viewVector.size() == 0) {
//	    leafViews.add(parentNode);
//	    addTrialsForView(parentNode);
//	}
    }

    public static void addTrialsForViews () {
	Iterator<DefaultMutableTreeNode> e = leafViews.iterator();
	while (e.hasNext()) {
	    PerfExplorerTreeNode node = (PerfExplorerTreeNode) e.next();
	    addTrialsForView(node);
	}
    }

	public static void addTrialsForView(DefaultMutableTreeNode node) {
		setConnectionIndex(node);
		// System.out.println("trial nodes...");
		Object[] objects = node.getUserObjectPath();
		List<View> views = new ArrayList<View>();
		for (int i = 0; i < objects.length; i++) {
			if (objects[i] instanceof View) {
				views.add((View) objects[i]);
			}
		}
		
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the trials
		if (views.size() > 0) {
			ListIterator<Trial> trials = server.getTrialsForView(views, false);
			Trial trial = null;
			DefaultMutableTreeNode trialNode = null;
			// loop through all the trials, and print out some info
			while (trials.hasNext()) {
				trial = trials.next();
				trialNode = new PerfExplorerTreeNode(trial);
				node.add(trialNode);
				trialNode.getParent();
			}
		}
	}


    @SuppressWarnings("unchecked") // for trial.getMetrics() call
    public static void addMetricNodes (DefaultMutableTreeNode node, Trial trial) {
	setConnectionIndex(node);
	//System.out.println("metric nodes...");
	// get the metrics
	List<Metric> metricVector = trial.getMetrics();
	if(metricVector==null){
		trial.getTrialMetrics(PerfExplorerServer.getServer().getDB());
		metricVector = trial.getMetrics();
	}
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
	RMISortableIntervalEvent event = null;
	DefaultMutableTreeNode eventNode = null;
	// loop through all the events, and print out some info
	while(events.hasNext())
	{
	    event = events.next();
	    eventNode = new PerfExplorerTreeNode (event, false);
	    node.add(eventNode);
	}
    }

    public static int getConnectionIndex(DefaultMutableTreeNode node) {
    	//Don't silent ignore if the connection index is not found
	
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
	    return conn.index;
	}
	return -1;
    }

    public static void setConnectionIndex(DefaultMutableTreeNode node) {
	int index = getConnectionIndex(node);
	PerfExplorerModel.getModel().setConnectionIndex(index);
	PerfExplorerConnection.getConnection().setConnectionIndex(index);
    }

    // If expand is true, expands all nodes in the tree.
    // Otherwise, collapses all nodes in the tree.
    public void expandAll(boolean expand) {
	
	TreeNode root = (TreeNode)getTree().getModel().getRoot();

	// Traverse tree from root
	expandAll(new TreePath(root), expand);
    }
    public void expandToMetricsAll(TreePath parent) {
	TreeNode node = (TreeNode)parent.getLastPathComponent();
	if(((PerfExplorerTreeNode)node).getUserObject() instanceof Metric){
	    return;
	}else if(((PerfExplorerTreeNode)node).getUserObject() instanceof Experiment){
	   collapsePath(parent);
	}
	    expandPath(parent);


	if (node.getChildCount() >= 0) {
	    
	    for (Enumeration e=node.children(); e.hasMoreElements(); ) {
		TreeNode n = (TreeNode)e.nextElement();
 		TreePath path = parent.pathByAddingChild(n);
		expandToMetricsAll(path);
	    }
	}


    }

    public void expandAll(TreePath parent, boolean expand) {
	// Traverse children
	TreeNode node = (TreeNode)parent.getLastPathComponent();


	// Expand current node (or else there won't be children)
	if (expand) {
	    expandPath(parent);
	} 



	if (node.getChildCount() >= 0) {
	    for (Enumeration e=node.children(); e.hasMoreElements(); ) {
		TreeNode n = (TreeNode)e.nextElement();
		TreePath path = parent.pathByAddingChild(n);
		expandAll(path, expand);
	    }
	}

	// collapse must be done bottom-up
	if (!expand) {
	    collapsePath(parent);
	}
    }


    // If expand is true, expands the most recently
    // added trial in the database.
    // Otherwise, collapses all nodes in the tree.
    public void expandLast(boolean expand) {
	TreeNode root = (TreeNode)getTree().getModel().getRoot();

	// Traverse tree from root
	expandLastAdded(new TreePath(root), expand);
    }

    public void expandLastAdded(TreePath parent, boolean expand) {

	PerfExplorerTreeNode node = (PerfExplorerTreeNode)parent.getLastPathComponent();
	Object obj = node.getUserObject();
	// don't expand the "view" node, but the last application before it.
	if (obj instanceof ConnectionNodeObject) {
	    String connString = PerfExplorerConnection.getConnection().getConnectionString();
	    if (connString == null || !connString.equals(obj.toString())) {
		return;
	    }
	} else if (obj instanceof Application) {
	    Application app = PerfExplorerModel.getModel().getApplication();
	    Application current = (Application)obj;
	    if (app == null || app.getID() != current.getID()) {
		return;  // this isn't our current path, so don't expand
	    }
	} else if (obj instanceof Experiment) {
	    Experiment exp = PerfExplorerModel.getModel().getExperiment();
	    Experiment current = (Experiment)obj;
	    if (exp == null || exp.getID() != current.getID()) {
		return;  // this isn't our current path, so don't expand
	    }
	} else if (obj instanceof Trial) {
	    Trial trial = PerfExplorerModel.getModel().getTrial();
	    Trial current = (Trial)obj;
	    if (trial == null || trial.getID() != current.getID()) {
		return;  // this isn't our current path, so don't expand
	    }
	} else if (obj instanceof View) {
	    return;  // this isn't our current path, so don't expand
	} else if (obj instanceof Metric) {
	    return;  // don't expand that deep
	} 

	// Expand current node (or else there won't be children)
	if (expand) {
	    expandPath(parent);
	} 

	// Traverse children
	if (node.getChildCount() >= 0) {
	    for (Enumeration e=node.children(); e.hasMoreElements(); ) {
		TreeNode n = (TreeNode)e.nextElement();
		TreePath path = parent.pathByAddingChild(n);
		expandLastAdded(path, expand);
	    }
	}

	// collapse must be done bottom-up
	if (!expand) {
	    collapsePath(parent);
	}
    }

	

}
