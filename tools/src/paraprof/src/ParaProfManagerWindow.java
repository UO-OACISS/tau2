/**
 * ParaProfManagerWindow
 * This is the manager window that allows the user to navigate the app/exp/trial tree, 
 * including database access.
 *  
 *
 * Notes: Makes heavy use of the TreeWillExpandListener listener to populate the tree nodes. Before a node
 * is expanded, the node is re-populated with nodes. This ensures that all the
 * user has to do to update the tree is collapse and expand the nodes. Care is
 * taken to ensure that DefaultMutableTreeNode references are cleaned when a node is collapsed.

 * 
 * <P>CVS $Id: ParaProfManagerWindow.java,v 1.50 2010/02/25 21:54:46 smillst Exp $</P>
 * @author   Robert Bell, Alan Morris
 * @version   $Revision: 1.50 $
 * @see      ParaProfManagerTableModel
 */

package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Frame;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.dnd.Autoscroll;
import java.awt.dnd.DnDConstants;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.io.File;
import java.io.FileReader;
import java.io.LineNumberReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Vector;

import javax.swing.Box;
import javax.swing.ImageIcon;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.JTree;
import javax.swing.event.TreeExpansionEvent;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.event.TreeWillExpandListener;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.TableCellRenderer;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.MutableTreeNode;
import javax.swing.tree.TreePath;
import javax.swing.tree.TreeSelectionModel;

import edu.uoregon.tau.common.MetaDataMap;
import edu.uoregon.tau.common.TauRuntimeException;
import edu.uoregon.tau.common.TreeUI;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.paraprof.tablemodel.ApplicationTableModel;
import edu.uoregon.tau.paraprof.tablemodel.ExperimentTableModel;
import edu.uoregon.tau.paraprof.tablemodel.MetricTableModel;
import edu.uoregon.tau.paraprof.tablemodel.TrialCellRenderer;
import edu.uoregon.tau.paraprof.tablemodel.TrialTableModel;
import edu.uoregon.tau.paraprof.tablemodel.ViewTableModel;
import edu.uoregon.tau.paraprof.treetable.TreeDragSource;
import edu.uoregon.tau.paraprof.treetable.TreeDropTarget;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.DBDataSource;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.DataSourceException;
import edu.uoregon.tau.perfdmf.DataSourceExport;
import edu.uoregon.tau.perfdmf.Database;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.UtilFncs;
import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.database.DBConnector;
import edu.uoregon.tau.perfdmf.database.DBManagerListener;
import edu.uoregon.tau.perfdmf.database.DatabaseManagerWindow;
import edu.uoregon.tau.perfdmf.database.ParseConfig;
import edu.uoregon.tau.perfdmf.database.PasswordCallback;
import edu.uoregon.tau.perfdmf.taudb.TAUdbDataSource;
import edu.uoregon.tau.perfdmf.taudb.TAUdbDatabaseAPI;
import edu.uoregon.tau.perfdmf.taudb.TAUdbTrial;
import edu.uoregon.tau.perfdmf.taudb.ViewCreatorGUI;

public class ParaProfManagerWindow extends JFrame implements ActionListener,
TreeSelectionListener, TreeWillExpandListener, DBManagerListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8355033122352555258L;
	private DefaultMutableTreeNode root;
	private JTree tree = null;
	private DefaultTreeModel treeModel = null;
	private DefaultMutableTreeNode standard = null;
	private DefaultMutableTreeNode runtime = null;
	private JSplitPane jSplitInnerPane = null;
	private JSplitPane jSplitOuterPane = null;

	private JCheckBoxMenuItem showApplyOperationItem = null;
	private DerivedMetricPanel derivedMetricPanel = new DerivedMetricPanel(this);
	// private DerivedMetricWindow derivedMetricWindow = new
	// DerivedMetricWindow(this);

	private JScrollPane treeScrollPane;

	private Vector<ParaProfTrial> loadedDBTrials = new Vector<ParaProfTrial>();
	private Vector<ParaProfTrial> loadedTrials = new Vector<ParaProfTrial>();

	// private boolean metaDataRetrieved;

	// Popup menu stuff.
	private JPopupMenu databasePopUp = null;//new JPopupMenu();
	private JPopupMenu TAUdbPopUp = null;// new JPopupMenu();
	private JPopupMenu ViewPopUp = null;//new JPopupMenu();

	private JPopupMenu stdAppPopup = null;// = new JPopupMenu();
	private JPopupMenu stdExpPopup = new JPopupMenu();
	private JPopupMenu stdTrialPopup = new JPopupMenu();
	private JPopupMenu dbAppPopup = null;//new JPopupMenu();
	private JPopupMenu dbExpPopup = new JPopupMenu();
	private JPopupMenu dbTrialPopup = new JPopupMenu();
	private JPopupMenu metricPopup = new JPopupMenu();
	private JPopupMenu multiPopup = null;//new JPopupMenu();

	private JPopupMenu runtimePopup = new JPopupMenu();

	private Object clickedOnObject = null;
	private DefaultMutableTreeNode selectedObject = null;
	private ParaProfMetric operand1 = null;
	private ParaProfMetric operand2 = null;

	private String dbDisplayName;

	private List<Database> databases;
	private JFileChooser expressionFileC = new JFileChooser();
	private static String commandLineConfig;

	public Object getClickedOnObject() {
		return clickedOnObject;
	}

	public DefaultMutableTreeNode getSelectedObject() {
		return selectedObject;
	}

	@SuppressWarnings("unchecked")
	public void refreshDatabases() {
		// System.out.println("refreshing databases...");
		// System.out.println("LOAD cfg file: " + commandLineConfig);
		databases = Database.getDatabases();
		if (commandLineConfig != null) {
			databases.add(new Database("Portal", commandLineConfig));
		}
		Iterator<Database> dbs = databases.iterator();

		DefaultMutableTreeNode treeNode;
		Enumeration<DefaultMutableTreeNode> nodes = root.children();
		while (nodes.hasMoreElements() && dbs.hasNext()) {
			treeNode = nodes.nextElement();
			if (treeNode.getUserObject() != "Standard Applications") {
				Object obj = dbs.next();
				treeNode.setUserObject(obj);
			}
		}

		List<DefaultMutableTreeNode> toRemove = new ArrayList<DefaultMutableTreeNode>();
		while (nodes.hasMoreElements()) {
			treeNode = nodes.nextElement();
			toRemove.add(treeNode);
		}

		while (dbs.hasNext()) {
			Object obj = dbs.next();
			root.add(new DefaultMutableTreeNode(obj));
		}

		for (int i = 0; i < toRemove.size(); i++) {
			treeNode = toRemove.get(i);
			treeNode.removeFromParent();
		}
		getTreeModel().reload();
	}

	public ParaProfManagerWindow() {
		this("");
	}

	public ParaProfManagerWindow(String dbConfig) {
		// System.out.println("load cfg file: " +
		// ParaProf.preferences.getDatabaseConfigurationFile());
		if (dbConfig != null && dbConfig != "") {
			commandLineConfig = dbConfig;
		}

		// Window Stuff.
		int windowWidth = 800;
		int windowHeight = 515;

		// Grab the screen size.
		Toolkit tk = Toolkit.getDefaultToolkit();
		Dimension screenDimension = tk.getScreenSize();
		int screenHeight = screenDimension.height;
		int screenWidth = screenDimension.width;

		Point savedPosition = ParaProf.preferences.getManagerWindowPosition();

		if (savedPosition == null
				|| (savedPosition.x + windowWidth) > screenWidth
				|| (savedPosition.y + windowHeight > screenHeight)) {

			// Find the center position with respect to this window.
			int xPosition = (screenWidth - windowWidth) / 2;
			int yPosition = (screenHeight - windowHeight) / 2;

			// Offset a little so that we do not interfere too much with the
			// main window which comes up in the center of the screen.
			if (xPosition > 50) {
				xPosition = xPosition - 50;
			}
			if (yPosition > 50) {
				yPosition = yPosition - 50;
			}

			this.setLocation(xPosition, yPosition);
		} else {
			this.setLocation(savedPosition);
		}

		if (ParaProf.demoMode) {
			this.setLocation(0, 0);
		}

		setSize(ParaProfUtils.checkSize(new java.awt.Dimension(windowWidth,
				windowHeight)));
		setTitle("TAU: ParaProf Manager");
		ParaProfUtils.setFrameIcon(this);

		// Add some window listener code
		addWindowListener(new java.awt.event.WindowAdapter() {
			public void windowClosing(java.awt.event.WindowEvent evt) {
				thisWindowClosing(evt);
			}
		});

		setupMenus();

		root = new DefaultMutableTreeNode("Applications");
		standard = new DefaultMutableTreeNode("Standard Applications");
		// dbApps = new DefaultMutableTreeNode("DB (" + getDatabaseName() +
		// ")");

		// root.add(std);

		root.add(standard);

		databases = Database.getDatabases();
		if (commandLineConfig != null) {
			databases.add(new Database("Portal", commandLineConfig));
		}
		for (Iterator<Database> it = databases.iterator(); it.hasNext();) {
			Database database = it.next();
			DefaultMutableTreeNode dbNode = new DefaultMutableTreeNode(database);
			root.add(dbNode);
		}
		// root.add(dbApps);

		setTreeModel(new DefaultTreeModel(root) {

			private static final long serialVersionUID = 1L;

			public void valueForPathChanged(TreePath path, Object newValue) {
				MutableTreeNode aNode = (MutableTreeNode) path
				.getLastPathComponent();
				handleRename((DefaultMutableTreeNode) aNode, newValue);
				nodeChanged(aNode);

			}
		});

		getTreeModel().setAsksAllowsChildren(true);

		tree = new AutoScrollingJTree(getTreeModel());
		// tree = new JTree(new DataManagerTreeModel());
		// tree.getSelectionModel().setSelectionMode(TreeSelectionModel.SINGLE_TREE_SELECTION);
		tree.getSelectionModel().setSelectionMode(
				TreeSelectionModel.DISCONTIGUOUS_TREE_SELECTION);
		ParaProfTreeCellRenderer renderer = new ParaProfTreeCellRenderer();
		tree.setCellRenderer(renderer);
		tree.setEditable(true);
		FontMetrics fm = tree.getFontMetrics(ParaProf.preferences.getFont());
		tree.setRowHeight(fm.getHeight());

		// Add a mouse listener for this tree.
		MouseListener ml = new MouseAdapter() {
			public void mousePressed(MouseEvent evt) {
				try {

					if (TreeUI.rightClick(evt)) {
						int row = tree
						.getRowForLocation(evt.getX(), evt.getY());
						int rows[] = tree.getSelectionRows();
						boolean found = false;
						if (rows != null) {
							for (int i = 0; i < rows.length; i++) {
								if (row == rows[i]) {
									found = true;
									break;
								}
							}
						}
						if (!found) {
							tree.setSelectionRow(row);
						}
					}

					TreePath[] paths = tree.getSelectionPaths();
					if (paths == null) {
						return;
					}

					if (paths.length > 1) {
						clickedOnObject = paths;
						if (TreeUI.rightClick(evt)) {
							// TreePath path = paths[0];
							multiPopup.show(tree, evt.getX(), evt.getY());
						}
					}

					if (paths.length == 1) { // only one item is selected
						TreePath path = paths[0];
						DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path
						.getLastPathComponent();
						Object userObject = selectedNode.getUserObject();

						if (TreeUI.rightClick(evt)) {
							if (userObject instanceof ParaProfApplication) {
								clickedOnObject = userObject;
								if (((ParaProfApplication) userObject)
										.dBApplication()) {
									dbAppPopup.show(tree, evt.getX(),
											evt.getY());
								} else {
									stdAppPopup.show(tree, evt.getX(),
											evt.getY());
								}
							} else if (userObject instanceof ParaProfExperiment) {
								clickedOnObject = userObject;
								if (((ParaProfExperiment) userObject)
										.dBExperiment()) {
									dbExpPopup.show(tree, evt.getX(),
											evt.getY());
								} else {
									stdExpPopup.show(tree, evt.getX(),
											evt.getY());
								}

							} else if (userObject instanceof ParaProfTrial) {
								clickedOnObject = userObject;
								if (((ParaProfTrial) userObject).dBTrial()) {
									dbTrialPopup.show(tree, evt.getX(),
											evt.getY());
								} else {
									stdTrialPopup.show(tree, evt.getX(),
											evt.getY());
								}
							} else if (userObject instanceof ParaProfMetric) {
								clickedOnObject = userObject;
								metricPopup.show(tree, evt.getX(), evt.getY());
							} else if (userObject instanceof Database) {
								// standard or database
								clickedOnObject = selectedNode;
								Database db = (Database)userObject;
								DatabaseAPI dbapi = getDatabaseAPI(db);
								if(dbapi.db().getSchemaVersion()>0){
									db.setTAUdb(true);
								}else{db.setTAUdb(false);}
								if(db.isTAUdb()){
									TAUdbPopUp.show(tree, evt.getX(), evt.getY());
								}else{
									databasePopUp.show(tree, evt.getX(), evt.getY());
								}
							
							} else if (userObject instanceof String) {
								// standard or database
								clickedOnObject = selectedNode;
								if (((String) userObject).indexOf("Standard") != -1) {
									databasePopUp.show(tree, evt.getX(), evt.getY());
								}

							} else if (userObject instanceof View){
								clickedOnObject = selectedNode;
								ViewPopUp.show(tree,  evt.getX(), evt.getY());
							}
						} else {

							if (userObject instanceof ParaProfMetric) {
								ParaProfMetric ppMetric = (ParaProfMetric) userObject;
								if (evt.getClickCount() == 2) {
									showMetric(ppMetric);
									if (showApplyOperationItem.isSelected()) {
										derivedMetricPanel
										.removeMetric(ppMetric);
									}
								} else if (showApplyOperationItem.isSelected()) {
									derivedMetricPanel.insertMetric(ppMetric);
								}
							}
						}
					}
				} catch (Exception e) {
					ParaProfUtils.handleException(e);
				}
			}
		};
		tree.addMouseListener(ml);

		// Add tree listeners.
		tree.addTreeSelectionListener(this);
		tree.addTreeWillExpandListener(this);

		TreeDragSource ds = new TreeDragSource(tree,
				DnDConstants.ACTION_COPY_OR_MOVE);
		TreeDropTarget dropTarget = new TreeDropTarget(tree);
		// don't do this because it will cause the swing DnD to be activated
		// which will conflict with the AWT
		// tree.setDragEnabled(true);

		// Place it in a scroll pane
		treeScrollPane = new JScrollPane(tree);

		// Set up the split panes, and add to content pane.
		jSplitInnerPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT,
				treeScrollPane, getPanelHelpMessage(0));
		jSplitInnerPane.setContinuousLayout(true);
		jSplitInnerPane.setResizeWeight(0.5);

		this.getContentPane().add(jSplitInnerPane, "Center");

		jSplitInnerPane.setDividerLocation(0.5);

		DBConnector.setPasswordCallback(new PasswordCallback() {

			public String getPassword(ParseConfig config) {
				JPanel promptPanel = new JPanel();

				promptPanel.setLayout(new GridBagLayout());
				GridBagConstraints gbc = new GridBagConstraints();
				gbc.insets = new Insets(5, 5, 5, 5);

				JLabel label = new JLabel("<html>Enter password for user '"
						+ config.getDBUserName() + "'<br> Database: '"
						+ config.getDBName() + "' (" + config.getDBHost() + ":"
						+ config.getDBPort() + ")</html>");

				JPasswordField password = new JPasswordField(15);

				gbc.fill = GridBagConstraints.BOTH;
				gbc.anchor = GridBagConstraints.CENTER;
				gbc.weightx = 1;
				gbc.weighty = 1;
				Utility.addCompItem(promptPanel, label, gbc, 0, 0, 1, 1);
				gbc.fill = GridBagConstraints.HORIZONTAL;
				Utility.addCompItem(promptPanel, password, gbc, 1, 0, 1, 1);

				if (JOptionPane.showConfirmDialog(null, promptPanel,
						"Enter Password", JOptionPane.OK_CANCEL_OPTION) == JOptionPane.OK_OPTION) {
					return new String(password.getPassword());
				} else {
					return null;
				}
			}
		});

		ParaProf.incrementNumWindows();
	}

	protected void handleRename(DefaultMutableTreeNode aNode, Object newValue) {
		if (newValue instanceof String) {
			String name = (String) newValue;
			if (aNode.getUserObject() instanceof ParaProfApplication) {
				ParaProfApplication application = (ParaProfApplication) aNode
				.getUserObject();
				application.setName(name);

				if (application.dBApplication()) {
					DatabaseAPI databaseAPI = getDatabaseAPI(application
							.getDatabase());
					if (databaseAPI != null) {
						databaseAPI.saveApplication(application);
						databaseAPI.terminate();
					}
				}

			} else if (aNode.getUserObject() instanceof ParaProfExperiment) {
				ParaProfExperiment experiment = (ParaProfExperiment) aNode
				.getUserObject();
				experiment.setName(name);

				if (experiment.dBExperiment()) {
					DatabaseAPI databaseAPI = getDatabaseAPI(experiment
							.getDatabase());
					if (databaseAPI != null) {
						databaseAPI.saveExperiment(experiment);
						databaseAPI.terminate();
					}
				}

			} else if (aNode.getUserObject() instanceof ParaProfTrial) {
				ParaProfTrial ppTrial = (ParaProfTrial) aNode.getUserObject();
				
				if (ppTrial.dBTrial()) {
					DatabaseAPI databaseAPI = getDatabaseAPI(ppTrial
							.getDatabase());
					ppTrial.setDatabaseAPI(databaseAPI);
				}
				ppTrial.rename(name);

			} else if (aNode.getUserObject() instanceof ParaProfMetric) {
				ParaProfMetric metric = (ParaProfMetric) aNode.getUserObject();
				if (metric.dbMetric()) {
					DatabaseAPI databaseAPI = getDatabaseAPI(metric.getParaProfTrial()
							.getDatabase());
					metric.rename(databaseAPI.db(), name);
				}
				
			} else if (aNode.getUserObject() instanceof ParaProfView) {
				ParaProfView view = (ParaProfView) aNode.getUserObject();
				DatabaseAPI databaseAPI = getDatabaseAPI(view.getDatabase());
				DefaultMutableTreeNode DMT = view.getDMTN();

				view.rename(databaseAPI.db(), name);
				view.getDMTN();
				

			}

		}

	}

	void setupMenus() {
		JMenuBar mainMenu = new JMenuBar();

		JMenu fileMenu = new JMenu("File");

		JMenuItem menuItem = new JMenuItem("Open...");
		menuItem.addActionListener(this);
		fileMenu.add(menuItem);

		menuItem = new JMenuItem("Preferences...");
		menuItem.addActionListener(this);
		fileMenu.add(menuItem);

		menuItem = new JMenuItem("Database Configuration");
		menuItem.addActionListener(this);
		fileMenu.add(menuItem);

		menuItem = new JMenuItem("Close This Window");
		menuItem.addActionListener(this);
		fileMenu.add(menuItem);

		menuItem = new JMenuItem("Exit ParaProf!");
		menuItem.addActionListener(this);
		fileMenu.add(menuItem);

		// Options menu.
		JMenu optionsMenu = new JMenu("Options");

		showApplyOperationItem = new JCheckBoxMenuItem(
				"Show Derived Metric Panel", false);
		showApplyOperationItem.addActionListener(this);
		optionsMenu.add(showApplyOperationItem);

		JMenuItem loadExpressionFile = new JMenuItem("Apply Expression File");
		loadExpressionFile.addActionListener(this);
		optionsMenu.add(loadExpressionFile);

		JMenuItem applyExpressionFile = new JMenuItem(
		"Re-Apply Expression File");
		applyExpressionFile.addActionListener(this);
		optionsMenu.add(applyExpressionFile);

		// Help menu.
		JMenu helpMenu = new JMenu("Help");

		JMenuItem showHelpWindowItem = new JMenuItem("Show Help Window");
		showHelpWindowItem.addActionListener(this);
		helpMenu.add(showHelpWindowItem);

		JMenuItem aboutItem = new JMenuItem("About ParaProf");
		aboutItem.addActionListener(this);
		helpMenu.add(aboutItem);

		// Now, add all the menus to the main menu.
		mainMenu.add(fileMenu);
		mainMenu.add(optionsMenu);
		mainMenu.add(helpMenu);
		setJMenuBar(mainMenu);

		databasePopUp=TreeUI.getDatabasePopup(this);
		TAUdbPopUp=TreeUI.getTauDBPopUp(this);
		ViewPopUp=TreeUI.getViewPopUp(this);
		
		JMenuItem jMenuItem;
		
		jMenuItem = new JMenuItem("Monitor Application");
		jMenuItem.addActionListener(this);
		runtimePopup.add(jMenuItem);
		
		stdAppPopup = TreeUI.getStdAppPopUp(this);

		dbAppPopup = TreeUI.getDbAppPopUp(this);


		// Standard experiment popup
		jMenuItem = new JMenuItem("Upload Experiment to DB");
		jMenuItem.addActionListener(this);
		stdExpPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Add Trial");
		jMenuItem.addActionListener(this);
		stdExpPopup.add(jMenuItem);
		// jMenuItem = new JMenuItem("Add all trials to Comparison Window");
		// jMenuItem.addActionListener(this);
		// stdExpPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Delete");
		jMenuItem.addActionListener(this);
		stdExpPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Rename");
		jMenuItem.addActionListener(this);
		stdExpPopup.add(jMenuItem);

		// DB experiment popup
		jMenuItem = new JMenuItem("Add Trial");
		jMenuItem.addActionListener(this);
		dbExpPopup.add(jMenuItem);
		// jMenuItem = new JMenuItem("Add all trials to Comparison Window");
		// jMenuItem.addActionListener(this);
		// dbExpPopup.add(jMenuItem);

		jMenuItem = new JMenuItem("Delete");
		jMenuItem.addActionListener(this);
		dbExpPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Rename");
		jMenuItem.addActionListener(this);
		dbExpPopup.add(jMenuItem);

		// Standard trial popup
		jMenuItem = new JMenuItem("Export Profile");
		jMenuItem.addActionListener(this);
		stdTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Convert to Phase Profile");
		jMenuItem.addActionListener(this);
		stdTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Create Selective Instrumentation File");
		jMenuItem.addActionListener(this);
		stdTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Add Mean to Comparison Window");
		jMenuItem.addActionListener(this);
		stdTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Add Metadata Field");
		jMenuItem.addActionListener(this);
		stdTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Upload Trial to DB");
		jMenuItem.addActionListener(this);
		stdTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Delete");
		jMenuItem.addActionListener(this);
		stdTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Rename");
		jMenuItem.addActionListener(this);
		stdTrialPopup.add(jMenuItem);

		jMenuItem = new JMenuItem("Show metric in new window");
		jMenuItem.addActionListener(this);
		metricPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Show metric in all sub-windows");
		jMenuItem.addActionListener(this);
		metricPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Delete");
		jMenuItem.addActionListener(this);
		metricPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Rename");
		jMenuItem.addActionListener(this);
		metricPopup.add(jMenuItem);

		// DB trial popup
		jMenuItem = new JMenuItem("Export Profile");
		jMenuItem.addActionListener(this);
		dbTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Convert to Phase Profile");
		jMenuItem.addActionListener(this);
		dbTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Create Selective Instrumentation File");
		jMenuItem.addActionListener(this);
		dbTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Add Mean to Comparison Window");
		jMenuItem.addActionListener(this);
		dbTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Add Metadata Field");
		jMenuItem.addActionListener(this);
		dbTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Delete");
		jMenuItem.addActionListener(this);
		dbTrialPopup.add(jMenuItem);
		jMenuItem = new JMenuItem("Rename");
		jMenuItem.addActionListener(this);
		dbTrialPopup.add(jMenuItem);



		multiPopup = TreeUI.getMultiPopUp(this);
		
		

	}

	public void recomputeStats() {
		for (int i = 0; i < loadedTrials.size(); i++) {
			ParaProfTrial ppTrial = loadedTrials.get(i);
			ppTrial.getDataSource().generateDerivedData();
		}
	}

	public void handleDelete(Object object) throws SQLException,DatabaseException {
			handleDelete(object, true);
	}
	private void handleDelete(Object object, boolean ShowConfirmation) throws SQLException,DatabaseException {

		if (ShowConfirmation) {
			int confirm = JOptionPane.showConfirmDialog(tree,
					"Are you sure you want to permanently delete this item from the database and all views?",
					"Confirm Delete", JOptionPane.YES_NO_OPTION);

			if (confirm != 0) {
				return;
			}
		}
		
		
		if (object instanceof TreePath[]) {
			TreePath[] paths = (TreePath[]) object;
			ArrayList<ParaProfTrial> trials = new ArrayList<ParaProfTrial>(paths.length);
			for (int i = 0; i < paths.length; i++) {
				DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) paths[i]
				                                                                     .getLastPathComponent();
				Object userObject = selectedNode.getUserObject();
				//Check if there are multiple trials to delete, if so delete them all at once
								if (userObject instanceof ParaProfTrial) {
										ParaProfTrial t = (ParaProfTrial) userObject;
										if (t.dBTrial()
					
												&& (trials.size() == 0
												|| trials.get(0).getDatabase()
					.equals(t.getDatabase()))) {
											trials.add(t);
										} else {
										handleDelete(userObject, false);
										}
									} else {
										handleDelete(userObject, false);
									}
								}
								if (trials.size() > 0) {
									ParaProfTrial[] trialArray = new ParaProfTrial[trials.size()];
									trialArray = trials.toArray(trialArray);
									handleDelete(trialArray, false);
			}

		} else if (object instanceof ParaProfApplication) {
			ParaProfApplication application = (ParaProfApplication) object;
			if (application.dBApplication()) {

				DatabaseAPI databaseAPI = this.getDatabaseAPI(application
						.getDatabase());
				if (databaseAPI != null) {
					databaseAPI.deleteApplication(application.getID());
					databaseAPI.terminate();
					// Remove any loaded trials associated with this
					// application.
					for (Enumeration<ParaProfTrial> e = loadedDBTrials
							.elements(); e.hasMoreElements();) {
						ParaProfTrial loadedTrial = e.nextElement();
						if (loadedTrial.getApplicationID() == application
								.getID() && loadedTrial.loading() == false) {
							loadedDBTrials.remove(loadedTrial);
						}
					}
					if (application.getDMTN() != null)
						getTreeModel().removeNodeFromParent(
								application.getDMTN());
				}

			} else {
				ParaProf.applicationManager.removeApplication(application);
				getTreeModel().removeNodeFromParent(application.getDMTN());
			}
		} else if (object instanceof ParaProfExperiment) {
			ParaProfExperiment experiment = (ParaProfExperiment) object;
			if (experiment.dBExperiment()) {

				DatabaseAPI databaseAPI = this.getDatabaseAPI(experiment
						.getDatabase());
				if (databaseAPI != null) {
					databaseAPI.deleteExperiment(experiment.getID());
					databaseAPI.terminate();
					// Remove any loaded trials associated with this
					// application.
					for (Enumeration<ParaProfTrial> e = loadedDBTrials
							.elements(); e.hasMoreElements();) {
						ParaProfTrial loadedTrial = e.nextElement();
						if (loadedTrial.getApplicationID() == experiment
								.getApplicationID()
								&& loadedTrial.getExperimentID() == experiment
								.getID()
								&& loadedTrial.loading() == false) {
							loadedDBTrials.remove(loadedTrial);
						}
					}
					if (experiment.getDMTN() != null) {
						getTreeModel().removeNodeFromParent(
								experiment.getDMTN());
					}
				}
			} else {
				experiment.getApplication().removeExperiment(experiment);
				getTreeModel().removeNodeFromParent(experiment.getDMTN());
			}

		} else if (object instanceof ParaProfTrial) {
			ParaProfTrial ppTrial = (ParaProfTrial) object;
			if (ppTrial.dBTrial()) {

				DatabaseAPI databaseAPI = this.getDatabaseAPI(ppTrial
						.getDatabase());
				if (databaseAPI != null) {
					databaseAPI.deleteTrial(ppTrial.getID());
					databaseAPI.terminate();
					// Remove any loaded trials associated with this
					// application.
					for (Enumeration<ParaProfTrial> e = loadedDBTrials
							.elements(); e.hasMoreElements();) {
						ParaProfTrial loadedTrial = e.nextElement();
						if (loadedTrial.getApplicationID() == ppTrial
								.getApplicationID()
								&& loadedTrial.getExperimentID() == ppTrial
								.getID()
								&& loadedTrial.getID() == ppTrial.getID()
								&& loadedTrial.loading() == false) {
							loadedDBTrials.remove(loadedTrial);
						}
					}
					getTreeModel().removeNodeFromParent(ppTrial.getDMTN());
				}
			} else {
				ppTrial.getExperiment().removeTrial(ppTrial);
				getTreeModel().removeNodeFromParent(ppTrial.getDMTN());
			}
		} else if (object instanceof ParaProfTrial[]) {
						ParaProfTrial[] ppTrials = (ParaProfTrial[]) object;
			
							DatabaseAPI databaseAPI = this.getDatabaseAPI(ppTrials[0]
									.getDatabase());
							if (databaseAPI != null) {
							int[] trialIDs = new int[ppTrials.length];
							for (int i = 0; i < ppTrials.length; i++) {
							ParaProfTrial ppTrial = ppTrials[i];
								trialIDs[i] = ppTrial.getID();
								// Remove any loaded trials associated with this
								// application.
								for (Enumeration<ParaProfTrial> e = loadedDBTrials
										.elements(); e.hasMoreElements();) {
									ParaProfTrial loadedTrial = e.nextElement();
									if (loadedTrial.getApplicationID() == ppTrial
											.getApplicationID()
											&& loadedTrial.getExperimentID() == ppTrial
													.getID()
											&& loadedTrial.getID() == ppTrial.getID()
											&& loadedTrial.loading() == false) {
										loadedDBTrials.remove(loadedTrial);
									}
								}
								getTreeModel().removeNodeFromParent(ppTrial.getDMTN());
							}
							databaseAPI.deleteTrial(trialIDs);
							databaseAPI.terminate();
						}
			
			 		}
		
		else if (object instanceof ParaProfMetric) {
			ParaProfMetric ppMetric = (ParaProfMetric) object;
			deleteMetric(ppMetric);
		} else if (object instanceof DefaultMutableTreeNode){
			
			View view = (View) ((DefaultMutableTreeNode)object).getUserObject();
			deleteView(view);
		} else if (object instanceof View){
			
			View view = (View) object;
			deleteView(view);
		}
		
	}

	private boolean isLoaded(ParaProfTrial ppTrial) {
		boolean loaded = true;
		if (ppTrial.dBTrial()) {
			loaded = false;
			for (Enumeration<ParaProfTrial> e = loadedDBTrials.elements(); e
			.hasMoreElements();) {
				ParaProfTrial loadedTrial = e.nextElement();
				if ((ppTrial.getID() == loadedTrial.getID())
						&& (ppTrial.getExperimentID() == loadedTrial
								.getExperimentID())
								&& (ppTrial.getApplicationID() == loadedTrial
										.getApplicationID())) {
					loaded = true;
				}
			}
		}
		return loaded;
	}

	public void actionPerformed(ActionEvent evt) {
		try {
			Object EventSrc = evt.getSource();
			if (EventSrc instanceof JMenuItem) {
				String arg = evt.getActionCommand();
				if (arg.equals("Exit ParaProf!")) {
					setVisible(false);
					dispose();
					ParaProf.exitParaProf(0);
				} else if (arg.equals("Open...")) {
					ParaProfApplication application = addApplication(false,
							standard);
					if (application != null) {
						this.expandApplicationType(0, application.getID(),
								application);
						ParaProfExperiment experiment = addExperiment(false,
								application);
						if (experiment != null) {
							this.expandApplication(0, application, experiment);
							(new LoadTrialWindow(this, application, experiment,
									true, true)).setVisible(true);
						}
					}
				} else if (arg.equals("Preferences...")) {
					ParaProf.preferencesWindow.showPreferencesWindow(this);
				} else if (arg.equals("Close This Window")) {
					closeThisWindow();

				} else if (arg.equals("Database Configuration")) {
					(new DatabaseManagerWindow(this, ParaProf.jarLocation,
							ParaProf.schemaLocation)).setVisible(true);

				} else if (arg.equals("Show Derived Metric Panel")) {
					if (showApplyOperationItem.isSelected()) {

						this.getContentPane().removeAll();

						jSplitOuterPane = new JSplitPane(
								JSplitPane.VERTICAL_SPLIT, jSplitInnerPane,
								derivedMetricPanel);
						this.getContentPane().add(jSplitOuterPane, "Center");

						this.validate();
						jSplitOuterPane.setDividerLocation(0.75);

					} else {

						double dividerLocation = jSplitInnerPane
						.getDividerLocation();
						this.getContentPane().removeAll();

						jSplitInnerPane = new JSplitPane(
								JSplitPane.HORIZONTAL_SPLIT, treeScrollPane,
								getPanelHelpMessage(0));
						jSplitInnerPane.setContinuousLayout(true);

						this.getContentPane().add(jSplitInnerPane, "Center");

						this.validate();
						jSplitInnerPane.setDividerLocation(dividerLocation
								/ this.getWidth());

						jSplitOuterPane.setDividerLocation(1.00);
					}
				} else if (arg.equals("Apply Expression File")) {
					if (selectedObject == null) {
						JOptionPane
						.showMessageDialog(
								this,
								"Please select a trial, experiment or application.",
								"Warning", JOptionPane.WARNING_MESSAGE);
						return;
					} else if (!((selectedObject.getUserObject() instanceof ParaProfMetric)
							|| (selectedObject.getUserObject() instanceof ParaProfTrial)
							|| (selectedObject.getUserObject() instanceof ParaProfTrial) || (selectedObject
									.getUserObject() instanceof ParaProfApplication))) {
						JOptionPane
						.showMessageDialog(
								this,
								"Please select a trial, experiment or application.",
								"Warning", JOptionPane.WARNING_MESSAGE);
						return;

					}
					int returnVal = expressionFileC.showOpenDialog(this);
					if (returnVal == JFileChooser.APPROVE_OPTION) {
						derivedMetricPanel
						.applyExpressionFile(new LineNumberReader(
								new FileReader(expressionFileC
										.getSelectedFile())));
					}
				} else if (arg.equals("Re-Apply Expression File")) {
					if (selectedObject == null) {
						JOptionPane
						.showMessageDialog(
								this,
								"Please select a trial, experiment or application.",
								"Warning", JOptionPane.WARNING_MESSAGE);
						return;
					} else if (!((selectedObject.getUserObject() instanceof ParaProfMetric)
							|| (selectedObject.getUserObject() instanceof ParaProfTrial)
							|| (selectedObject.getUserObject() instanceof ParaProfTrial) || (selectedObject
									.getUserObject() instanceof ParaProfApplication))) {
						JOptionPane
						.showMessageDialog(
								this,
								"Please select a trial, experiment or application.",
								"Warning", JOptionPane.WARNING_MESSAGE);
						return;

					}
					if (expressionFileC.getSelectedFile() != null) {
						derivedMetricPanel
						.applyExpressionFile(new LineNumberReader(
								new FileReader(expressionFileC
										.getSelectedFile())));
					} else {
						int returnVal = expressionFileC.showOpenDialog(this);
						if (returnVal == JFileChooser.APPROVE_OPTION) {
							derivedMetricPanel
							.applyExpressionFile(new LineNumberReader(
									new FileReader(expressionFileC
											.getSelectedFile())));
						}
					}

				} else if (arg.equals("About ParaProf")) {
					ImageIcon icon = Utility
					.getImageIconResource("tau-medium.png");
					JOptionPane.showMessageDialog(this,
							ParaProf.getInfoString(), "About ParaProf",
							JOptionPane.INFORMATION_MESSAGE, icon);
				} else if (arg.equals("Show Help Window")) {
					ParaProf.getHelpWindow().setVisible(true);
					// Clear the window first.
					ParaProf.getHelpWindow().clearText();
					ParaProf.getHelpWindow().writeText(
					"This is ParaProf's manager window!");
					ParaProf.getHelpWindow().writeText("");
					ParaProf.getHelpWindow()
					.writeText(
							"This window allows you to manage all of ParaProf's data sources,"
							+ " including loading data from local files, or from a database."
							+ " We also support the generation of derived metrics. Please see the"
							+ " items below for more help.");
					ParaProf.getHelpWindow().writeText("");
					ParaProf.getHelpWindow().writeText("------------------");
					ParaProf.getHelpWindow().writeText("");

					ParaProf.getHelpWindow()
					.writeText(
							"1) Navigation:"
							+ " The window is split into two17 halves, the left side gives a tree representation"
							+ " of all data. The right side gives information about items clicked on in the left"
							+ " half. You can also update information in the right half by double clicking in"
							+ " the fields, and entering new data.  This automatically updates the left half."
							+ " Right-clicking on the tree nodes in the left half displays popup menus which"
							+ " allow you to add/delete applications, experiments, or trials.");
					ParaProf.getHelpWindow().writeText("");
					ParaProf.getHelpWindow()
					.writeText(
							"2) DB Configuration:"
							+ " By default, ParaProf looks in the .ParaProf home directory in your home"
							+ " directory for the database configuration file.  If that file is found, then"
							+ " you are done, and can just expand the DB Applications node.  If there was a"
							+ " problem finding the file, you can enter the location of the file by selecting"
							+ " File -> Database Configuration.  You can also override the configuration file"
							+ " password in the same manner.");
					ParaProf.getHelpWindow().writeText("");
					ParaProf.getHelpWindow()
					.writeText(
							"3) Deriving new metrics:"
							+ " By selecting Options -> Show Derived Metric Panel, you will display the apply"
							+ " operations window.  Clicking on the metrics of a trial will update the"
							+ " arguments to the selected operation.  Currently, you can only derive metrics"
							+ " from metric in the same trial (thus for example creating floating point"
							+ " operations per second by taking PAPI_FP_INS and dividing it by GET_TIME_OF_DAY)."
							+ " The 2nd argument is a user editable textbox and can be filled in with scalar "
							+ " values using the keyword 'val' (e.g. \"val 1.5\".");
					ParaProf.getHelpWindow().writeText("");
					ParaProf.getHelpWindow().writeText("------------------");
					ParaProf.getHelpWindow().writeText("");
				} else if (arg.equals("Delete")) {
					handleDelete(clickedOnObject);
				} else if (arg.equals("Rename")) {
					if (clickedOnObject instanceof ParaProfApplication) {
						tree.startEditingAtPath(new TreePath(
								((ParaProfApplication) clickedOnObject).getDMTN()
										.getPath()));
					} else if (clickedOnObject instanceof ParaProfExperiment) {
						tree.startEditingAtPath(new TreePath(
								((ParaProfExperiment) clickedOnObject).getDMTN()
										.getPath()));
					} else if (clickedOnObject instanceof ParaProfTrial) {
						tree.startEditingAtPath(new TreePath(
								((ParaProfTrial) clickedOnObject).getDMTN()
										.getPath()));
					} else if (clickedOnObject instanceof ParaProfView) {
						tree.startEditingAtPath(new TreePath(
								((ParaProfView) clickedOnObject).getDMTN()
										.getPath()));
					} else if (clickedOnObject instanceof ParaProfMetric) {
						tree.startEditingAtPath(new TreePath(
								((ParaProfMetric) clickedOnObject).getDMTN()
										.getPath()));
					}
				} else if (arg.equals("Add Application")) {
					if (clickedOnObject == standard) {
						ParaProfApplication application = addApplication(false,
								standard);
						this.expandApplicationType(0, application.getID(),
								application);
					} else {
						ParaProfApplication application = addApplication(true,
								(DefaultMutableTreeNode) clickedOnObject);
						this.expandApplicationType(2, application.getID(),
								application);
					}
				} else if (arg.equals("Add Experiment")) {
					if (clickedOnObject == standard) {
						ParaProfApplication application = addApplication(false,
								standard);
						ParaProfExperiment experiment = addExperiment(false,
								application);
						if (application != null || experiment != null) {
							this.expandApplicationType(0, application.getID(),
									application);
							this.expandApplication(0, application, experiment);
						}
					} else if (clickedOnObject instanceof DefaultMutableTreeNode) {
						ParaProfApplication application = addApplication(true,
								(DefaultMutableTreeNode) clickedOnObject);
						ParaProfExperiment experiment = addExperiment(true,
								application);
						if (application != null || experiment != null) {
							this.expandApplicationType(2, application.getID(),
									application);
							this.expandApplication(2, application, experiment);
						}
					} else if (clickedOnObject instanceof ParaProfApplication) {
						ParaProfApplication application = (ParaProfApplication) clickedOnObject;
						if (application.dBApplication()) {
							ParaProfExperiment experiment = addExperiment(true,
									application);
							if (experiment != null)
								this.expandApplication(2, application,
										experiment);
						} else {
							ParaProfExperiment experiment = addExperiment(
									false, application);
							if (experiment != null)
								this.expandApplication(0, application,
										experiment);
						}
					}
				} else if (arg.equals("Add Trial")) {
					if (clickedOnObject == standard) {
						ParaProfApplication application = addApplication(false,
								standard);
						if (application != null) {
							this.expandApplicationType(0, application.getID(),
									application);
							ParaProfExperiment experiment = addExperiment(
									false, application);
							if (experiment != null) {
								this.expandApplication(0, application,
										experiment);
								(new LoadTrialWindow(this, application,
										experiment, true, true))
										.setVisible(true);
							}
						}
					} else if (clickedOnObject instanceof ParaProfApplication) {
						ParaProfApplication application = (ParaProfApplication) clickedOnObject;
						if (application.dBApplication()) {
							ParaProfExperiment experiment = addExperiment(true,
									application);
							if (experiment != null) {
								this.expandApplication(2, application,
										experiment);
								(new LoadTrialWindow(this, null, experiment,
										false, true)).setVisible(true);
							}
						} else {
							ParaProfExperiment experiment = addExperiment(
									false, application);
							if (experiment != null) {
								this.expandApplication(0, application,
										experiment);
								(new LoadTrialWindow(this, null, experiment,
										false, true)).setVisible(true);
							}
						}
					} else if (clickedOnObject instanceof ParaProfExperiment) {
						ParaProfExperiment experiment = (ParaProfExperiment) clickedOnObject;
						(new LoadTrialWindow(this, null, experiment, false,
								false)).setVisible(true);
					} else if (clickedOnObject instanceof DefaultMutableTreeNode){
						DefaultMutableTreeNode node = (DefaultMutableTreeNode) clickedOnObject;
						Database database = (Database) node.getUserObject();
						DatabaseAPI dbapi = getDatabaseAPI(database);
						if(dbapi.db().getSchemaVersion()>0){
							database.setTAUdb(true);
						}else{database.setTAUdb(false);}
						
						if (!database.isTAUdb()) {
							// a database
							ParaProfApplication application = addApplication(
									true,
									(DefaultMutableTreeNode) clickedOnObject);
							if (application != null) {
								this.expandApplicationType(2,
										application.getID(), application);
								ParaProfExperiment experiment = addExperiment(
										true, application);
								if (experiment != null) {
									this.expandApplication(2, application,
											experiment);
									(new LoadTrialWindow(this, application,
											experiment, true, true))
											.setVisible(true);
								}
							}
						}else{
//							new TAUdbLoadTrialWindow(this).setVisible(true);
							TAUdbDatabaseAPI api = (TAUdbDatabaseAPI) getDatabaseAPI(database);
							
							
							
							
							new LoadTrialWindow(this, api.getView(1)).setVisible(true);
							
						}

					}
					
					
					
					
				}else if (arg.equals("Add View")) {

					Database database = (Database) ((DefaultMutableTreeNode) clickedOnObject)
							.getUserObject();
					DatabaseAPI dbAPI = this.getDatabaseAPI(database);
					if(dbAPI instanceof TAUdbDatabaseAPI){
						ViewCreatorGUI frame = new ViewCreatorGUI((TAUdbDatabaseAPI) dbAPI);

						// Display the window.
						frame.pack();
						frame.setVisible(true);

						}else{
							System.out.println("Error: Cannot create a view on a non TAUdb database.");
						}
					
				}else if (arg.equals("Add Sub-View")) {
					View view = (View) ((DefaultMutableTreeNode) clickedOnObject)
							.getUserObject();

					Database database = view.getDatabase();
					DatabaseAPI dbAPI = this.getDatabaseAPI(database);
					if(dbAPI instanceof TAUdbDatabaseAPI){
					ViewCreatorGUI frame = new ViewCreatorGUI((TAUdbDatabaseAPI) dbAPI, view.getID());

					// Display the window.
					frame.pack();
					frame.setVisible(true);

					}else{
						System.out.println("Error: Cannot create a view on a non TAUdb database.");
					}

				} else if (arg.equals("Add Metadata Field To All Trials")) {
					View view = (View) ((DefaultMutableTreeNode) clickedOnObject)
							.getUserObject();
					Database database = view.getDatabase();
					DatabaseAPI dbAPI = this.getDatabaseAPI(database);
					if (dbAPI instanceof TAUdbDatabaseAPI) {

						String[] results = promptForMetadata();
						if (results == null) {
							return;
						}

						ArrayList<View> views = new ArrayList<View>(1);
						views.add(view);
						DB db = dbAPI.db();
						List<Trial> trials = View.getTrialsForView(views, true,
								db);
						if (trials == null) {
							return;
						}
						Iterator<Trial> it = trials.iterator();
						while (it.hasNext()) {
							Trial t = it.next();
							t.loadXMLMetadata(db);
							MetaDataMap mdm = t.getMetaData();

							String currentValue = mdm.get(results[0]);

							mdm.put(results[0], results[1]);
							int id = t.getID();
							DB tdb = this.getDatabaseAPI(t.getDatabase()).db();

							if (currentValue == null) {
								TAUdbTrial.addToPrimaryMetadataField(tdb, id,
									results[0], results[1]);
							} else if (!currentValue.equals(results[1])) {
								TAUdbTrial.updatePrimaryMetadataField(tdb, id,
										results[0], results[1]);
							}

						}


					} else {
						System.out
								.println("Error: Cannot add metadata for all trials in a view on a non TAUdb database.");
					}

				} else if (arg.equals("Remove Metadata Field From All Trials")) {
					View view = (View) ((DefaultMutableTreeNode) clickedOnObject)
							.getUserObject();
					Database database = view.getDatabase();
					DatabaseAPI dbAPI = this.getDatabaseAPI(database);
					if (dbAPI instanceof TAUdbDatabaseAPI) {

						String result = promptForMetadataRemoval();
						if (result == null) {
							return;
						}

						ArrayList<View> views = new ArrayList<View>(1);
						views.add(view);
						DB db = dbAPI.db();
						List<Trial> trials = View.getTrialsForView(views, true,
								db);
						if (trials == null) {
							return;
						}
						Iterator<Trial> it = trials.iterator();
						while (it.hasNext()) {
							Trial t = it.next();
							t.loadXMLMetadata(db);
							MetaDataMap mdm = t.getMetaData();

							// String currentValue = mdm.get(results[0]);

							// mdm.put(results[0], results[1]);
							mdm.remove(result);
							int id = t.getID();
							DB tdb = this.getDatabaseAPI(t.getDatabase()).db();

							// if (currentValue != null) {
							TAUdbTrial.removeFromPrimaryMetadataField(tdb, id,
									result);
							// }

						}

					} else {
						System.out
								.println("Error: Cannot remove metadata for all trials in a view on a non TAUdb database.");
					}

				} else if (arg.equals("Edit")) {
					View view = (View) ((DefaultMutableTreeNode) clickedOnObject)
							.getUserObject();

					Database database = view.getDatabase();
					DatabaseAPI dbAPI = this.getDatabaseAPI(database);
					if (dbAPI instanceof TAUdbDatabaseAPI) {
						ViewCreatorGUI frame = new ViewCreatorGUI(
								(TAUdbDatabaseAPI) dbAPI, view);

						// Display the window.
						frame.pack();
						frame.setVisible(true);

					} else {
						System.out
								.println("Error: Cannot edit a view on a non TAUdb database.");
					}
				} else if (arg.equals("Upload Application to DB")) {

					java.lang.Thread thread = new java.lang.Thread(
							new Runnable() {
								public void run() {
									try {
										ParaProfApplication clickedOnApp = (ParaProfApplication) clickedOnObject;
										uploadApplication(clickedOnApp, true,
												true);
									} catch (final Exception e) {
										EventQueue.invokeLater(new Runnable() {
											public void run() {
												ParaProfUtils
												.handleException(e);
											}
										});
									}
								}
							});
					thread.start();
				} else if (arg.equals("Export Application to Filesystem")) {

					ParaProfApplication dbApp = (ParaProfApplication) clickedOnObject;

					DatabaseAPI databaseAPI = this.getDatabaseAPI(dbApp
							.getDatabase());
					if (databaseAPI != null) {

						// boolean found = false;
						// ListIterator l =
						// databaseAPI.getApplicationList().listIterator();
						// while (l.hasNext()) {
						// ParaProfApplication dbApp = new
						// ParaProfApplication((Application) l.next());

						String appname = dbApp.getName().replace('/', '%');
						// boolean success =
						new File(appname).mkdir();

						databaseAPI.setApplication(dbApp);
						for (Iterator<Experiment> it = databaseAPI
								.getExperimentList().iterator(); it.hasNext();) {
							ParaProfExperiment dbExp = new ParaProfExperiment(
									it.next());

							String expname = appname + File.separator
							+ dbExp.getName().replace('/', '%');
							// success = (
							new File(expname).mkdir();

							databaseAPI.setExperiment(dbExp);
							for (Iterator<Trial> it2 = databaseAPI
									.getTrialList(true).iterator(); it2
									.hasNext();) {
								Trial trial = it2.next();

								databaseAPI.setTrial(trial.getID(), true);// TODO:
								// Do
								// these
								// really
								// require
								// xml
								// metadata?
								DBDataSource dbDataSource = new DBDataSource(
										databaseAPI);
								dbDataSource.load();

								String filename = expname + File.separator
								+ trial.getName().replace('/', '%')
								+ ".ppk";

								DataSourceExport.writePacked(dbDataSource,
										new File(filename));

							}

						}

						// }
					}

				} else if (arg.equals("Upload Experiment to DB")) {
					java.lang.Thread thread = new java.lang.Thread(
							new Runnable() {
								public void run() {
									try {
										ParaProfExperiment clickedOnExp = (ParaProfExperiment) clickedOnObject;
										uploadExperiment(null, clickedOnExp,
												true, true);
									} catch (final Exception e) {
										EventQueue.invokeLater(new Runnable() {
											public void run() {
												ParaProfUtils
												.handleException(e);
											}
										});
									}
								}
							});
					thread.start();

				} else if (arg.equals("Upload Trial to DB")) {
					java.lang.Thread thread = new java.lang.Thread(
							new Runnable() {
								public void run() {
									try {
										uploadTrial(null,
												(ParaProfTrial) clickedOnObject);
									} catch (final Exception e) {
										EventQueue.invokeLater(new Runnable() {
											public void run() {
												ParaProfUtils
												.handleException(e);
											}
										});
									}
								}
							});
					thread.start();

				} else if (arg.equals("Add all trials to Comparison Window")) {
					ParaProfExperiment clickedOnExp = (ParaProfExperiment) clickedOnObject;
					compareAllTrials(clickedOnExp);

				} else if (arg.equals("Add Mean to Comparison Window")) {
					ParaProfTrial ppTrial = (ParaProfTrial) clickedOnObject;
					addMeanToComparisonWindow(ppTrial);

				} else if (arg.equals("Add Metadata Field")) {
					ParaProfTrial ppTrial = (ParaProfTrial) clickedOnObject;
					addMetadataToTrial(ppTrial);
				} else if (arg.equals("Export Profile")) {
					ParaProfTrial ppTrial = (ParaProfTrial) clickedOnObject;
					if (ppTrial.loading()) {
						JOptionPane.showMessageDialog(this,
						"Cannot export trial while loading");
						return;
					}
					if (!isLoaded(ppTrial)) {
						JOptionPane
						.showMessageDialog(this,
								"Please load the trial before exporting (expand the tree)");
						return;
					}

					ParaProfUtils.exportTrial(ppTrial, this);

				} else if (arg.equals("Convert to Phase Profile")) {
					ParaProfTrial ppTrial = (ParaProfTrial) clickedOnObject;
					if (ppTrial.loading()) {
						JOptionPane.showMessageDialog(this,
						"Cannot convert while loading");
						return;
					}
					if (!isLoaded(ppTrial)) {
						JOptionPane
						.showMessageDialog(this,
								"Please load the trial before converting (expand the tree)");
						return;
					}

					ParaProfUtils.phaseConvertTrial(ppTrial, this);

				} else if (arg.equals("Create Selective Instrumentation File")) {
					ParaProfTrial ppTrial = (ParaProfTrial) clickedOnObject;
					if (ppTrial.loading()) {
						JOptionPane.showMessageDialog(this,
						"Cannot convert while loading");
						return;
					}
					if (!isLoaded(ppTrial)) {
						JOptionPane
						.showMessageDialog(this,
								"Please load the trial before converting (expand the tree)");
						return;
					}

					SelectiveFileGenerator.showWindow(ppTrial, this);
				} else if (arg.equals("Show metric in new window")) {
					ParaProfMetric ppMetric = (ParaProfMetric) clickedOnObject;
					showMetric(ppMetric);
				} else if (arg.equals("Show metric in all sub-windows")) {
					ParaProfMetric ppMetric = (ParaProfMetric) clickedOnObject;
					switchToMetric(ppMetric);
				}
			}
		} catch (Exception e) {
			ParaProfUtils.handleException(e);
		}
	}

	private void deleteView(View view)  {
		Database database = view.getDatabase();
		DatabaseAPI dbAPI = this.getDatabaseAPI(database);
	
		try {
			View.deleteView(view.getID(), dbAPI.getDb());
			getTreeModel().removeNodeFromParent(view.getDMTN());		
		} catch (SQLException e) {
			ParaProfUtils.handleException(e);
		}
	}

	private void addMetadataToTrial(ParaProfTrial ppTrial) {

		if (ppTrial.loading()) {
			JOptionPane.showMessageDialog(this,
					"Cannot perform operation while loading");
			return;
		}

		boolean loaded = true;

		if (ppTrial.dBTrial()) {
			loaded = false;
			for (Enumeration<ParaProfTrial> e = loadedDBTrials.elements(); e
					.hasMoreElements();) {
				ParaProfTrial loadedTrial = e.nextElement();
				if ((ppTrial.getID() == loadedTrial.getID())
						&& (ppTrial.getExperimentID() == loadedTrial
								.getExperimentID())
						&& (ppTrial.getApplicationID() == loadedTrial
								.getApplicationID())) {
					ppTrial = loadedTrial;
					loaded = true;
				}
			}
		}

		if (!loaded) {
			JOptionPane.showMessageDialog(this,
					"Please load the trial first (expand the tree)");
			return;
		}

		String[] fields = promptForMetadata();
		if (fields != null) {
			String name = fields[0];
			String value = fields[1];

			if (ppTrial.getTrial().getMetaData().get(name) != null) {
				JOptionPane.showMessageDialog(null, "Field Already Exists",
						"Please input a unique string for the name.",
						JOptionPane.ERROR_MESSAGE);
			} else {
				ppTrial.getTrial().getMetaData().put(name, value);

				DB db = getDatabaseAPI(ppTrial.getDatabase()).db();
				if ((ppTrial.getDatabase() != null && ppTrial.getDatabase()
						.isTAUdb())
						|| db.getSchemaVersion() > 0) {
					int id = ppTrial.getID();
					TAUdbTrial.addToPrimaryMetadataField(db, id, name, value);
				}

				int location = jSplitInnerPane.getDividerLocation();
				jSplitInnerPane.setRightComponent(getTable(ppTrial));
				jSplitInnerPane.setDividerLocation(location);

				// System.out.println("name: " + name.getText());
				// System.out.println("value: " + value.getText());

			}

		}
	}

	private static String[] promptForMetadata() {
		JTextField nameF = new JTextField(15);
		JTextField valueF = new JTextField(15);
		JPanel qpanel = new JPanel();
		qpanel.add(new JLabel("Name:"));
		qpanel.add(nameF);
		qpanel.add(Box.createHorizontalStrut(15)); // a spacer
		qpanel.add(new JLabel("Value:"));
		qpanel.add(valueF);
		int result = JOptionPane.showConfirmDialog(null, qpanel,
				"Input name and value for new metadata field",
				JOptionPane.OK_CANCEL_OPTION);
		if (result == JOptionPane.OK_OPTION) {
			String[] results = { nameF.getText(), valueF.getText() };
			if (results[0] == null || results[1] == null
					|| results[0].trim().length() == 0) {
				JOptionPane
						.showMessageDialog(
								null,
								"Invalid Input",
								"Please input a valid string for the name and value entry",
								JOptionPane.ERROR_MESSAGE);
				return null;
			} else
				return results;
		}
		return null;
	}

	private static String promptForMetadataRemoval() {
		JTextField nameF = new JTextField(15);
		JPanel qpanel = new JPanel();
		qpanel.add(new JLabel("Name:"));
		qpanel.add(nameF);
		int resultVal = JOptionPane.showConfirmDialog(null, qpanel,
				"Input name of Metadata Field to Remove",
				JOptionPane.OK_CANCEL_OPTION);
		if (resultVal == JOptionPane.OK_OPTION) {
			String result = nameF.getText();
			if (result == null || result.trim().length() == 0) {
				JOptionPane.showMessageDialog(null, "Invalid Input",
						"Please input a valid string for the name",
						JOptionPane.ERROR_MESSAGE);
				return null;
			} else
				return result;
		}
		return null;
	}

	private void addMeanToComparisonWindow(ParaProfTrial ppTrial) {
		if (ppTrial.loading()) {
			JOptionPane.showMessageDialog(this,
			"Cannot perform operation while loading");
		} else {
			boolean loaded = true;

			if (ppTrial.dBTrial()) {
				loaded = false;
				for (Enumeration<ParaProfTrial> e = loadedDBTrials.elements(); e
				.hasMoreElements();) {
					ParaProfTrial loadedTrial = e.nextElement();
					if ((ppTrial.getID() == loadedTrial.getID())
							&& (ppTrial.getExperimentID() == loadedTrial
									.getExperimentID())
									&& (ppTrial.getApplicationID() == loadedTrial
											.getApplicationID())) {
						ppTrial = loadedTrial;
						loaded = true;
					}
				}
			}

			if (!loaded) {
				JOptionPane.showMessageDialog(this,
						"Please load the trial first (expand the tree)");
			} else {
				if (ParaProf.theComparisonWindow == null) {
					ParaProf.theComparisonWindow = FunctionBarChartWindow
					.CreateComparisonWindow(ppTrial, ppTrial
							.getDataSource().getMeanData(), this);
				} else {
					ParaProf.theComparisonWindow.addThread(ppTrial, ppTrial
							.getDataSource().getMeanData());
				}
				ParaProf.theComparisonWindow.setVisible(true);
			}
		}
	}

	private void compareAllTrials(ParaProfExperiment ppExp) {
		// for (Iterator it = ppExp.getTrialList(); it.hasNext();) {
		// ParaProfTrial ppTrial = (ParaProfTrial) it.next();
		// System.out.println(ppTrial);
		// //addMeanToComparisonWindow(ppTrial);
		// }
		RegressionGraph chart = RegressionGraph.createBasicChart(ppExp
				.getTrials());
		Frame frame = chart.createFrame();
		// chart.savePNG("/home/amorris/foo.png");
		frame.setVisible(true);
	}

	private ParaProfApplication uploadApplication(ParaProfApplication ppApp,
			boolean allowOverwrite, boolean uploadChildren)
	throws SQLException, DatabaseException {

		DatabaseAPI databaseAPI = this.getDatabaseAPI(null);
		if (databaseAPI != null) {

			boolean found = false;
			ListIterator<Application> l = databaseAPI.getApplicationList()
			.listIterator();

			while (l.hasNext()) {
				ParaProfApplication dbApp = new ParaProfApplication(l.next());

				if (dbApp.getName().equals(ppApp.getName())) {
					found = true;

					if (allowOverwrite) {
						String options[] = { "Overwrite", "Don't overwrite",
						"Cancel" };
						int value = JOptionPane
						.showOptionDialog(
								this,
								"An Application with the name \""
								+ ppApp.getName()
								+ "\" already exists, it will be updated new experiments/trials, should the metadata be overwritten?",
								"Upload Application",
								JOptionPane.YES_NO_OPTION, // Need
								// something
								JOptionPane.QUESTION_MESSAGE, null, // Use
								// default
								// icon
								// for
								// message
								// type
								options, options[1]);
						if (value == JOptionPane.CLOSED_OPTION
								|| value == JOptionPane.CANCEL_OPTION) {
							return null;
						} else {
							if (value == 0) {
								// overwrite the metadata
								dbApp.setFields(ppApp.getFields());
								databaseAPI.saveApplication(dbApp);
							}

							if (uploadChildren) {
								for (Iterator<ParaProfExperiment> it2 = ppApp
										.getExperimentList(); it2.hasNext();) {
									ParaProfExperiment ppExp = it2.next();
									uploadExperiment(dbApp, ppExp, true, true);
								}
							}
						}
					}
					dbApp.setDatabase(getDefaultDatabase());
					return dbApp;
				}
			}

			if (!found) {
				Application newApp = new Application(ppApp);
				newApp.setID(-1); // must set the ID to -1 to indicate that this
				// is a new application (bug found by Sameer
				// on 2005-04-19)
				ParaProfApplication application = new ParaProfApplication(
						newApp);
				application.setDBApplication(true);
				application.setID(databaseAPI.saveApplication(application));

				if (uploadChildren) {
					for (Iterator<ParaProfExperiment> it2 = ppApp
							.getExperimentList(); it2.hasNext();) {
						ParaProfExperiment ppExp = it2.next();
						uploadExperiment(application, ppExp, true, true);
					}
				}

				return application;
			}
		}
		return null;
	}

	private ParaProfExperiment uploadExperiment(ParaProfApplication dbApp,
			ParaProfExperiment ppExp, boolean allowOverwrite,
			boolean uploadChildren) throws SQLException, DatabaseException {
		DatabaseAPI databaseAPI = this.getDatabaseAPI(null);
		if (databaseAPI == null)
			return null;

		if (dbApp == null) {
			dbApp = uploadApplication(ppExp.getApplication(), false, false);
		}

		// we now have a ParaProfApplication (dbApp) that is in the database

		boolean found = false;
		databaseAPI.setApplication(dbApp);
		ListIterator<Experiment> l = databaseAPI.getExperimentList()
		.listIterator();
		while (l.hasNext()) {
			ParaProfExperiment dbExp = new ParaProfExperiment(l.next());

			if (dbExp.getName().equals(ppExp.getName())) {
				found = true;

				if (allowOverwrite) {

					String options[] = { "Overwrite", "Don't overwrite",
					"Cancel" };
					int value = JOptionPane
					.showOptionDialog(
							this,
							"An Experiment with the name \""
							+ ppExp.getName()
							+ "\" already exists, it will be updated new trials, should the metadata be overwritten?",
							"Upload Application",
							JOptionPane.YES_NO_OPTION, // Need something
							JOptionPane.QUESTION_MESSAGE, null, // Use
							// default
							// icon
							// for
							// message
							// type
							options, options[1]);
					if (value == JOptionPane.CLOSED_OPTION || value == 2) {
						return null;
					} else {
						if (value == 0) {
							// overwrite the metadata
							dbExp.setFields(ppExp.getFields());
							databaseAPI.saveExperiment(dbExp);
						}

						if (uploadChildren) {
							for (Iterator<ParaProfTrial> it2 = ppExp
									.getTrialList(); it2.hasNext();) {
								ParaProfTrial ppTrial = it2.next();
								uploadTrial(dbExp, ppTrial);
							}
						}
					}
				}
				dbExp.setApplication(dbApp);
				return dbExp;
			}
		}

		if (!found) {
			Experiment newExp = new Experiment(ppExp);
			ParaProfExperiment experiment = new ParaProfExperiment(newExp);
			newExp.setID(-1); // must set the ID to -1 to indicate that this is
			// a new application (bug found by Sameer on
			// 2005-04-19)
			experiment.setDBExperiment(true);
			experiment.setApplicationID(dbApp.getID());
			experiment.setApplication(dbApp);
			experiment.setID(databaseAPI.saveExperiment(experiment));

			if (uploadChildren) {
				for (Iterator<ParaProfTrial> it2 = ppExp.getTrialList(); it2
				.hasNext();) {
					ParaProfTrial ppTrial = it2.next();
					uploadTrial(experiment, ppTrial);
				}
			}

			return experiment;
		}
		return null;
	}

	public ParaProfTrial uploadTrial(ParaProfExperiment dbExp,
			ParaProfTrial ppTrial) throws SQLException, DatabaseException {
		DatabaseAPI databaseAPI = this.getDatabaseAPI(null);
		if (databaseAPI == null)
			return null;

		if (dbExp == null) {
			dbExp = uploadExperiment(null, ppTrial.getExperiment(), false,
					false);
		}

		ParaProfTrial dbTrial = new ParaProfTrial(ppTrial.getTrial());
		dbTrial.setID(-1);
		dbTrial.setExperimentID(dbExp.getID());
		dbTrial.setApplicationID(dbExp.getApplicationID());
		dbTrial.getTrial().setDataSource(ppTrial.getDataSource());
		dbTrial.setExperiment(dbExp);

		dbTrial.setUpload(true); // This trial is not set to a db trial until
		// after it has finished loading.

		LoadTrialProgressWindow lpw = new LoadTrialProgressWindow(this,
				dbTrial.getDataSource(), dbTrial, true);
		lpw.setVisible(true);

		lpw.waitForLoad();

		// we now have a ParaProfExperiment (dbExp) that is in the database
		return dbTrial;

	}

	public void valueChanged(TreeSelectionEvent event) {
		try {
			TreePath path = tree.getSelectionPath();
			if (path == null)
				return;
			DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path
			.getLastPathComponent();
			DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode
			.getParent();
			Object userObject = selectedNode.getUserObject();
			selectedObject = selectedNode;

			int location = jSplitInnerPane.getDividerLocation();
			if (selectedNode.isRoot()) {
				jSplitInnerPane.setRightComponent(getPanelHelpMessage(0));
			} else if ((parentNode.isRoot())) {
				if (userObject.toString().equals("Standard Applications")) {
					jSplitInnerPane.setRightComponent(getPanelHelpMessage(1));
				} else if (userObject.toString().equals("Runtime Applications")) {
					jSplitInnerPane.setRightComponent(getPanelHelpMessage(2));
				} else if (userObject.toString().equals("DB Applications")) {
					jSplitInnerPane.setRightComponent(getPanelHelpMessage(3));
				}
			} else if (userObject instanceof ParaProfApplication) {
				jSplitInnerPane.setRightComponent(getTable(userObject));
			} else if (userObject instanceof ParaProfExperiment) {
				jSplitInnerPane.setRightComponent(getTable(userObject));
			} else if (userObject instanceof ParaProfTrial) {
				jSplitInnerPane.setRightComponent(getTable(userObject));
			} else if (userObject instanceof ParaProfMetric) {
				this.metricSelected((ParaProfMetric) userObject, false);
			}

			jSplitInnerPane.setDividerLocation(location);

		} catch (Exception e) {
			ParaProfUtils.handleException(e);
		}
	}

	private void clearDefaultMutableTreeNodes(
			DefaultMutableTreeNode defaultMutableTreeNode) {
		int count = defaultMutableTreeNode.getChildCount();
		for (int i = 0; i < count; i++) {
			DefaultMutableTreeNode child = (DefaultMutableTreeNode) defaultMutableTreeNode
			.getChildAt(i);
			((ParaProfTreeNodeUserObject) child.getUserObject())
			.clearDefaultMutableTreeNode();
			clearDefaultMutableTreeNodes((DefaultMutableTreeNode) defaultMutableTreeNode
					.getChildAt(i));
		}
	}

	public void treeWillCollapse(TreeExpansionEvent event) {
		try {
			TreePath path = event.getPath();
			if (path == null) {
				return;
			}
			DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path
			.getLastPathComponent();
			DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode
			.getParent();
			Object userObject = selectedNode.getUserObject();

			if (selectedNode.isRoot()) {
				clearDefaultMutableTreeNodes(standard);
				// clearDefaultMutableTreeNodes(runtime);
				// clearDefaultMutableTreeNodes(dbApps);
			} else if (parentNode.isRoot()) {
				clearDefaultMutableTreeNodes(selectedNode);
			} else if (userObject instanceof ParaProfTreeNodeUserObject) {
				this.clearDefaultMutableTreeNodes(selectedNode);
			}
		} catch (Exception e) {
			ParaProfUtils.handleException(e);
		}
	}

	public void treeWillExpandNEW(TreeExpansionEvent event) {
		TreePath path = event.getPath();
		DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path
		.getLastPathComponent();

		if (selectedNode.isRoot()) {
			return;
		}

	}

	public void expand(DefaultMutableTreeNode node) {
		try {
			TreePath t = new TreePath(node);
			// TreeExpansionEvent event =
			new TreeExpansionEvent(this.tree, t);
			DefaultMutableTreeNode selectedNode = node;
			Object userObject = node.getUserObject();
			if (userObject instanceof ParaProfApplication) {
				ParaProfApplication application = (ParaProfApplication) userObject;
				if (application.dBApplication()) {
					// remove the old experiments
					for (int i = selectedNode.getChildCount(); i > 0; i--) {
						getTreeModel().removeNodeFromParent(
								((DefaultMutableTreeNode) selectedNode
										.getChildAt(i - 1)));
					}

					// reload from the database
					DatabaseAPI databaseAPI = this.getDatabaseAPI(application
							.getDatabase());
					if (databaseAPI != null) {
						databaseAPI.setApplication(application.getID());
						ListIterator<Experiment> l = databaseAPI
						.getExperimentList().listIterator();
						while (l.hasNext()) {
							ParaProfExperiment experiment = new ParaProfExperiment(
									(Experiment) l.next());
							experiment.setDBExperiment(true);
							experiment.setApplication(application);
							DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(
									experiment);
							experiment.setDMTN(experimentNode);
							getTreeModel().insertNodeInto(experimentNode,
									selectedNode, selectedNode.getChildCount());
						}
						databaseAPI.terminate();
					}

				} else {
					for (int i = selectedNode.getChildCount(); i > 0; i--) {
						getTreeModel().removeNodeFromParent(
								((DefaultMutableTreeNode) selectedNode
										.getChildAt(i - 1)));
					}
					ListIterator<ParaProfExperiment> l = application
					.getExperimentList();
					while (l.hasNext()) {
						ParaProfExperiment experiment = l.next();
						DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(
								experiment);
						experiment.setDMTN(experimentNode);
						getTreeModel().insertNodeInto(experimentNode,
								selectedNode, selectedNode.getChildCount());
					}
				}
				int location = jSplitInnerPane.getDividerLocation();
				jSplitInnerPane.setRightComponent(getTable(userObject));
				jSplitInnerPane.setDividerLocation(location);
			}

			if (userObject instanceof ParaProfExperiment) {
				ParaProfExperiment experiment = (ParaProfExperiment) userObject;
				if (experiment.dBExperiment()) {
					// refresh the trials list
					for (int i = selectedNode.getChildCount(); i > 0; i--) {
						getTreeModel().removeNodeFromParent(
								((DefaultMutableTreeNode) selectedNode
										.getChildAt(i - 1)));
					}
					experiment
					.setApplication((ParaProfApplication) ((DefaultMutableTreeNode) selectedNode
							.getParent()).getUserObject());
					DatabaseAPI databaseAPI = this.getDatabaseAPI(experiment
							.getDatabase());
					if (databaseAPI != null) {
						databaseAPI.setExperiment(experiment.getID());
						if (databaseAPI.getTrialList(false) != null) {
							ListIterator<Trial> l = databaseAPI.getTrialList(
									true).listIterator();// TODO: Is xml
									// metadata required
							// here?
							while (l.hasNext()) {
								ParaProfTrial ppTrial = new ParaProfTrial(
										(Trial) l.next());
								ppTrial.setDBTrial(true);
								ppTrial.setExperiment(experiment);
								DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(
										ppTrial);
								ppTrial.setDMTN(trialNode);
								getTreeModel().insertNodeInto(trialNode,
										selectedNode,
										selectedNode.getChildCount());
							}
						}
						databaseAPI.terminate();
					}
				} else {
					for (int i = selectedNode.getChildCount(); i > 0; i--) {
						getTreeModel().removeNodeFromParent(
								((DefaultMutableTreeNode) selectedNode
										.getChildAt(i - 1)));
					}
					ListIterator<ParaProfTrial> l = experiment.getTrialList();
					while (l.hasNext()) {
						ParaProfTrial ppTrial = l.next();
						DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(
								ppTrial);
						ppTrial.setDMTN(trialNode);
						getTreeModel().insertNodeInto(trialNode, selectedNode,
								selectedNode.getChildCount());
						ppTrial.setTreePath(new TreePath(trialNode.getPath()));
					}
				}
				int location = jSplitInnerPane.getDividerLocation();
				jSplitInnerPane.setRightComponent(getTable(userObject));
				jSplitInnerPane.setDividerLocation(location);
			}
			if (userObject instanceof ParaProfTrial) {
				trialWillExpand(selectedNode);
			}
		} catch (Exception e) {
			ParaProfUtils.handleException(e);
		}
	}

	public void treeWillExpand(TreeExpansionEvent event) {
		try {
			TreePath path = event.getPath();
			if (path == null) {
				return;
			}
			DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path
			.getLastPathComponent();
			DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode
			.getParent();
			Object userObject = selectedNode.getUserObject();

			if (selectedNode.isRoot()) {
				// Do not need to do anything here.
				return;
			}

			if ((parentNode.isRoot())) {
				int location = jSplitInnerPane.getDividerLocation();
				if (selectedNode == standard) {
					jSplitInnerPane.setRightComponent(getPanelHelpMessage(1));

					// refresh the application list
					for (int i = standard.getChildCount(); i > 0; i--) {
						getTreeModel().removeNodeFromParent(
								((DefaultMutableTreeNode) standard
										.getChildAt(i - 1)));
					}
					Iterator<ParaProfApplication> l = ParaProf.applicationManager
					.getApplications().iterator();
					while (l.hasNext()) {
						ParaProfApplication application = l.next();
						DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(
								application);
						application.setDMTN(applicationNode);
						getTreeModel().insertNodeInto(applicationNode,
								standard, standard.getChildCount());
					}
				} else if (selectedNode == runtime) {
					jSplitInnerPane.setRightComponent(getPanelHelpMessage(2));
				} else {
					jSplitInnerPane.setRightComponent(getPanelHelpMessage(3));
					
					// refresh the application/trial list
					for (int i = selectedNode.getChildCount(); i > 0; i--) {
						getTreeModel().removeNodeFromParent(
								((DefaultMutableTreeNode) selectedNode
										.getChildAt(i - 1)));
					}

					Database database = (Database) userObject;
					DatabaseAPI databaseAPI = getDatabaseAPI(database);
					if (databaseAPI != null) {
						if (databaseAPI.db().getSchemaVersion() > 0) {
							// oh, this is so ugly. Create a TAUdbDatabase object
							TAUdbDatabaseAPI api = new TAUdbDatabaseAPI(databaseAPI);
							// get the views
							ListIterator<View> l = api.getViewList().listIterator();
							while (l.hasNext()) {
								ParaProfView view = new ParaProfView((View) l.next());
								DefaultMutableTreeNode viewNode = new DefaultMutableTreeNode(
										view);
								view.setDMTN(viewNode);
								getTreeModel().insertNodeInto(viewNode,
										selectedNode, selectedNode.getChildCount());
							}
						} else {
						ListIterator<Application> l = databaseAPI
						.getApplicationList().listIterator();
						while (l.hasNext()) {
							ParaProfApplication application = new ParaProfApplication(
									(Application) l.next());
							application.setDBApplication(true);
							DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(
									application);
							application.setDMTN(applicationNode);
							getTreeModel().insertNodeInto(applicationNode,
									selectedNode, selectedNode.getChildCount());
						}
						
						}
						databaseAPI.terminate();
					}

				}
				jSplitInnerPane.setDividerLocation(location);
				return;
			}

			if (userObject instanceof ParaProfApplication) {
				ParaProfApplication application = (ParaProfApplication) userObject;
				if (application.dBApplication()) {
					// remove the old experiments
					for (int i = selectedNode.getChildCount(); i > 0; i--) {
						getTreeModel().removeNodeFromParent(
								((DefaultMutableTreeNode) selectedNode
										.getChildAt(i - 1)));
					}

					// reload from the database
					DatabaseAPI databaseAPI = this.getDatabaseAPI(application
							.getDatabase());
					if (databaseAPI != null) {
						databaseAPI.setApplication(application.getID());
						ListIterator<Experiment> l = databaseAPI
						.getExperimentList().listIterator();
						while (l.hasNext()) {
							ParaProfExperiment experiment = new ParaProfExperiment(
									(Experiment) l.next());
							experiment.setDBExperiment(true);
							experiment.setApplication(application);
							DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(
									experiment);
							experiment.setDMTN(experimentNode);
							getTreeModel().insertNodeInto(experimentNode,
									selectedNode, selectedNode.getChildCount());
						}
						databaseAPI.terminate();
					}

				} else {
					for (int i = selectedNode.getChildCount(); i > 0; i--) {
						getTreeModel().removeNodeFromParent(
								((DefaultMutableTreeNode) selectedNode
										.getChildAt(i - 1)));
					}
					ListIterator<ParaProfExperiment> l = application
					.getExperimentList();
					while (l.hasNext()) {
						ParaProfExperiment experiment = l.next();
						DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(
								experiment);
						experiment.setDMTN(experimentNode);
						getTreeModel().insertNodeInto(experimentNode,
								selectedNode, selectedNode.getChildCount());
					}
				}
				int location = jSplitInnerPane.getDividerLocation();
				jSplitInnerPane.setRightComponent(getTable(userObject));
				jSplitInnerPane.setDividerLocation(location);
			}

			if (userObject instanceof ParaProfExperiment) {
				ParaProfExperiment experiment = (ParaProfExperiment) userObject;
				if (experiment.dBExperiment()) {
					// refresh the trials list
					for (int i = selectedNode.getChildCount(); i > 0; i--) {
						getTreeModel().removeNodeFromParent(
								((DefaultMutableTreeNode) selectedNode
										.getChildAt(i - 1)));
					}
					experiment.setApplication((ParaProfApplication) parentNode
							.getUserObject());
					DatabaseAPI databaseAPI = this.getDatabaseAPI(experiment
							.getDatabase());
					if (databaseAPI != null) {
						databaseAPI.setExperiment(experiment.getID());
						if (databaseAPI.getTrialList(false) != null) {
							ListIterator<Trial> l = databaseAPI.getTrialList(
									true).listIterator();// TODO: Is xml
							// metadata required
							// here?
							while (l.hasNext()) {
								ParaProfTrial ppTrial = new ParaProfTrial(
										(Trial) l.next());
								ppTrial.setDBTrial(true);
								ppTrial.setExperiment(experiment);
								DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(
										ppTrial);
								ppTrial.setDMTN(trialNode);
								getTreeModel().insertNodeInto(trialNode,
										selectedNode,
										selectedNode.getChildCount());
							}
						}
						databaseAPI.terminate();
					}
				} else {
					for (int i = selectedNode.getChildCount(); i > 0; i--) {
						getTreeModel().removeNodeFromParent(
								((DefaultMutableTreeNode) selectedNode
										.getChildAt(i - 1)));
					}
					ListIterator<ParaProfTrial> l = experiment.getTrialList();
					while (l.hasNext()) {
						ParaProfTrial ppTrial = l.next();
						DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(
								ppTrial);
						ppTrial.setDMTN(trialNode);
						getTreeModel().insertNodeInto(trialNode, selectedNode,
								selectedNode.getChildCount());
						ppTrial.setTreePath(new TreePath(trialNode.getPath()));
					}
				}
				int location = jSplitInnerPane.getDividerLocation();
				jSplitInnerPane.setRightComponent(getTable(userObject));
				jSplitInnerPane.setDividerLocation(location);
			}
			if (userObject instanceof ParaProfView) {
				ParaProfView parentView = (ParaProfView) userObject;
				// refresh the views/trials list
				for (int i = selectedNode.getChildCount(); i > 0; i--) {
					getTreeModel().removeNodeFromParent(
							((DefaultMutableTreeNode) selectedNode
									.getChildAt(i - 1)));
				}
//				parentView.setParent((ParaProfView) parentNode
//						.getUserObject());
				TAUdbDatabaseAPI databaseAPI = (TAUdbDatabaseAPI) this.getDatabaseAPI(parentView
						.getDatabase());
				if (databaseAPI != null) {
					databaseAPI.setView(parentView);
					ListIterator<View> l = databaseAPI.getViewList().listIterator();
					if(l.hasNext()){
					while (l.hasNext()) {
						ParaProfView view = new ParaProfView((View) l.next());
						DefaultMutableTreeNode viewNode = new DefaultMutableTreeNode(
								view);
						view.setDMTN(viewNode);
						getTreeModel().insertNodeInto(viewNode,
								selectedNode, selectedNode.getChildCount());
					}
					
					View allView = View.VirtualView(parentView);
					ParaProfView view = new ParaProfView(allView);
					DefaultMutableTreeNode viewNode = new DefaultMutableTreeNode(view);
					view.setDMTN(viewNode);
					getTreeModel().insertNodeInto(viewNode,	selectedNode, selectedNode.getChildCount());
					
					}
					else{
					List<Trial> l2 = databaseAPI.getTrialList(
							true);// TODO: Is xml
					          // metadata required here?
					for(Trial trial:l2){
						ParaProfTrial ppTrial = new ParaProfTrial( trial);
						ppTrial.setDBTrial(true);
						ppTrial.setView(parentView);
						DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(
								ppTrial);
						ppTrial.setDMTN(trialNode);
						getTreeModel().insertNodeInto(trialNode,
								selectedNode,
								selectedNode.getChildCount());
					}
					}
					databaseAPI.terminate();
				}
				int location = jSplitInnerPane.getDividerLocation();
				jSplitInnerPane.setRightComponent(getTable(userObject));
				jSplitInnerPane.setDividerLocation(location);
			}
			if (userObject instanceof ParaProfTrial) {
				trialWillExpand(selectedNode);
			}
		} catch (Exception e) {
			ParaProfUtils.handleException(e);
		}
	}

	private void trialWillExpand(DefaultMutableTreeNode selectedNode) {
		Object userObject = selectedNode.getUserObject();

		ParaProfTrial ppTrial = (ParaProfTrial) userObject;
		if (ppTrial.dBTrial()) {

			// test to see if trial has already been loaded
			// if so, we re-associate the ParaProfTrial with the DMTN since
			// the old one is gone
			boolean loaded = false;
			for (Enumeration<ParaProfTrial> e = loadedDBTrials.elements(); e
			.hasMoreElements();) {
				ParaProfTrial loadedTrial = e.nextElement();
				if ((ppTrial.getID() == loadedTrial.getID())
						&& (ppTrial.getExperimentID() == loadedTrial
								.getExperimentID())
								&& (ppTrial.getApplicationID() == loadedTrial
										.getApplicationID())) {
					selectedNode.setUserObject(loadedTrial);
					loadedTrial.setDMTN(selectedNode);
					ppTrial = loadedTrial;
					loaded = true;
				}
			}

			if (!loaded) {

				if (ppTrial.loading()) {
					return;
				}

				// load the trial in from the db
				ppTrial.setLoading(true);

				Database database=ppTrial.getDatabase();
				if(database!=null&&database.getLatch()!=null){
				try {
					database.getLatch().await();
				} catch (InterruptedException e1) {
					e1.printStackTrace();
				}
				}
				database.setLatch();
				
				DatabaseAPI databaseAPI = this.getDatabaseAPI(database);
				if (databaseAPI != null) {
					databaseAPI.setApplication(ppTrial.getApplicationID());
					databaseAPI.setExperiment(ppTrial.getExperimentID());
					databaseAPI.setTrial(ppTrial.getID(), true);// TODO: Is XML
					// metadata
					// required
					// here?
					DataSource dbDataSource;
					if(databaseAPI.getDb().getSchemaVersion() >0 ){
						dbDataSource = new TAUdbDataSource(databaseAPI);
					}else{
					dbDataSource = new DBDataSource(databaseAPI);
					}
					dbDataSource
					.setGenerateIntermediateCallPathData(ParaProf.preferences
							.getGenerateIntermediateCallPathData());
					ppTrial.getTrial().setDataSource(dbDataSource);
				
					final DataSource dataSource = dbDataSource;
					final ParaProfTrial theTrial = ppTrial;
					java.lang.Thread thread = new java.lang.Thread(
							new Runnable() {

								public void run() {
									try {
										dataSource.load();
										theTrial.finishLoad();
										ParaProf.paraProfManagerWindow
										.populateTrialMetrics(theTrial);
										theTrial.getDatabase().getLatch().countDown();
									} catch (final Exception e) {
										EventQueue.invokeLater(new Runnable() {
											public void run() {
												ParaProfUtils
												.handleException(e);
											}
										});
									}
								}
							});
					thread.start();

					// Add to the list of loaded trials.
					loadedDBTrials.add(ppTrial);
				}
			}
		}

		// at this point, in both the db and non-db cases, the trial
		// is either loading or not. Check this before displaying
		if (!ppTrial.loading()) {
			// refresh the metrics list
			for (int i = selectedNode.getChildCount(); i > 0; i--) {
				getTreeModel().removeNodeFromParent(
						((DefaultMutableTreeNode) selectedNode
								.getChildAt(i - 1)));
			}
			Iterator<Metric> l = ppTrial.getMetrics().iterator();
			while (l.hasNext()) {
				ParaProfMetric metric = (ParaProfMetric) l.next();
				DefaultMutableTreeNode metricNode = new DefaultMutableTreeNode(
						metric, false);
				metric.setDMTN(metricNode);
				getTreeModel().insertNodeInto(metricNode, selectedNode,
						selectedNode.getChildCount());
			}

			int location = jSplitInnerPane.getDividerLocation();
			jSplitInnerPane.setRightComponent(getTable(userObject));
			jSplitInnerPane.setDividerLocation(location);
		}
	}

	private void showMetric(ParaProfMetric ppMetric) {
		ParaProfTrial ppTrial = ppMetric.getParaProfTrial();
		ppTrial.setDefaultMetric(ppMetric);
		ppTrial.showMainWindow();
	}

	private void deleteMetric(ParaProfMetric ppMetric) {
		try {
			ParaProfTrial ppTrial = ppMetric.getParaProfTrial();

			if (ppTrial.dBTrial()) {
				DatabaseAPI databaseAPI = this.getDatabaseAPI(ppTrial
						.getDatabase());
				Trial.deleteMetric(databaseAPI.getDb(), ppTrial.getID(),
						ppMetric.getDbMetricID());
			}

			getTreeModel().removeNodeFromParent(ppMetric.getDMTN());

			ppTrial.deleteMetric(ppMetric);
		} catch (Exception e) {
			ParaProfUtils.handleException(e);
		}
	}

	private void switchToMetric(ParaProfMetric metric) {
		try {
			ParaProfTrial ppTrial = metric.getParaProfTrial();
			ppTrial.setDefaultMetric(metric);
			ppTrial.updateRegisteredObjects("dataEvent");
		} catch (Exception e) {
			ParaProfUtils.handleException(e);
		}
	}

	private void metricSelected(ParaProfMetric metric, boolean show) {
		int location = jSplitInnerPane.getDividerLocation();
		jSplitInnerPane.setRightComponent(getTable(metric));
		jSplitInnerPane.setDividerLocation(location);
		this.operand2 = this.operand1;
		// if (showApplyOperationItem.isSelected()){
		// operand1 = metric;
		// derivedMetricPanel.insertMetric(metric);
		// }

		if (show) {
			switchToMetric(metric);
		}
	}

	public ParaProfMetric getOperand1() {
		return operand1;
	}

	public ParaProfMetric getOperand2() {
		return operand2;
	}

	public void uploadMetric(ParaProfMetric metric) {
		if (metric != null) {
			DatabaseAPI databaseAPI = this.getDatabaseAPI(metric
					.getParaProfTrial().getDatabase());
			if (databaseAPI != null) {
				try {
					databaseAPI.saveTrial(metric.getParaProfTrial().getTrial(),
							metric);
				} catch (DatabaseException e) {
					ParaProfUtils.handleException(e);
				}
				databaseAPI.terminate();
			}
		}
	}

	public int[] getSelectedDBExperiment() {
		if (ParaProf.preferences.getDatabaseConfigurationFile() == null
				|| ParaProf.preferences.getDatabasePassword() == null) {
			// Check to see if the user has set configuration information.
			JOptionPane
			.showMessageDialog(
					this,
					"Please set the database configuration information (file menu).",
					"DB Configuration Error!",
					JOptionPane.ERROR_MESSAGE);
			return null;
		}

		TreePath path = tree.getSelectionPath();
		boolean error = false;
		if (path == null)
			error = true;
		else {
			DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path
			.getLastPathComponent();
			Object userObject = selectedNode.getUserObject();
			if (userObject instanceof ParaProfExperiment) {
				ParaProfExperiment paraProfExperiment = (ParaProfExperiment) userObject;
				if (paraProfExperiment.dBExperiment()) {
					int[] array = new int[2];
					array[0] = paraProfExperiment.getApplicationID();
					array[1] = paraProfExperiment.getID();
					return array;
				} else {
					error = true;
				}
			} else {
				error = true;
			}
		}
		if (error)
			JOptionPane.showMessageDialog(this,
					"Please select an db experiment first!",
					"DB Upload Error!", JOptionPane.ERROR_MESSAGE);
		return null;
	}

	private Component getPanelHelpMessage(int type) {
		JTextArea jTextArea = new JTextArea();
		jTextArea.setLineWrap(true);
		jTextArea.setWrapStyleWord(true);
		switch (type) {
		case 0:
			jTextArea.append("ParaProf Manager\n\n");
			jTextArea
			.append("This window allows you to manage all of ParaProf's loaded data.\n");
			jTextArea
			.append("Data can be static (ie, not updated at runtime),"
					+ " and loaded either remotely or locally.  You can also specify data to be uploaded at runtime.\n\n");
			break;
		case 1:
			jTextArea.append("ParaProf Manager\n\n");
			jTextArea.append("This is the Standard application section:\n\n");
			jTextArea
			.append("Standard - The classic ParaProf mode.  Data sets that are loaded at startup are placed"
					+ " under the default application automatically. Please see the ParaProf documentation for more details.\n");
			break;
		case 2:
			jTextArea.append("ParaProf Manager\n\n");
			jTextArea.append("This is the Runtime application section:\n\n");
			jTextArea
			.append("Runtime - A new feature allowing ParaProf to update data at runtime.  Please see"
					+ " the ParaProf documentation if the options are not clear.\n");
			jTextArea.append("*** THIS FEATURE IS CURRENTLY OFF ***\n");
			break;
		case 3:
			jTextArea.append("ParaProf Manager\n\n");
			jTextArea.append("This is the DB Apps application section:\n\n");
			jTextArea
			.append("DB Apps - ParaProf can load data from a database.  Please see"
					+ " the ParaProf and PerfDMF documentation for more details.\n");
			break;
		default:
			break;
		}
		return (new JScrollPane(jTextArea));
	}

	private Component getTable(Object obj) {
		final JTable table = new JTable() {
    
      public Component prepareRenderer(TableCellRenderer renderer, int row, int column) {
        Component c = super.prepareRenderer(renderer, row, column);
        if (c instanceof JComponent) {
          JComponent jc = (JComponent) c;
          Object value = getValueAt(row, column);
          if (value != null) {
            jc.setToolTipText(value.toString());
          }
        }
        return c;
      }
    };
    
    
    Utility.setTableFontHeight(table, ParaProf.preferences.getFont());

    
    table.setAutoResizeMode(JTable.AUTO_RESIZE_ALL_COLUMNS);
    if (obj instanceof ParaProfApplication) {
			table.setModel(new ApplicationTableModel(this,
					(ParaProfApplication) obj, getTreeModel()));
		} else if (obj instanceof ParaProfExperiment) {
			table.setModel(new ExperimentTableModel(this,
					(ParaProfExperiment) obj, getTreeModel()));
		} else if (obj instanceof ParaProfView) {
			table.setModel(new ViewTableModel(this,
					(ParaProfView) obj, getTreeModel()));
		} else if (obj instanceof ParaProfTrial) {
			ParaProfTrial ppTrial = (ParaProfTrial) obj;
			TrialTableModel model = new TrialTableModel(this, ppTrial,
					getTreeModel());

		  table.setModel(model);
			table.addMouseListener(model.getMouseListener(table));

			table.setDefaultRenderer(Object.class, new TrialCellRenderer(
					ppTrial.getTrial().getMetaData(), ppTrial.getTrial()
					.getUncommonMetaData()));
		} else {
			table.setModel(new MetricTableModel(this,
					(ParaProfMetric) obj, getTreeModel()));
		}
    return (new JScrollPane(table));
	}

	public ParaProfApplication addApplication(boolean dBApplication,
			DefaultMutableTreeNode treeNode) throws SQLException {
		ParaProfApplication application = null;
		if (dBApplication) {
			Database database = (Database) treeNode.getUserObject();
			DatabaseAPI databaseAPI = this.getDatabaseAPI(database);
			if (databaseAPI != null) {
				application = new ParaProfApplication();
				application.setDBApplication(true);
				application.setName("New Application");
				application.setID(databaseAPI.saveApplication(application));
				application.setDatabase(((Database) treeNode.getUserObject()));
				databaseAPI.terminate();
			}

		} else {
			application = ParaProf.applicationManager.addApplication();
			application.setName("New Application");
		}
		return application;
	}

	public ParaProfExperiment addExperiment(boolean dBExperiment,
			ParaProfApplication application) {
		ParaProfExperiment experiment = null;
		if (dBExperiment) {
			DatabaseAPI databaseAPI = this.getDatabaseAPI(application
					.getDatabase());
			if (databaseAPI != null) {
				try {
					experiment = new ParaProfExperiment(databaseAPI.db());
					experiment.setDBExperiment(true);
					experiment.setApplication(application);
					experiment.setApplicationID(application.getID());
					experiment.setName("New Experiment");
					experiment.setID(databaseAPI.saveExperiment(experiment));
					experiment.setDatabase(application.getDatabase());
				} catch (DatabaseException de) {
					ParaProfUtils.handleException(de);
				}
				databaseAPI.terminate();
			}
		} else {
			experiment = application.addExperiment();
			experiment.setName("New Experiment");
		}
		return experiment;
	}

	public void addTrial(ParaProfExperiment experiment, File files[], int fileType,
			boolean fixGprofNames, boolean monitorProfiles, String range) {
		addTrial(experiment,null, files, fileType, fixGprofNames, monitorProfiles, range);
	}
	public void addTrial(ParaProfView view, File files[], int fileType,
			boolean fixGprofNames, boolean monitorProfiles, String range) {
		addTrial(null, view, files, fileType, fixGprofNames, monitorProfiles, range);
	}
	private void addTrial(ParaProfExperiment experiment, ParaProfView view,  File files[], int fileType,
			boolean fixGprofNames, boolean monitorProfiles, String range) {
		ParaProfTrial ppTrial = null;
		DataSource dataSource = null;

		try {
			dataSource = UtilFncs.initializeDataSource(files, fileType,
					fixGprofNames, range);
			if (dataSource == null) {
				throw new RuntimeException("Error creating dataSource!");
			}
			dataSource.setGenerateIntermediateCallPathData(ParaProf.preferences
					.getGenerateIntermediateCallPathData());
		} catch (DataSourceException e) {

			if (files == null || files.length != 0) // We don't output an error
				// message if paraprof was
				// just invoked with no
				// parameters.
				ParaProfUtils.handleException(e);

			return;
		}

		ppTrial = new ParaProfTrial();
		// this must be done before setting the monitored flag
		ppTrial.getTrial().setDataSource(dataSource);
		ppTrial.setLoading(true);
		dataSource.setMonitored(monitorProfiles);
		ppTrial.setMonitored(monitorProfiles);
		if (experiment != null) {
			ppTrial.setExperiment(experiment);
			ppTrial.setApplicationID(experiment.getApplicationID());
			ppTrial.setExperimentID(experiment.getID());
		}
		if(view != null)
		ppTrial.setView(view);
		if (files.length != 0) {
			ppTrial.setPaths(files[0].getPath());
		} else {
			ppTrial.setPaths(System.getProperty("user.dir"));
		}
		if (!ParaProf.usePathNameInTrial && files.length == 1) {
			ppTrial.getTrial().setName(files[0].toString());
			ppTrial.setPaths(files[0].toString());
		} else {
			ppTrial.getTrial().setName(ppTrial.getPathReverse());
		}
		if (experiment == null || experiment.dBExperiment()) {
			loadedDBTrials.add(ppTrial);
			ppTrial.setUpload(true); // This trial is not set to a db trial
			// until after it has finished loading.
		} else {
			experiment.addTrial(ppTrial);
		}

		LoadTrialProgressWindow lpw = new LoadTrialProgressWindow(this,
				dataSource, ppTrial, false);
		lpw.setVisible(true);
	}


	public void addTrial(ParaProfApplication application,
			ParaProfExperiment experiment, File files[], int fileType,
			boolean fixGprofNames, boolean monitorProfiles) {

		addTrial(experiment, files, fileType, fixGprofNames,
				monitorProfiles, null);
		

	}

	public void populateTrialMetrics(final ParaProfTrial ppTrial) {
		try {
			loadedTrials.add(ppTrial);

			EventQueue.invokeLater(new Runnable() {
				public void run() {
					try {
						if (ppTrial.upload()) {
							// Add to the list of loaded trials.
							ppTrial.setUpload(false);
						}

						expandTrial(ppTrial);
					} catch (Exception e) {
						ParaProfUtils.handleException(e);
					}
				}
			});

		} catch (Exception e) {
			ParaProfUtils.handleException(e);
		}
	}

	// private void recurseExpand(DefaultMutableTreeNode node) {
	// if (node == null || node.isRoot()) {
	// return;
	// }
	// recurseExpand((DefaultMutableTreeNode) node.getParent());
	// tree.expandPath(new TreePath(node));
	//
	// }

	public DefaultMutableTreeNode expandApplicationType(int type,
			int applicationID, ParaProfApplication application) {
		switch (type) {
		case 0:
			// Test to see if standard is expanded, if not, expand it.
			if (!(tree.isExpanded(new TreePath(standard.getPath()))))
				tree.expandPath(new TreePath(standard.getPath()));

			// Try and find the required application node.
			for (int i = standard.getChildCount(); i > 0; i--) {
				DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) standard
				.getChildAt(i - 1);
				if (applicationID == ((ParaProfApplication) defaultMutableTreeNode
						.getUserObject()).getID())
					return defaultMutableTreeNode;
			}
			// Required application node was not found, try adding it.
			if (application != null) {
				DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(
						application);
				application.setDMTN(applicationNode);
				getTreeModel().insertNodeInto(applicationNode, standard,
						standard.getChildCount());
				return applicationNode;
			}
			return null;

		case 2:

			DefaultMutableTreeNode dbNode = null;
			// Database db = null;
			try {
				for (int i = 0; i < root.getChildCount(); i++) {
					DefaultMutableTreeNode node = (DefaultMutableTreeNode) root
					.getChildAt(i);
					if (node.getUserObject() == application.getDatabase()) {
						dbNode = node;
						// db = (Database)
						node.getUserObject();
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}

			// Test to see if dbApps is expanded, if not, expand it.
			if (!(tree.isExpanded(new TreePath(dbNode.getPath()))))
				tree.expandPath(new TreePath(dbNode.getPath()));

			// Try and find the required application node.
			for (int i = dbNode.getChildCount(); i > 0; i--) {
				DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) dbNode
				.getChildAt(i - 1);
				if (applicationID == ((ParaProfApplication) defaultMutableTreeNode
						.getUserObject()).getID())
					return defaultMutableTreeNode;
			}
			// Required application node was not found, try adding it.
			if (application != null) {
				DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(
						application);
				application.setDMTN(applicationNode);
				getTreeModel().insertNodeInto(applicationNode, dbNode,
						dbNode.getChildCount());
				return applicationNode;
			}
			return null;

		default:
			break;
		}
		return null;
	}

	public DefaultMutableTreeNode expandApplicationType(Database database,
			ParaProfApplication application) {

		if (database == null) {
			// Test to see if standard is expanded, if not, expand it.
			if (!(tree.isExpanded(new TreePath(standard.getPath()))))
				tree.expandPath(new TreePath(standard.getPath()));

			// Try and find the required application node.
			for (int i = standard.getChildCount(); i > 0; i--) {
				DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) standard
				.getChildAt(i - 1);
				if (application.getID() == ((ParaProfApplication) defaultMutableTreeNode
						.getUserObject()).getID())
					return defaultMutableTreeNode;
			}
			// Required application node was not found, try adding it.
			if (application != null) {
				DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(
						application);
				application.setDMTN(applicationNode);
				getTreeModel().insertNodeInto(applicationNode, standard,
						standard.getChildCount());
				return applicationNode;
			}
			return null;

		} else {

			DefaultMutableTreeNode dbNode = null;
			// Database db = null;
			for (int i = 0; i < root.getChildCount(); i++) {
				DefaultMutableTreeNode node = (DefaultMutableTreeNode) root
				.getChildAt(i);
				if ((node.getUserObject() instanceof Database)
						&& (((Database) node.getUserObject()).getConfig()
								.getPath()).compareTo((application
										.getDatabase()).getConfig().getPath()) == 0) {
					dbNode = node;
					// db = (Database)
					node.getUserObject();
				}
			}

			// Test to see if dbApps is expanded, if not, expand it.
			if (!(tree.isExpanded(new TreePath(dbNode.getPath()))))
				tree.expandPath(new TreePath(dbNode.getPath()));

			// Try and find the required application node.
			for (int i = dbNode.getChildCount(); i > 0; i--) {
				DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) dbNode
				.getChildAt(i - 1);
				if (application.getID() == ((ParaProfApplication) defaultMutableTreeNode
						.getUserObject()).getID())
					return defaultMutableTreeNode;
			}
			// Required application node was not found, try adding it.
			if (application != null) {
				DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(
						application);
				application.setDMTN(applicationNode);
				getTreeModel().insertNodeInto(applicationNode, dbNode,
						dbNode.getChildCount());
				return applicationNode;
			}
			return null;

		}
	}

	// Expands the given application
	public DefaultMutableTreeNode expandApplication(int type,
			ParaProfApplication application, ParaProfExperiment experiment) {
		DefaultMutableTreeNode applicationNode = this.expandApplicationType(
				type, application.getID(), application);
		if (applicationNode != null) {
			// Expand the application.
			tree.expandPath(new TreePath(applicationNode.getPath()));

			// Try and find the required experiment node.
			tree.expandPath(new TreePath(standard.getPath()));
			for (int i = applicationNode.getChildCount(); i > 0; i--) {
				DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) applicationNode
				.getChildAt(i - 1);
				if (experiment.getID() == ((ParaProfExperiment) defaultMutableTreeNode
						.getUserObject()).getID())
					return defaultMutableTreeNode;
			}
			// Required experiment node was not found, try adding it.
			if (experiment != null) {
				DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(
						experiment);
				experiment.setDMTN(experimentNode);
				getTreeModel().insertNodeInto(experimentNode, applicationNode,
						applicationNode.getChildCount());
				return experimentNode;
			}
			return null;
		}
		return null;
	}

	// Expands the given application
	public DefaultMutableTreeNode expandApplication(
			ParaProfApplication application, ParaProfExperiment experiment) {
		DefaultMutableTreeNode applicationNode = this.expandApplicationType(
				application.getDatabase(), application);
		if (applicationNode != null) {
			// Expand the application.
			tree.expandPath(new TreePath(applicationNode.getPath()));

			// Try and find the required experiment node.
			tree.expandPath(new TreePath(standard.getPath()));
			for (int i = applicationNode.getChildCount(); i > 0; i--) {
				DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) applicationNode
				.getChildAt(i - 1);
				if (experiment.getID() == ((ParaProfExperiment) defaultMutableTreeNode
						.getUserObject()).getID())
					return defaultMutableTreeNode;
			}
			// Required experiment node was not found, try adding it.
			if (experiment != null) {
				DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(
						experiment);
				experiment.setDMTN(experimentNode);
				getTreeModel().insertNodeInto(experimentNode, applicationNode,
						applicationNode.getChildCount());
				return experimentNode;
			}
			return null;
		}
		return null;
	}

	public DefaultMutableTreeNode expandExperiment(
			ParaProfExperiment experiment, ParaProfTrial ppTrial) {
		ParaProfApplication app = (ParaProfApplication) experiment
		.getApplication();
		if (app == null)
			app = new ParaProfApplication();
		DefaultMutableTreeNode experimentNode = this.expandApplication(app,
				experiment);
		if (experimentNode != null) {
			// Expand the experiment.
			tree.expandPath(new TreePath(experimentNode.getPath()));

			// Try and find the required trial node.
			for (int i = experimentNode.getChildCount(); i > 0; i--) {
				DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) experimentNode
				.getChildAt(i - 1);
				if (ppTrial.getID() == ((ParaProfTrial) defaultMutableTreeNode
						.getUserObject()).getID())
					return defaultMutableTreeNode;
			}
			// Required trial node was not found, try adding it.
			if (ppTrial != null) {
				DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(
						ppTrial);
				ppTrial.setDMTN(trialNode);
				getTreeModel().insertNodeInto(trialNode, experimentNode,
						experimentNode.getChildCount());
				return trialNode;
			}
			return null;
		}
		return null;
	}


	public DefaultMutableTreeNode expandExperiment(int type, int applicationID,
			int experimentID, int trialID, ParaProfApplication application,
			ParaProfExperiment experiment, ParaProfTrial ppTrial) {
		DefaultMutableTreeNode experimentNode = this.expandApplication(type,
				application, experiment);
		if (experimentNode != null) {
			// Expand the experiment.
			tree.expandPath(new TreePath(experimentNode.getPath()));

			// Try and find the required trial node.
			for (int i = experimentNode.getChildCount(); i > 0; i--) {
				DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) experimentNode
				.getChildAt(i - 1);
				if (trialID == ((ParaProfTrial) defaultMutableTreeNode
						.getUserObject()).getID())
					return defaultMutableTreeNode;
			}
			// Required trial node was not found, try adding it.
			if (ppTrial != null) {
				DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(
						ppTrial);
				ppTrial.setDMTN(trialNode);
				getTreeModel().insertNodeInto(trialNode, experimentNode,
						experimentNode.getChildCount());
				return trialNode;
			}
			return null;
		}
		return null;
	}

	public void expandTrial(ParaProfTrial ppTrial) {
		DefaultMutableTreeNode trialNode = null;
		if (ppTrial.getView() != null) {
			trialNode = this.expandView(ppTrial.getView(), ppTrial);
		} else {
			trialNode = this.expandExperiment(ppTrial.getExperiment(), ppTrial);
		}
		// Expand the trial.
		if (trialNode != null) {
			if (tree.isExpanded(new TreePath(trialNode.getPath())))
				tree.collapsePath(new TreePath(trialNode.getPath()));
			tree.expandPath(new TreePath(trialNode.getPath()));
		}
	}

	private DefaultMutableTreeNode expandView(ParaProfView view) {
		DefaultMutableTreeNode dbNode = null;
		// Database db = null;
		try {
			for (int i = 0; i < root.getChildCount(); i++) {
				DefaultMutableTreeNode node = (DefaultMutableTreeNode) root
				.getChildAt(i);
				if (node.getUserObject() == view.getDatabase()) {
					dbNode = node;
					break;
					// db = (Database)
					//node.getUserObject();
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		// Test to see if dbApps is expanded, if not, expand it.
		if (!(tree.isExpanded(new TreePath(dbNode.getPath()))))
			tree.expandPath(new TreePath(dbNode.getPath()));
		
		//Make a list of the selected view's parents
		ArrayList<View> treePath = new ArrayList<View>();
		View tmpView=view;
		while(tmpView!=null){
			treePath.add(tmpView);
			tmpView=tmpView.getParent();
		}

		//From the top of the tree, find each parent view
		DefaultMutableTreeNode tmpNode=dbNode;
		for(int v = treePath.size()-1;v>=0;v--)
		{
		// Try and find the required view node.
		for (int i = tmpNode.getChildCount(); i > 0; i--) {
			DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) tmpNode
			.getChildAt(i - 1);
			if (treePath.get(v).getID() == ((ParaProfView) defaultMutableTreeNode
					.getUserObject()).getID()){
				tmpNode = defaultMutableTreeNode;
				break;
			}
		}
		}
		
		if(tmpNode!=null&&((ParaProfView)tmpNode.getUserObject()).getID()==view.getID()){
			return tmpNode;
		}
		
		// Required view node was not found, try adding it.
		if (view != null) {
			DefaultMutableTreeNode viewNode = new DefaultMutableTreeNode(
					view);
			view.setDMTN(viewNode);
			getTreeModel().insertNodeInto(viewNode, dbNode,
					dbNode.getChildCount());
			return viewNode;
		}
		return null;
	}
	
	private DefaultMutableTreeNode expandView(ParaProfView view,
			ParaProfTrial ppTrial) {
		DefaultMutableTreeNode viewNode = this.expandView(view);
		if (viewNode != null) {
			// Expand the experiment.
			tree.expandPath(new TreePath(viewNode.getPath()));

			// Try and find the required trial node.
			for (int i = viewNode.getChildCount(); i > 0; i--) {
				DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) viewNode
				.getChildAt(i - 1);
				if( defaultMutableTreeNode.getUserObject() instanceof ParaProfTrial)
				if (ppTrial.getID() == ((ParaProfTrial) defaultMutableTreeNode
						.getUserObject()).getID())
					return defaultMutableTreeNode;
			}
			// Required trial node was not found, try adding it.
			if (ppTrial != null) {
				DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(
						ppTrial);
				ppTrial.setDMTN(trialNode);
				getTreeModel().insertNodeInto(trialNode, viewNode,
						viewNode.getChildCount());
				return trialNode;
			}
			return null;
		}
		return null;
	}

	public void expandTrial(int type, int applicationID, int experimentID,
			int trialID, ParaProfApplication application,
			ParaProfExperiment experiment, ParaProfTrial ppTrial) {
		DefaultMutableTreeNode trialNode = this.expandExperiment(type,
				applicationID, experimentID, trialID, application, experiment,
				ppTrial);
		// Expand the trial.
		if (trialNode != null) {
			if (tree.isExpanded(new TreePath(trialNode.getPath())))
				tree.collapsePath(new TreePath(trialNode.getPath()));
			tree.expandPath(new TreePath(trialNode.getPath()));
		}
	}

	public String getDatabaseName() {
		if (dbDisplayName == null) {
			try {
				if (ParaProf.preferences.getDatabaseConfigurationFile() == null) {
					dbDisplayName = "";
				}

				ParseConfig parser = new ParseConfig(
						ParaProf.preferences.getDatabaseConfigurationFile());
				// dbDisplayName = "[" + parser.getDBHost() + " (" +
				// parser.getDBType() + ")]";
				dbDisplayName = parser.getConnectionString();
				if (dbDisplayName.compareTo("") == 0) {
					dbDisplayName = "none";
				}
			} catch (Exception e) {
				// Forget it
				dbDisplayName = "none";
			}
		}
		return dbDisplayName;
	}

	public Database getDefaultDatabase() {
		return databases.get(0);
		// ParseConfig config = new
		// ParseConfig(ParaProf.preferences.getDatabaseConfigurationFile());
		// return new Database("default", config);
	}

	/*
	 * Creating a new connection for every operation creates a lot of overhead. Cache the last created DBAPI object for reuse in batch operaitons 
	 */
	private DatabaseAPI tmpDBAPI=null;
	public DatabaseAPI getDatabaseAPI(Database database) {
		try {

			if (database == null) {
				database = getDefaultDatabase();
			}

			// Basic checks done, try to access the db.
			if(tmpDBAPI!=null&&tmpDBAPI.getDb().getDatabase().getID()==(database.getID()) && !tmpDBAPI.getDb().isClosed()){
				return tmpDBAPI;
			}
			
			DatabaseAPI databaseAPI = new DatabaseAPI();
			databaseAPI.initialize(database);
			if (databaseAPI.db().getSchemaVersion() > 0) {
				// copy the DatabaseAPI object data into a new TAUdbDatabaseAPI object
				databaseAPI = new TAUdbDatabaseAPI(databaseAPI);
			}
			tmpDBAPI=databaseAPI;

			// // Some strangeness here, we retrieve the metadata columns for
			// the non-db trials
			// // from the "first" database in the list. Very screwy in my
			// opinion.
			// if (!metaDataRetrieved) {
			// DatabaseAPI defaultDatabaseAPI = new DatabaseAPI();
			// defaultDatabaseAPI.initialize(getDefaultDatabase());
			// metaDataRetrieved = true;
			// for (Iterator it =
			// ParaProf.applicationManager.getApplications().iterator();
			// it.hasNext();) {
			// ParaProfApplication ppApp = (ParaProfApplication) it.next();
			// if (!ppApp.dBApplication()) {
			// ppApp.setDatabase(getDefaultDatabase());
			// for (Iterator it2 = ppApp.getExperimentList(); it2.hasNext();) {
			// ParaProfExperiment ppExp = (ParaProfExperiment) it2.next();
			// ppExp.setDatabase(getDefaultDatabase());
			// for (Iterator it3 = ppExp.getTrialList(); it3.hasNext();) {
			// ParaProfTrial ppTrial = (ParaProfTrial) it3.next();
			// ppTrial.getTrial().setDatabase(getDefaultDatabase());
			// }
			// }
			// }
			// }
			// defaultDatabaseAPI.terminate();
			// }

			// dbAPI = databaseAPI;
			return databaseAPI;
		} catch (Exception e) {
			// Try and determine what went wrong, and then popup the help window
			// giving the user some idea of what to do.
			ParaProf.getHelpWindow().setVisible(true);
			// Clear the window first.
			ParaProf.getHelpWindow().clearText();
			ParaProf.getHelpWindow().writeText(
			"There was an error connecting to the database!");
			ParaProf.getHelpWindow().writeText("");
			ParaProf.getHelpWindow()
			.writeText(
					"Please see the help items below to try and resolve this issue."
					+ " If none of those work, send an email to tau-bugs@cs.uoregon.edu"
					+ " including as complete a description of the problem as possible.");
			ParaProf.getHelpWindow().writeText("");
			ParaProf.getHelpWindow().writeText("------------------");
			ParaProf.getHelpWindow().writeText("");

			ParaProf.getHelpWindow()
			.writeText(
					"1) JDBC driver issue:"
					+ " The JDBC driver is required in your classpath. If you ran ParaProf using"
					+ " the shell script provided in tau (paraprof), then the default."
					+ " location used is $LOCATION_OF_TAU_ROOT/$ARCH/lib.");
			ParaProf.getHelpWindow().writeText("");
			ParaProf.getHelpWindow()
			.writeText(
					" If you ran ParaProf manually, make sure that the location of"
					+ " the JDBC driver is in your classpath (you can set this in your."
					+ " environment, or as a commmand line option to java. As an example, PostgreSQL"
					+ " uses postgresql.jar as its JDBC driver name.");
			ParaProf.getHelpWindow().writeText("");
			ParaProf.getHelpWindow()
			.writeText(
					"2) Network connection issue:"
					+ " Check your ability to connect to the database. You might be connecting to the"
					+ " incorrect port (PostgreSQL uses port 5432 by default). Also make sure that"
					+ " if there exists a firewall on you network (or local machine), it is not"
					+ " blocking you connection. Also check your database logs to ensure that you have"
					+ " permission to connect to the server.");
			ParaProf.getHelpWindow().writeText("");
			ParaProf.getHelpWindow()
			.writeText(
					"3) Password issue:"
					+ " Make sure that your password is set correctly. If it is not in the perfdmf"
					+ " configuration file, you can enter it manually by selecting"
					+ "  File -> Database Configuration in the ParaProfManagerWindow window.");
			ParaProf.getHelpWindow().writeText("");
			ParaProf.getHelpWindow().writeText("------------------");
			ParaProf.getHelpWindow().writeText("");

			ParaProf.getHelpWindow().writeText(
			"The full error is given below:\n");

			StringWriter sw = new StringWriter();
			PrintWriter pw = new PrintWriter(sw);
			if (e instanceof TauRuntimeException) { // unwrap
				ParaProf.getHelpWindow().writeText(
						((TauRuntimeException) e).getMessage() + "\n\n");
				e = ((TauRuntimeException) e).getActualException();
			}
			e.printStackTrace(pw);
			pw.close();
			ParaProf.getHelpWindow().writeText(sw.toString());

			EventQueue.invokeLater(new Runnable() {
				public void run() {
					ParaProf.getHelpWindow().getScrollPane()
					.getVerticalScrollBar().setValue(0);
				}
			});

			return null;
		}
	}

	// Respond correctly when this window is closed.
	void thisWindowClosing(java.awt.event.WindowEvent e) {
		closeThisWindow();
	}

	void closeThisWindow() {
		try {
			ParaProf.preferences.setManagerWindowPosition(this.getLocation());
			setVisible(false);
			ParaProf.decrementNumWindows();
		} catch (Exception e) {
			// do nothing
		}
		dispose();
	}

	/**
	 * Returns all the loaded trials
	 * 
	 * @return the loaded trials
	 */
	public Vector<ParaProfTrial> getLoadedTrials() {
		return loadedTrials;
	}

	public Vector<ParaProfTrial> getLoadedDBTrials() {
		return loadedDBTrials;
	}

	public void setTreeModel(DefaultTreeModel treeModel) {
		this.treeModel = treeModel;
	}

	public DefaultTreeModel getTreeModel() {
		return treeModel;
	}

	public class AutoScrollingJTree extends JTree implements Autoscroll {
		/**
		 * 
		 */
		private static final long serialVersionUID = 2307960353540807206L;
		/**
		 * 
		 */
		private int margin = 12;

		public AutoScrollingJTree() {
			super();
		}

		public AutoScrollingJTree(DefaultTreeModel treeModel) {
			super(treeModel);
		}

		public void autoscroll(Point p) {
			int realrow = getRowForLocation(p.x, p.y);
			Rectangle outer = getBounds();
			realrow = (p.y + outer.y <= margin ? realrow < 1 ? 0 : realrow - 1
					: realrow < getRowCount() - 1 ? realrow + 1 : realrow);
			scrollRowToVisible(realrow);
		}

		public Insets getAutoscrollInsets() {
			Rectangle outer = getBounds();
			Rectangle inner = getParent().getBounds();
			return new Insets(inner.y - outer.y + margin, inner.x - outer.x
					+ margin, outer.height - inner.height - inner.y + outer.y
					+ margin, outer.width - inner.width - inner.x + outer.x
					+ margin);
		}

	}

}
