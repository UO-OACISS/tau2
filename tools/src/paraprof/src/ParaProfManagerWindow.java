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
 * <P>CVS $Id: ParaProfManagerWindow.java,v 1.26 2007/05/31 19:22:53 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.26 $
 * @see		ParaProfManagerTableModel
 */

package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.sql.SQLException;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.*;

import edu.uoregon.tau.common.TauRuntimeException;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.paraprof.tablemodel.*;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.perfdmf.database.*;

public class ParaProfManagerWindow extends JFrame implements ActionListener, TreeSelectionListener, TreeWillExpandListener {

    private DefaultMutableTreeNode root;
    private JTree tree = null;
    private DefaultTreeModel treeModel = null;
    private DefaultMutableTreeNode standard = null;
    private DefaultMutableTreeNode runtime = null;
    //private DefaultMutableTreeNode dbApps = null;
    private JSplitPane jSplitInnerPane = null;
    private JSplitPane jSplitOuterPane = null;

    private JCheckBoxMenuItem showApplyOperationItem = null;
    private DerivedMetricPanel derivedMetricPanel = new DerivedMetricPanel(this);

    private JScrollPane treeScrollPane;

    private Vector loadedDBTrials = new Vector();
    private Vector loadedTrials = new Vector();

    private boolean metaDataRetrieved;

    //Popup menu stuff.
    private JPopupMenu popup1 = new JPopupMenu();
    private JPopupMenu stdAppPopup = new JPopupMenu();
    private JPopupMenu stdExpPopup = new JPopupMenu();
    private JPopupMenu stdTrialPopup = new JPopupMenu();
    private JPopupMenu dbAppPopup = new JPopupMenu();
    private JPopupMenu dbExpPopup = new JPopupMenu();
    private JPopupMenu dbTrialPopup = new JPopupMenu();

    private JPopupMenu runtimePopup = new JPopupMenu();

    private Object clickedOnObject = null;
    private ParaProfMetric operand1 = null;
    private ParaProfMetric operand2 = null;

    private String dbDisplayName;

    private List databases;

    public void refreshDatabases() {
        DefaultMutableTreeNode treeNode;
        Iterator dbs = Database.getDatabases().iterator();
        Enumeration nodes = root.children();
        while (nodes.hasMoreElements() && dbs.hasNext()) {
            treeNode = (DefaultMutableTreeNode) nodes.nextElement();
            if (treeNode.getUserObject() != "Standard Applications") {
                treeNode.setUserObject(dbs.next());
            }
        }
        while (dbs.hasNext()) {
            root.add(new DefaultMutableTreeNode(dbs.next()));
        }

        List toRemove = new ArrayList();
        while (nodes.hasMoreElements()) {
            treeNode = (DefaultMutableTreeNode) nodes.nextElement();
            toRemove.add(treeNode);
        }

        for (int i = 0; i < toRemove.size(); i++) {
            treeNode = (DefaultMutableTreeNode) toRemove.get(i);
            treeNode.removeFromParent();
        }
        treeModel.reload();

    }

    public ParaProfManagerWindow() {

        //Window Stuff.
        int windowWidth = 800;
        int windowHeight = 515;

        //Grab the screen size.
        Toolkit tk = Toolkit.getDefaultToolkit();
        Dimension screenDimension = tk.getScreenSize();
        int screenHeight = screenDimension.height;
        int screenWidth = screenDimension.width;

        Point savedPosition = ParaProf.preferences.getManagerWindowPosition();

        if (savedPosition == null || (savedPosition.x + windowWidth) > screenWidth
                || (savedPosition.y + windowHeight > screenHeight)) {

            //Find the center position with respect to this window.
            int xPosition = (screenWidth - windowWidth) / 2;
            int yPosition = (screenHeight - windowHeight) / 2;

            //Offset a little so that we do not interfere too much with the
            //main window which comes up in the center of the screen.
            if (xPosition > 50)
                xPosition = xPosition - 50;
            if (yPosition > 50)
                yPosition = yPosition - 50;

            this.setLocation(xPosition, yPosition);
        } else {
            this.setLocation(savedPosition);
        }

        if (ParaProf.demoMode) {
            this.setLocation(0, 0);
        }

        setSize(ParaProfUtils.checkSize(new java.awt.Dimension(windowWidth, windowHeight)));
        setTitle("TAU: ParaProf Manager");
        ParaProfUtils.setFrameIcon(this);

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        setupMenus();

        root = new DefaultMutableTreeNode("Applications");
        standard = new DefaultMutableTreeNode("Standard Applications");
        //dbApps = new DefaultMutableTreeNode("DB (" + getDatabaseName() + ")");

        //root.add(std);

        root.add(standard);

        databases = Database.getDatabases();
        for (Iterator it = databases.iterator(); it.hasNext();) {
            Database database = (Database) it.next();
            DefaultMutableTreeNode dbNode = new DefaultMutableTreeNode(database);
            root.add(dbNode);
        }
        //root.add(dbApps);

        treeModel = new DefaultTreeModel(root);
        treeModel.setAsksAllowsChildren(true);

        tree = new JTree(treeModel);
        //tree = new JTree(new DataManagerTreeModel());
        tree.getSelectionModel().setSelectionMode(TreeSelectionModel.SINGLE_TREE_SELECTION);
        ParaProfTreeCellRenderer renderer = new ParaProfTreeCellRenderer();
        tree.setCellRenderer(renderer);

        //Add a mouse listener for this tree.
        MouseListener ml = new MouseAdapter() {
            public void mousePressed(MouseEvent evt) {
                try {
                    int selRow = tree.getRowForLocation(evt.getX(), evt.getY());
                    TreePath path = tree.getPathForLocation(evt.getX(), evt.getY());
                    if (path != null) {
                        DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
                        DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
                        Object userObject = selectedNode.getUserObject();

                        if (ParaProfUtils.rightClick(evt)) {
                            if (userObject instanceof ParaProfApplication) {
                                clickedOnObject = userObject;
                                if (((ParaProfApplication) userObject).dBApplication()) {
                                    dbAppPopup.show(ParaProfManagerWindow.this, evt.getX(), evt.getY()
                                            - treeScrollPane.getVerticalScrollBar().getValue());
                                } else {
                                    stdAppPopup.show(ParaProfManagerWindow.this, evt.getX(), evt.getY()
                                            - treeScrollPane.getVerticalScrollBar().getValue());
                                }
                            } else if (userObject instanceof ParaProfExperiment) {
                                clickedOnObject = userObject;
                                if (((ParaProfExperiment) userObject).dBExperiment()) {
                                    dbExpPopup.show(ParaProfManagerWindow.this, evt.getX(), evt.getY()
                                            - treeScrollPane.getVerticalScrollBar().getValue());
                                } else {
                                    stdExpPopup.show(ParaProfManagerWindow.this, evt.getX(), evt.getY()
                                            - treeScrollPane.getVerticalScrollBar().getValue());
                                }

                            } else if (userObject instanceof ParaProfTrial) {
                                clickedOnObject = userObject;
                                if (((ParaProfTrial) userObject).dBTrial()) {
                                    dbTrialPopup.show(ParaProfManagerWindow.this, evt.getX(), evt.getY()
                                            - treeScrollPane.getVerticalScrollBar().getValue());
                                } else {
                                    stdTrialPopup.show(ParaProfManagerWindow.this, evt.getX(), evt.getY()
                                            - treeScrollPane.getVerticalScrollBar().getValue());
                                }

                            } else {
                                // standard or database
                                clickedOnObject = selectedNode;
                                popup1.show(ParaProfManagerWindow.this, evt.getX(), evt.getY()
                                        - treeScrollPane.getVerticalScrollBar().getValue());

                            }
                        } else {
                            if (evt.getClickCount() == 2) {
                                if (userObject instanceof ParaProfMetric) {
                                    metricSelected((ParaProfMetric) userObject, true);
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

        //Add tree listeners.
        tree.addTreeSelectionListener(this);
        tree.addTreeWillExpandListener(this);

        // Place it in a scroll pane
        treeScrollPane = new JScrollPane(tree);

        //####################################
        //Set up the split panes, and add to content pane.
        //####################################
        jSplitInnerPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, treeScrollPane, getPanelHelpMessage(0));
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

                JLabel label = new JLabel("<html>Enter password for user '" + config.getDBUserName() + "'<br> Database: '"
                        + config.getDBName() + "' (" + config.getDBHost() + ":" + config.getDBPort() + ")</html>");

                JPasswordField password = new JPasswordField(15);

                gbc.fill = GridBagConstraints.BOTH;
                gbc.anchor = GridBagConstraints.CENTER;
                gbc.weightx = 1;
                gbc.weighty = 1;
                ParaProfUtils.addCompItem(promptPanel, label, gbc, 0, 0, 1, 1);
                gbc.fill = GridBagConstraints.HORIZONTAL;
                ParaProfUtils.addCompItem(promptPanel, password, gbc, 1, 0, 1, 1);

                if (JOptionPane.showConfirmDialog(null, promptPanel, "Enter Password", JOptionPane.OK_CANCEL_OPTION) == JOptionPane.OK_OPTION) {
                    return new String(password.getPassword());
                } else {
                    return null;
                }
            }
        });

        ParaProf.incrementNumWindows();
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

        //Options menu.
        JMenu optionsMenu = new JMenu("Options");

        showApplyOperationItem = new JCheckBoxMenuItem("Show Derived Metric Panel", false);
        showApplyOperationItem.addActionListener(this);
        optionsMenu.add(showApplyOperationItem);

        //Help menu.
        JMenu helpMenu = new JMenu("Help");

        JMenuItem showHelpWindowItem = new JMenuItem("Show Help Window");
        showHelpWindowItem.addActionListener(this);
        helpMenu.add(showHelpWindowItem);

        JMenuItem aboutItem = new JMenuItem("About ParaProf");
        aboutItem.addActionListener(this);
        helpMenu.add(aboutItem);

        //Now, add all the menus to the main menu.
        mainMenu.add(fileMenu);
        mainMenu.add(optionsMenu);
        mainMenu.add(helpMenu);
        setJMenuBar(mainMenu);

        //popup menus
        JMenuItem jMenuItem = new JMenuItem("Add Application");
        jMenuItem.addActionListener(this);
        popup1.add(jMenuItem);
        jMenuItem = new JMenuItem("Add Experiment");
        jMenuItem.addActionListener(this);
        popup1.add(jMenuItem);
        jMenuItem = new JMenuItem("Add Trial");
        jMenuItem.addActionListener(this);
        popup1.add(jMenuItem);

        jMenuItem = new JMenuItem("Monitor Application");
        jMenuItem.addActionListener(this);
        runtimePopup.add(jMenuItem);

        // Standard application popup
        jMenuItem = new JMenuItem("Add Experiment");
        jMenuItem.addActionListener(this);
        stdAppPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Add Trial");
        jMenuItem.addActionListener(this);
        stdAppPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Upload Application to DB");
        jMenuItem.addActionListener(this);
        stdAppPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Delete");
        jMenuItem.addActionListener(this);
        stdAppPopup.add(jMenuItem);

        // DB application popup
        jMenuItem = new JMenuItem("Add Experiment");
        jMenuItem.addActionListener(this);
        dbAppPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Add Trial");
        jMenuItem.addActionListener(this);
        dbAppPopup.add(jMenuItem);
        //        jMenuItem = new JMenuItem("Export Application to Filesystem");
        //        jMenuItem.addActionListener(this);
        //        dbAppPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Delete");
        jMenuItem.addActionListener(this);
        dbAppPopup.add(jMenuItem);

        // Standard experiment popup
        jMenuItem = new JMenuItem("Upload Experiment to DB");
        jMenuItem.addActionListener(this);
        stdExpPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Add Trial");
        jMenuItem.addActionListener(this);
        stdExpPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Delete");
        jMenuItem.addActionListener(this);
        stdExpPopup.add(jMenuItem);

        // DB experiment popup
        jMenuItem = new JMenuItem("Add Trial");
        jMenuItem.addActionListener(this);
        dbExpPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Delete");
        jMenuItem.addActionListener(this);
        dbExpPopup.add(jMenuItem);

        // Standard trial popup
        jMenuItem = new JMenuItem("Export Profile");
        jMenuItem.addActionListener(this);
        stdTrialPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Convert to Phase Profile");
        jMenuItem.addActionListener(this);
        stdTrialPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Add Mean to Comparison Window");
        jMenuItem.addActionListener(this);
        stdTrialPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Upload Trial to DB");
        jMenuItem.addActionListener(this);
        stdTrialPopup.add(jMenuItem);

        jMenuItem = new JMenuItem("Delete");
        jMenuItem.addActionListener(this);
        stdTrialPopup.add(jMenuItem);

        // DB trial popup
        jMenuItem = new JMenuItem("Export Profile");
        jMenuItem.addActionListener(this);
        dbTrialPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Convert to Phase Profile");
        jMenuItem.addActionListener(this);
        dbTrialPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Add Mean to Comparison Window");
        jMenuItem.addActionListener(this);
        dbTrialPopup.add(jMenuItem);
        jMenuItem = new JMenuItem("Delete");
        jMenuItem.addActionListener(this);
        dbTrialPopup.add(jMenuItem);

    }

    public void recomputeStats() {
        for (int i = 0; i < loadedTrials.size(); i++) {
            ParaProfTrial ppTrial = (ParaProfTrial) loadedTrials.get(i);
            ppTrial.getDataSource().generateDerivedData();
        }
    }

    public void handleDelete(Object clickedOnObject) throws SQLException, DatabaseException {
        if (clickedOnObject instanceof ParaProfApplication) {
            ParaProfApplication application = (ParaProfApplication) clickedOnObject;
            if (application.dBApplication()) {

                DatabaseAPI databaseAPI = this.getDatabaseAPI(application.getDatabase());
                if (databaseAPI != null) {
                    databaseAPI.deleteApplication(application.getID());
                    databaseAPI.terminate();
                    //Remove any loaded trials associated with this application.
                    for (Enumeration e = loadedDBTrials.elements(); e.hasMoreElements();) {
                        ParaProfTrial loadedTrial = (ParaProfTrial) e.nextElement();
                        if (loadedTrial.getApplicationID() == application.getID() && loadedTrial.loading() == false)
                            loadedDBTrials.remove(loadedTrial);
                    }
                    treeModel.removeNodeFromParent(application.getDMTN());
                }

            } else {
                ParaProf.applicationManager.removeApplication(application);
                treeModel.removeNodeFromParent(application.getDMTN());
            }
        } else if (clickedOnObject instanceof ParaProfExperiment) {
            ParaProfExperiment experiment = (ParaProfExperiment) clickedOnObject;
            if (experiment.dBExperiment()) {

                DatabaseAPI databaseAPI = this.getDatabaseAPI(experiment.getDatabase());
                if (databaseAPI != null) {
                    databaseAPI.deleteExperiment(experiment.getID());
                    databaseAPI.terminate();
                    //Remove any loaded trials associated with this application.
                    for (Enumeration e = loadedDBTrials.elements(); e.hasMoreElements();) {
                        ParaProfTrial loadedTrial = (ParaProfTrial) e.nextElement();
                        if (loadedTrial.getApplicationID() == experiment.getApplicationID()
                                && loadedTrial.getExperimentID() == experiment.getID() && loadedTrial.loading() == false)
                            loadedDBTrials.remove(loadedTrial);
                    }
                    if (experiment.getDMTN() != null) {
                        treeModel.removeNodeFromParent(experiment.getDMTN());
                    }
                }

            } else {

                //                for (Iterator it = loadedTrials.iterator(); it.hasNext();) {
                //                    ParaProfTrial ppTrial = (ParaProfTrial) it.next();
                //                    
                //                    if (ppTrial.getExperiment() == experiment) {
                //                        System.out.println("found it");
                //                    }
                //                    
                //                    
                //                }
                //                for (Iterator it=experiment.getTrialList(); it.hasNext();) {
                //                    ParaProfTrial ppTrial = (ParaProfTrial) it.next();
                //                    handleDelete(ppTrial);
                //                }
                experiment.getApplication().removeExperiment(experiment);
                treeModel.removeNodeFromParent(experiment.getDMTN());
            }

        } else if (clickedOnObject instanceof ParaProfTrial) {
            ParaProfTrial ppTrial = (ParaProfTrial) clickedOnObject;
            if (ppTrial.dBTrial()) {

                DatabaseAPI databaseAPI = this.getDatabaseAPI(ppTrial.getDatabase());
                if (databaseAPI != null) {
                    databaseAPI.deleteTrial(ppTrial.getID());
                    databaseAPI.terminate();
                    //Remove any loaded trials associated with this application.
                    for (Enumeration e = loadedDBTrials.elements(); e.hasMoreElements();) {
                        ParaProfTrial loadedTrial = (ParaProfTrial) e.nextElement();
                        if (loadedTrial.getApplicationID() == ppTrial.getApplicationID()
                                && loadedTrial.getExperimentID() == ppTrial.getID() && loadedTrial.getID() == ppTrial.getID()
                                && loadedTrial.loading() == false)
                            loadedDBTrials.remove(loadedTrial);
                    }
                    treeModel.removeNodeFromParent(ppTrial.getDMTN());
                }
            } else {
                ppTrial.getExperiment().removeTrial(ppTrial);
                treeModel.removeNodeFromParent(ppTrial.getDMTN());
                //                ppTrial.getFullDataWindow().dispose();
                //                loadedTrials.remove(ppTrial);
                //                System.gc();
            }
        }
    }

    private boolean isLoaded(ParaProfTrial ppTrial) {
        boolean loaded = true;
        if (ppTrial.dBTrial()) {
            loaded = false;
            for (Enumeration e = loadedDBTrials.elements(); e.hasMoreElements();) {
                ParaProfTrial loadedTrial = (ParaProfTrial) e.nextElement();
                if ((ppTrial.getID() == loadedTrial.getID()) && (ppTrial.getExperimentID() == loadedTrial.getExperimentID())
                        && (ppTrial.getApplicationID() == loadedTrial.getApplicationID())) {
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
                    ParaProfApplication application = addApplication(false, standard);
                    if (application != null) {
                        this.expandApplicationType(0, application.getID(), application);
                        ParaProfExperiment experiment = addExperiment(false, application);
                        if (experiment != null) {
                            this.expandApplication(0, application, experiment);
                            (new LoadTrialWindow(this, application, experiment, true, true)).setVisible(true);
                        }
                    }
                } else if (arg.equals("Preferences...")) {
                    ParaProf.preferencesWindow.showPreferencesWindow(this);
                } else if (arg.equals("Close This Window")) {
                    closeThisWindow();

                } else if (arg.equals("Database Configuration")) {
                    (new DatabaseManagerWindow(this)).setVisible(true);

                } else if (arg.equals("Show Derived Metric Panel")) {
                    if (showApplyOperationItem.isSelected()) {

                        this.getContentPane().removeAll();

                        jSplitOuterPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, jSplitInnerPane, derivedMetricPanel);
                        this.getContentPane().add(jSplitOuterPane, "Center");

                        this.validate();
                        jSplitOuterPane.setDividerLocation(0.75);

                    } else {

                        double dividerLocation = jSplitInnerPane.getDividerLocation();
                        this.getContentPane().removeAll();

                        jSplitInnerPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, treeScrollPane, getPanelHelpMessage(0));
                        jSplitInnerPane.setContinuousLayout(true);

                        this.getContentPane().add(jSplitInnerPane, "Center");

                        this.validate();
                        jSplitInnerPane.setDividerLocation(dividerLocation / this.getWidth());

                        jSplitOuterPane.setDividerLocation(1.00);
                    }
                } else if (arg.equals("About ParaProf")) {
                    ImageIcon icon = Utility.getImageIconResource("tau-medium.png");
                    JOptionPane.showMessageDialog(this, ParaProf.getInfoString(), "About ParaProf",
                            JOptionPane.INFORMATION_MESSAGE, icon);
                } else if (arg.equals("Show Help Window")) {
                    ParaProf.getHelpWindow().setVisible(true);
                    //Clear the window first.
                    ParaProf.getHelpWindow().clearText();
                    ParaProf.getHelpWindow().writeText("This is ParaProf's manager window!");
                    ParaProf.getHelpWindow().writeText("");
                    ParaProf.getHelpWindow().writeText(
                            "This window allows you to manage all of ParaProf's data sources,"
                                    + " including loading data from local files, or from a database."
                                    + " We also support the generation of derived metrics. Please see the"
                                    + " items below for more help.");
                    ParaProf.getHelpWindow().writeText("");
                    ParaProf.getHelpWindow().writeText("------------------");
                    ParaProf.getHelpWindow().writeText("");

                    ParaProf.getHelpWindow().writeText(
                            "1) Navigation:" + " The window is split into two halves, the left side gives a tree representation"
                                    + " of all data. The right side gives information about items clicked on in the left"
                                    + " half. You can also update information in the right half by double clicking in"
                                    + " the fields, and entering new data.  This automatically updates the left half."
                                    + " Right-clicking on the tree nodes in the left half displays popup menus which"
                                    + " allow you to add/delete applications, experiments, or trials.");
                    ParaProf.getHelpWindow().writeText("");
                    ParaProf.getHelpWindow().writeText(
                            "2) DB Configuration:" + " By default, ParaProf looks in the .ParaProf home directory in your home"
                                    + " directory for the database configuration file.  If that file is found, then"
                                    + " you are done, and can just expand the DB Applications node.  If there was a"
                                    + " problem finding the file, you can enter the location of the file by selecting"
                                    + " File -> Database Configuration.  You can also override the configuration file"
                                    + " password in the same manner.");
                    ParaProf.getHelpWindow().writeText("");
                    ParaProf.getHelpWindow().writeText(
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

                } else if (arg.equals("Add Application")) {
                    if (clickedOnObject == standard) {
                        ParaProfApplication application = addApplication(false, standard);
                        this.expandApplicationType(0, application.getID(), application);
                    } else {
                        ParaProfApplication application = addApplication(true, (DefaultMutableTreeNode) clickedOnObject);
                        this.expandApplicationType(2, application.getID(), application);
                    }
                } else if (arg.equals("Add Experiment")) {
                    if (clickedOnObject == standard) {
                        ParaProfApplication application = addApplication(false, standard);
                        ParaProfExperiment experiment = addExperiment(false, application);
                        if (application != null || experiment != null) {
                            this.expandApplicationType(0, application.getID(), application);
                            this.expandApplication(0, application, experiment);
                        }
                    } else if (clickedOnObject instanceof DefaultMutableTreeNode) {
                        ParaProfApplication application = addApplication(true, (DefaultMutableTreeNode) clickedOnObject);
                        ParaProfExperiment experiment = addExperiment(true, application);
                        if (application != null || experiment != null) {
                            this.expandApplicationType(2, application.getID(), application);
                            this.expandApplication(2, application, experiment);
                        }
                    } else if (clickedOnObject instanceof ParaProfApplication) {
                        ParaProfApplication application = (ParaProfApplication) clickedOnObject;
                        if (application.dBApplication()) {
                            ParaProfExperiment experiment = addExperiment(true, application);
                            if (experiment != null)
                                this.expandApplication(2, application, experiment);
                        } else {
                            ParaProfExperiment experiment = addExperiment(false, application);
                            if (experiment != null)
                                this.expandApplication(0, application, experiment);
                        }
                    }
                } else if (arg.equals("Add Trial")) {
                    if (clickedOnObject == standard) {
                        ParaProfApplication application = addApplication(false, standard);
                        if (application != null) {
                            this.expandApplicationType(0, application.getID(), application);
                            ParaProfExperiment experiment = addExperiment(false, application);
                            if (experiment != null) {
                                this.expandApplication(0, application, experiment);
                                (new LoadTrialWindow(this, application, experiment, true, true)).setVisible(true);
                            }
                        }
                    } else if (clickedOnObject instanceof ParaProfApplication) {
                        ParaProfApplication application = (ParaProfApplication) clickedOnObject;
                        if (application.dBApplication()) {
                            ParaProfExperiment experiment = addExperiment(true, application);
                            if (experiment != null) {
                                this.expandApplication(2, application, experiment);
                                (new LoadTrialWindow(this, null, experiment, false, true)).setVisible(true);
                            }
                        } else {
                            ParaProfExperiment experiment = addExperiment(false, application);
                            if (experiment != null) {
                                this.expandApplication(0, application, experiment);
                                (new LoadTrialWindow(this, null, experiment, false, true)).setVisible(true);
                            }
                        }
                    } else if (clickedOnObject instanceof ParaProfExperiment) {
                        ParaProfExperiment experiment = (ParaProfExperiment) clickedOnObject;
                        (new LoadTrialWindow(this, null, experiment, false, false)).setVisible(true);
                    } else {
                        // a database
                        ParaProfApplication application = addApplication(true, (DefaultMutableTreeNode) clickedOnObject);
                        if (application != null) {
                            this.expandApplicationType(2, application.getID(), application);
                            ParaProfExperiment experiment = addExperiment(true, application);
                            if (experiment != null) {
                                this.expandApplication(2, application, experiment);
                                (new LoadTrialWindow(this, application, experiment, true, true)).setVisible(true);
                            }
                        }

                    }
                } else if (arg.equals("Upload Application to DB")) {

                    java.lang.Thread thread = new java.lang.Thread(new Runnable() {
                        public void run() {
                            try {
                                ParaProfApplication clickedOnApp = (ParaProfApplication) clickedOnObject;
                                uploadApplication(clickedOnApp, true, true);
                            } catch (final Exception e) {
                                EventQueue.invokeLater(new Runnable() {
                                    public void run() {
                                        ParaProfUtils.handleException(e);
                                    }
                                });
                            }
                        }
                    });
                    thread.start();
                } else if (arg.equals("Export Application to Filesystem")) {
                    //???

                    ParaProfApplication dbApp = (ParaProfApplication) clickedOnObject;

                    DatabaseAPI databaseAPI = this.getDatabaseAPI(dbApp.getDatabase());
                    if (databaseAPI != null) {

                        boolean found = false;
                        //ListIterator l = databaseAPI.getApplicationList().listIterator();
                        //while (l.hasNext()) {
                        //    ParaProfApplication dbApp = new ParaProfApplication((Application) l.next());

                        String appname = dbApp.getName().replace('/', '%');
                        boolean success = (new File(appname).mkdir());

                        databaseAPI.setApplication(dbApp);
                        for (Iterator it = databaseAPI.getExperimentList().iterator(); it.hasNext();) {
                            ParaProfExperiment dbExp = new ParaProfExperiment((Experiment) it.next());

                            String expname = appname + "/" + dbExp.getName().replace('/', '%');
                            success = (new File(expname).mkdir());

                            databaseAPI.setExperiment(dbExp);
                            for (Iterator it2 = databaseAPI.getTrialList().iterator(); it2.hasNext();) {
                                Trial trial = (Trial) it2.next();

                                databaseAPI.setTrial(trial.getID());
                                DBDataSource dbDataSource = new DBDataSource(databaseAPI);
                                dbDataSource.load();

                                String filename = expname + "/" + trial.getName().replace('/', '%') + ".ppk";

                                DataSourceExport.writePacked(dbDataSource, new File(filename));

                            }

                        }

                        //  }
                    }

                } else if (arg.equals("Upload Experiment to DB")) {
                    java.lang.Thread thread = new java.lang.Thread(new Runnable() {
                        public void run() {
                            try {
                                ParaProfExperiment clickedOnExp = (ParaProfExperiment) clickedOnObject;
                                uploadExperiment(null, clickedOnExp, true, true);
                            } catch (final Exception e) {
                                EventQueue.invokeLater(new Runnable() {
                                    public void run() {
                                        ParaProfUtils.handleException(e);
                                    }
                                });
                            }
                        }
                    });
                    thread.start();

                } else if (arg.equals("Upload Trial to DB")) {
                    java.lang.Thread thread = new java.lang.Thread(new Runnable() {
                        public void run() {
                            try {
                                uploadTrial(null, (ParaProfTrial) clickedOnObject);
                            } catch (final Exception e) {
                                EventQueue.invokeLater(new Runnable() {
                                    public void run() {
                                        ParaProfUtils.handleException(e);
                                    }
                                });
                            }
                        }
                    });
                    thread.start();

                } else if (arg.equals("Add Mean to Comparison Window")) {
                    ParaProfTrial ppTrial = (ParaProfTrial) clickedOnObject;
                    if (ppTrial.loading()) {
                        JOptionPane.showMessageDialog(this, "Cannot perform operation while loading");
                    } else {
                        boolean loaded = true;

                        if (ppTrial.dBTrial()) {
                            loaded = false;
                            for (Enumeration e = loadedDBTrials.elements(); e.hasMoreElements();) {
                                ParaProfTrial loadedTrial = (ParaProfTrial) e.nextElement();
                                if ((ppTrial.getID() == loadedTrial.getID())
                                        && (ppTrial.getExperimentID() == loadedTrial.getExperimentID())
                                        && (ppTrial.getApplicationID() == loadedTrial.getApplicationID())) {
                                    loaded = true;
                                }
                            }
                        }

                        if (!loaded) {
                            JOptionPane.showMessageDialog(this, "Please load the trial first (expand the tree)");
                        } else {
                            if (ParaProf.theComparisonWindow == null) {
                                ParaProf.theComparisonWindow = FunctionBarChartWindow.CreateComparisonWindow(ppTrial,
                                        ppTrial.getDataSource().getMeanData(), this);
                            } else {
                                ParaProf.theComparisonWindow.addThread(ppTrial, ppTrial.getDataSource().getMeanData());
                            }
                            ParaProf.theComparisonWindow.setVisible(true);
                        }
                    }

                } else if (arg.equals("Export Profile")) {
                    ParaProfTrial ppTrial = (ParaProfTrial) clickedOnObject;
                    if (ppTrial.loading()) {
                        JOptionPane.showMessageDialog(this, "Cannot export trial while loading");
                        return;
                    }
                    if (!isLoaded(ppTrial)) {
                        JOptionPane.showMessageDialog(this, "Please load the trial before exporting (expand the tree)");
                        return;
                    }

                    ParaProfUtils.exportTrial(ppTrial, this);

                } else if (arg.equals("Convert to Phase Profile")) {
                    ParaProfTrial ppTrial = (ParaProfTrial) clickedOnObject;
                    if (ppTrial.loading()) {
                        JOptionPane.showMessageDialog(this, "Cannot convert while loading");
                        return;
                    }
                    if (!isLoaded(ppTrial)) {
                        JOptionPane.showMessageDialog(this, "Please load the trial before converting (expand the tree)");
                        return;
                    }

                    ParaProfUtils.phaseConvertTrial(ppTrial, this);

                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    private ParaProfApplication uploadApplication(ParaProfApplication ppApp, boolean allowOverwrite, boolean uploadChildren)
            throws SQLException, DatabaseException {

        DatabaseAPI databaseAPI = this.getDatabaseAPI(null);
        if (databaseAPI != null) {

            boolean found = false;
            ListIterator l = databaseAPI.getApplicationList().listIterator();

            while (l.hasNext()) {
                ParaProfApplication dbApp = new ParaProfApplication((Application) l.next());

                if (dbApp.getName().equals(ppApp.getName())) {
                    found = true;

                    if (allowOverwrite) {
                        String options[] = { "Overwrite", "Don't overwrite", "Cancel" };
                        int value = JOptionPane.showOptionDialog(
                                this,
                                "An Application with the name \""
                                        + ppApp.getName()
                                        + "\" already exists, it will be updated new experiments/trials, should the metadata be overwritten?",
                                "Upload Application", JOptionPane.YES_NO_OPTION, // Need something  
                                JOptionPane.QUESTION_MESSAGE, null, // Use default icon for message type
                                options, options[1]);
                        if (value == JOptionPane.CLOSED_OPTION || value == JOptionPane.CANCEL_OPTION) {
                            return null;
                        } else {
                            if (value == 0) {
                                // overwrite the metadata
                                dbApp.setFields(ppApp.getFields());
                                databaseAPI.saveApplication(dbApp);
                            }

                            if (uploadChildren) {
                                for (Iterator it2 = ppApp.getExperimentList(); it2.hasNext();) {
                                    ParaProfExperiment ppExp = (ParaProfExperiment) it2.next();
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
                newApp.setID(-1); // must set the ID to -1 to indicate that this is a new application (bug found by Sameer on 2005-04-19)
                ParaProfApplication application = new ParaProfApplication(newApp);
                application.setDBApplication(true);
                application.setID(databaseAPI.saveApplication(application));

                if (uploadChildren) {
                    for (Iterator it2 = ppApp.getExperimentList(); it2.hasNext();) {
                        ParaProfExperiment ppExp = (ParaProfExperiment) it2.next();
                        uploadExperiment(application, ppExp, true, true);
                    }
                }

                return application;
            }
        }
        return null;
    }

    private ParaProfExperiment uploadExperiment(ParaProfApplication dbApp, ParaProfExperiment ppExp, boolean allowOverwrite,
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
        ListIterator l = databaseAPI.getExperimentList().listIterator();
        while (l.hasNext()) {
            ParaProfExperiment dbExp = new ParaProfExperiment((Experiment) l.next());

            if (dbExp.getName().equals(ppExp.getName())) {
                found = true;

                if (allowOverwrite) {

                    String options[] = { "Overwrite", "Don't overwrite", "Cancel" };
                    int value = JOptionPane.showOptionDialog(this, "An Experiment with the name \"" + ppExp.getName()
                            + "\" already exists, it will be updated new trials, should the metadata be overwritten?",
                            "Upload Application", JOptionPane.YES_NO_OPTION, // Need something  
                            JOptionPane.QUESTION_MESSAGE, null, // Use default icon for message type
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
                            for (Iterator it2 = ppExp.getTrialList(); it2.hasNext();) {
                                ParaProfTrial ppTrial = (ParaProfTrial) it2.next();
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
            newExp.setID(-1); // must set the ID to -1 to indicate that this is a new application (bug found by Sameer on 2005-04-19)
            experiment.setDBExperiment(true);
            experiment.setApplicationID(dbApp.getID());
            experiment.setApplication(dbApp);
            experiment.setID(databaseAPI.saveExperiment(experiment));

            if (uploadChildren) {
                for (Iterator it2 = ppExp.getTrialList(); it2.hasNext();) {
                    ParaProfTrial ppTrial = (ParaProfTrial) it2.next();
                    uploadTrial(experiment, ppTrial);
                }
            }

            return experiment;
        }
        return null;
    }

    private ParaProfTrial uploadTrial(ParaProfExperiment dbExp, ParaProfTrial ppTrial) throws SQLException, DatabaseException {
        DatabaseAPI databaseAPI = this.getDatabaseAPI(null);
        if (databaseAPI == null)
            return null;

        if (dbExp == null) {
            dbExp = uploadExperiment(null, ppTrial.getExperiment(), false, false);
        }

        ParaProfTrial dbTrial = new ParaProfTrial(ppTrial.getTrial());
        dbTrial.setID(-1);
        dbTrial.setExperimentID(dbExp.getID());
        dbTrial.setApplicationID(dbExp.getApplicationID());
        dbTrial.getTrial().setDataSource(ppTrial.getDataSource());
        dbTrial.setExperiment(dbExp);

        dbTrial.setUpload(true); // This trial is not set to a db trial until after it has finished loading.

        LoadTrialProgressWindow lpw = new LoadTrialProgressWindow(this, dbTrial.getDataSource(), dbTrial, true);
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
            DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
            DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
            Object userObject = selectedNode.getUserObject();

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

    private void clearDefaultMutableTreeNodes(DefaultMutableTreeNode defaultMutableTreeNode) {
        int count = defaultMutableTreeNode.getChildCount();
        for (int i = 0; i < count; i++) {
            DefaultMutableTreeNode child = (DefaultMutableTreeNode) defaultMutableTreeNode.getChildAt(i);
            ((ParaProfTreeNodeUserObject) child.getUserObject()).clearDefaultMutableTreeNode();
            clearDefaultMutableTreeNodes((DefaultMutableTreeNode) defaultMutableTreeNode.getChildAt(i));
        }
    }

    public void treeWillCollapse(TreeExpansionEvent event) {
        try {
            TreePath path = event.getPath();
            if (path == null) {
                return;
            }
            DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
            DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
            Object userObject = selectedNode.getUserObject();

            if (selectedNode.isRoot()) {
                clearDefaultMutableTreeNodes(standard);
                //clearDefaultMutableTreeNodes(runtime);
                //clearDefaultMutableTreeNodes(dbApps);
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
        System.out.println("treeWillExpand: " + event);
        TreePath path = event.getPath();
        DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();

        if (selectedNode.isRoot()) {
            return;
        }

    }

    public void treeWillExpand(TreeExpansionEvent event) {
        try {
            TreePath path = event.getPath();
            if (path == null) {
                return;
            }
            DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
            DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
            Object userObject = selectedNode.getUserObject();

            if (selectedNode.isRoot()) {
                //Do not need to do anything here.
                return;
            }

            if ((parentNode.isRoot())) {
                int location = jSplitInnerPane.getDividerLocation();
                if (selectedNode == standard) {
                    jSplitInnerPane.setRightComponent(getPanelHelpMessage(1));

                    // refresh the application list
                    for (int i = standard.getChildCount(); i > 0; i--) {
                        treeModel.removeNodeFromParent(((DefaultMutableTreeNode) standard.getChildAt(i - 1)));
                    }
                    Iterator l = ParaProf.applicationManager.getApplications().iterator();
                    while (l.hasNext()) {
                        ParaProfApplication application = (ParaProfApplication) l.next();
                        DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
                        application.setDMTN(applicationNode);
                        treeModel.insertNodeInto(applicationNode, standard, standard.getChildCount());
                    }
                } else if (selectedNode == runtime) {
                    jSplitInnerPane.setRightComponent(getPanelHelpMessage(2));
                    //                } else if (selectedNode == dbApps) {
                    //                    jSplitInnerPane.setRightComponent(getPanelHelpMessage(3));
                    //
                    //                    // refresh the application list
                    //                    for (int i = dbApps.getChildCount(); i > 0; i--) {
                    //                        treeModel.removeNodeFromParent(((DefaultMutableTreeNode) dbApps.getChildAt(i - 1)));
                    //                    }
                    //                    DatabaseAPI databaseAPI = getDatabaseAPI();
                    //                    if (databaseAPI != null) {
                    //                        ListIterator l = databaseAPI.getApplicationList().listIterator();
                    //                        while (l.hasNext()) {
                    //                            ParaProfApplication application = new ParaProfApplication((Application) l.next());
                    //                            application.setDBApplication(true);
                    //                            DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
                    //                            application.setDMTN(applicationNode);
                    //                            treeModel.insertNodeInto(applicationNode, dbApps, dbApps.getChildCount());
                    //                        }
                    //                        databaseAPI.terminate();
                    //                    }
                } else {
                    jSplitInnerPane.setRightComponent(getPanelHelpMessage(3));

                    // refresh the application list
                    for (int i = selectedNode.getChildCount(); i > 0; i--) {
                        treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i - 1)));
                    }

                    Database database = (Database) userObject;
                    DatabaseAPI databaseAPI = getDatabaseAPI(database);
                    if (databaseAPI != null) {
                        ListIterator l = databaseAPI.getApplicationList().listIterator();
                        while (l.hasNext()) {
                            ParaProfApplication application = new ParaProfApplication((Application) l.next());
                            application.setDBApplication(true);
                            DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
                            application.setDMTN(applicationNode);
                            treeModel.insertNodeInto(applicationNode, selectedNode, selectedNode.getChildCount());
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
                        treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i - 1)));
                    }

                    // reload from the database
                    DatabaseAPI databaseAPI = this.getDatabaseAPI(application.getDatabase());
                    if (databaseAPI != null) {
                        databaseAPI.setApplication(application.getID());
                        ListIterator l = databaseAPI.getExperimentList().listIterator();
                        while (l.hasNext()) {
                            ParaProfExperiment experiment = new ParaProfExperiment((Experiment) l.next());
                            experiment.setDBExperiment(true);
                            experiment.setApplication(application);
                            DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(experiment);
                            experiment.setDMTN(experimentNode);
                            treeModel.insertNodeInto(experimentNode, selectedNode, selectedNode.getChildCount());
                        }
                        databaseAPI.terminate();
                    }

                } else {
                    for (int i = selectedNode.getChildCount(); i > 0; i--) {
                        treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i - 1)));
                    }
                    ListIterator l = application.getExperimentList();
                    while (l.hasNext()) {
                        ParaProfExperiment experiment = (ParaProfExperiment) l.next();
                        DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(experiment);
                        experiment.setDMTN(experimentNode);
                        treeModel.insertNodeInto(experimentNode, selectedNode, selectedNode.getChildCount());
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
                        treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i - 1)));
                    }
                    experiment.setApplication((ParaProfApplication) parentNode.getUserObject());
                    DatabaseAPI databaseAPI = this.getDatabaseAPI(experiment.getDatabase());
                    if (databaseAPI != null) {
                        databaseAPI.setExperiment(experiment.getID());
                        if (databaseAPI.getTrialList() != null) {
                            ListIterator l = databaseAPI.getTrialList().listIterator();
                            while (l.hasNext()) {
                                ParaProfTrial ppTrial = new ParaProfTrial((Trial) l.next());
                                ppTrial.setDBTrial(true);
                                ppTrial.setExperiment(experiment);
                                DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(ppTrial);
                                ppTrial.setDMTN(trialNode);
                                treeModel.insertNodeInto(trialNode, selectedNode, selectedNode.getChildCount());
                            }
                        }
                        databaseAPI.terminate();
                    }
                } else {
                    for (int i = selectedNode.getChildCount(); i > 0; i--) {
                        treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i - 1)));
                    }
                    ListIterator l = experiment.getTrialList();
                    while (l.hasNext()) {
                        ParaProfTrial ppTrial = (ParaProfTrial) l.next();
                        DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(ppTrial);
                        ppTrial.setDMTN(trialNode);
                        treeModel.insertNodeInto(trialNode, selectedNode, selectedNode.getChildCount());
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

    private void trialWillExpand(DefaultMutableTreeNode selectedNode) {
        Object userObject = selectedNode.getUserObject();

        ParaProfTrial ppTrial = (ParaProfTrial) userObject;
        if (ppTrial.dBTrial()) {

            // test to see if trial has already been loaded
            // if so, we re-associate the ParaProfTrial with the DMTN since
            // the old one is gone
            boolean loaded = false;
            for (Enumeration e = loadedDBTrials.elements(); e.hasMoreElements();) {
                ParaProfTrial loadedTrial = (ParaProfTrial) e.nextElement();
                if ((ppTrial.getID() == loadedTrial.getID()) && (ppTrial.getExperimentID() == loadedTrial.getExperimentID())
                        && (ppTrial.getApplicationID() == loadedTrial.getApplicationID())) {
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

                DatabaseAPI databaseAPI = this.getDatabaseAPI(ppTrial.getDatabase());
                if (databaseAPI != null) {
                    databaseAPI.setApplication(ppTrial.getApplicationID());
                    databaseAPI.setExperiment(ppTrial.getExperimentID());
                    databaseAPI.setTrial(ppTrial.getID());

                    DBDataSource dbDataSource = new DBDataSource(databaseAPI);
                    dbDataSource.setGenerateIntermediateCallPathData(ParaProf.preferences.getGenerateIntermediateCallPathData());
                    ppTrial.getTrial().setDataSource(dbDataSource);
                    final DataSource dataSource = dbDataSource;
                    final ParaProfTrial theTrial = ppTrial;
                    java.lang.Thread thread = new java.lang.Thread(new Runnable() {

                        public void run() {
                            try {
                                dataSource.load();
                                theTrial.finishLoad();
                                ParaProf.paraProfManagerWindow.populateTrialMetrics(theTrial);
                            } catch (final Exception e) {
                                EventQueue.invokeLater(new Runnable() {
                                    public void run() {
                                        ParaProfUtils.handleException(e);
                                    }
                                });
                            }
                        }
                    });
                    thread.start();

                    //Add to the list of loaded trials.
                    loadedDBTrials.add(ppTrial);
                }
            }
        }

        // at this point, in both the db and non-db cases, the trial
        // is either loading or not. Check this before displaying
        if (!ppTrial.loading()) {
            // refresh the metrics list
            for (int i = selectedNode.getChildCount(); i > 0; i--) {
                treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i - 1)));
            }
            Iterator l = ppTrial.getMetrics().iterator();
            while (l.hasNext()) {
                ParaProfMetric metric = (ParaProfMetric) l.next();
                DefaultMutableTreeNode metricNode = new DefaultMutableTreeNode(metric, false);
                metric.setDMTN(metricNode);
                treeModel.insertNodeInto(metricNode, selectedNode, selectedNode.getChildCount());
            }

            int location = jSplitInnerPane.getDividerLocation();
            jSplitInnerPane.setRightComponent(getTable(userObject));
            jSplitInnerPane.setDividerLocation(location);
        }
    }

    private void showMetric(ParaProfMetric metric) {
        try {
            ParaProfTrial ppTrial = metric.getParaProfTrial();
            if (ppTrial.getDefaultMetricID() != metric.getID()) {
                ppTrial.setDefaultMetricID(metric.getID());
                ppTrial.updateRegisteredObjects("dataEvent");
            }
            ppTrial.showMainWindow();
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    private void metricSelected(ParaProfMetric metric, boolean show) {
        int location = jSplitInnerPane.getDividerLocation();
        jSplitInnerPane.setRightComponent(getTable(metric));
        jSplitInnerPane.setDividerLocation(location);
        this.operand2 = this.operand1;
        derivedMetricPanel.setArg2Field(derivedMetricPanel.getArg1Field());
        operand1 = metric;
        derivedMetricPanel.setArg1Field(metric.getApplicationID() + ":" + metric.getExperimentID() + ":" + metric.getTrialID()
                + ":" + metric.getID() + " - " + metric.getName());
        if (show) {
            this.showMetric(metric);
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
            DatabaseAPI databaseAPI = this.getDatabaseAPI(metric.getParaProfTrial().getDatabase());
            if (databaseAPI != null) {
                try {
                    databaseAPI.saveTrial(metric.getParaProfTrial().getTrial(), metric.getID());
                } catch (DatabaseException e) {
                    ParaProfUtils.handleException(e);
                }
                databaseAPI.terminate();
            }
        }
    }

    public int[] getSelectedDBExperiment() {
        if (ParaProf.preferences.getDatabaseConfigurationFile() == null || ParaProf.preferences.getDatabasePassword() == null) {
            // Check to see if the user has set configuration information.
            JOptionPane.showMessageDialog(this, "Please set the database configuration information (file menu).",
                    "DB Configuration Error!", JOptionPane.ERROR_MESSAGE);
            return null;
        }

        TreePath path = tree.getSelectionPath();
        boolean error = false;
        if (path == null)
            error = true;
        else {
            DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
            Object userObject = selectedNode.getUserObject();
            if (userObject instanceof ParaProfExperiment) {
                ParaProfExperiment paraProfExperiment = (ParaProfExperiment) userObject;
                if (paraProfExperiment.dBExperiment()) {
                    int[] array = new int[2];
                    array[0] = paraProfExperiment.getApplicationID();
                    array[1] = paraProfExperiment.getID();
                    return array;
                } else
                    error = true;
            } else
                error = true;
        }
        if (error)
            JOptionPane.showMessageDialog(this, "Please select an db experiment first!", "DB Upload Error!",
                    JOptionPane.ERROR_MESSAGE);
        return null;
    }

    private Component getPanelHelpMessage(int type) {
        JTextArea jTextArea = new JTextArea();
        jTextArea.setLineWrap(true);
        jTextArea.setWrapStyleWord(true);
        switch (type) {
        case 0:
            jTextArea.append("ParaProf Manager\n\n");
            jTextArea.append("This window allows you to manage all of ParaProf's loaded data.\n");
            jTextArea.append("Data can be static (ie, not updated at runtime),"
                    + " and loaded either remotely or locally.  You can also specify data to be uploaded at runtime.\n\n");
            break;
        case 1:
            jTextArea.append("ParaProf Manager\n\n");
            jTextArea.append("This is the Standard application section:\n\n");
            jTextArea.append("Standard - The classic ParaProf mode.  Data sets that are loaded at startup are placed"
                    + " under the default application automatically. Please see the ParaProf documentation for more details.\n");
            break;
        case 2:
            jTextArea.append("ParaProf Manager\n\n");
            jTextArea.append("This is the Runtime application section:\n\n");
            jTextArea.append("Runtime - A new feature allowing ParaProf to update data at runtime.  Please see"
                    + " the ParaProf documentation if the options are not clear.\n");
            jTextArea.append("*** THIS FEATURE IS CURRENTLY OFF ***\n");
            break;
        case 3:
            jTextArea.append("ParaProf Manager\n\n");
            jTextArea.append("This is the DB Apps application section:\n\n");
            jTextArea.append("DB Apps - ParaProf can load data from a database.  Please see"
                    + " the ParaProf and PerfDMF documentation for more details.\n");
            break;
        default:
            break;
        }
        return (new JScrollPane(jTextArea));
    }

    private Component getTable(Object obj) {
        if (obj instanceof ParaProfApplication) {
            return (new JScrollPane(new JTable(new ApplicationTableModel(this, (ParaProfApplication) obj, treeModel))));
        } else if (obj instanceof ParaProfExperiment) {
            return (new JScrollPane(new JTable(new ExperimentTableModel(this, (ParaProfExperiment) obj, treeModel))));
        } else if (obj instanceof ParaProfTrial) {
            return (new JScrollPane(new JTable(new TrialTableModel(this, (ParaProfTrial) obj, treeModel))));
        } else {
            return (new JScrollPane(new JTable(new MetricTableModel(this, (ParaProfMetric) obj, treeModel))));
        }
    }

    public ParaProfApplication addApplication(boolean dBApplication, DefaultMutableTreeNode treeNode) throws SQLException {
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

    public ParaProfExperiment addExperiment(boolean dBExperiment, ParaProfApplication application) {
        ParaProfExperiment experiment = null;
        if (dBExperiment) {
            DatabaseAPI databaseAPI = this.getDatabaseAPI(application.getDatabase());
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

    public void addTrial(ParaProfApplication application, ParaProfExperiment experiment, File files[], int fileType,
            boolean fixGprofNames, boolean monitorProfiles) {

        ParaProfTrial ppTrial = null;
        DataSource dataSource = null;

        try {
            dataSource = UtilFncs.initializeDataSource(files, fileType, fixGprofNames);
            dataSource.setGenerateIntermediateCallPathData(ParaProf.preferences.getGenerateIntermediateCallPathData());
        } catch (DataSourceException e) {

            if (files == null || files.length != 0) // We don't output an error message if paraprof was just invoked with no parameters.
                ParaProfUtils.handleException(e);

            return;
        }

        ppTrial = new ParaProfTrial();
        // this must be done before setting the monitored flag
        ppTrial.getTrial().setDataSource(dataSource);
        ppTrial.setLoading(true);

        ppTrial.setMonitored(monitorProfiles);

        ppTrial.setExperiment(experiment);
        ppTrial.setApplicationID(experiment.getApplicationID());
        ppTrial.setExperimentID(experiment.getID());
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
        if (experiment.dBExperiment()) {
            loadedDBTrials.add(ppTrial);
            ppTrial.setUpload(true); // This trial is not set to a db trial until after it has finished loading.
        } else {
            experiment.addTrial(ppTrial);
        }

        //        if (experiment.dBExperiment()) // Check needs to occur on the experiment as trial 
        //            // not yet a recognized db trial.
        //            this.expandTrial(2, ppTrial.getApplicationID(), ppTrial.getExperimentID(), ppTrial.getID(),
        //                    application, experiment, ppTrial);
        //        else
        //            this.expandTrial(0, ppTrial.getApplicationID(), ppTrial.getExperimentID(), ppTrial.getID(),
        //                    application, experiment, ppTrial);
        //  
        //        
        LoadTrialProgressWindow lpw = new LoadTrialProgressWindow(this, dataSource, ppTrial, false);
        lpw.show();

    }

    public void populateTrialMetrics(final ParaProfTrial ppTrial) {
        try {
            loadedTrials.add(ppTrial);

            EventQueue.invokeLater(new Runnable() {
                public void run() {
                    try {
                        if (ppTrial.upload()) {
                            //Add to the list of loaded trials.
                            ppTrial.setUpload(false);
                        }

                        expandTrial(ppTrial);

                        //                        if (ppTrial.dBTrial()) {
                        //                            expandTrial(2, ppTrial.getApplicationID(), ppTrial.getExperimentID(), ppTrial.getID(), null, null,
                        //                                    ppTrial);
                        //                        } else {
                        //
                        //                            //recurseExpand(ppTrial.getDMTN());
                        //                            //                            DefaultMutableTreeNode node = ppTrial.getDMTN();
                        //                            //                            while (node != null && !node.isRoot()) {
                        //                            //                                tree.expandPath(new TreePath(node));
                        //                            //                                node = (DefaultMutableTreeNode) node.getParent();
                        //                            //                            }
                        //
                        //                            expandTrial(0, ppTrial.getApplicationID(), ppTrial.getExperimentID(), ppTrial.getID(), null, null,
                        //                                    ppTrial);
                        //                        }

                    } catch (Exception e) {
                        ParaProfUtils.handleException(e);
                    }
                }
            });

        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    private void recurseExpand(DefaultMutableTreeNode node) {
        if (node == null || node.isRoot()) {
            return;
        }
        recurseExpand((DefaultMutableTreeNode) node.getParent());
        tree.expandPath(new TreePath(node));

    }

    public DefaultMutableTreeNode expandApplicationType(int type, int applicationID, ParaProfApplication application) {
        switch (type) {
        case 0:
            //Test to see if standard is expanded, if not, expand it.
            if (!(tree.isExpanded(new TreePath(standard.getPath()))))
                tree.expandPath(new TreePath(standard.getPath()));

            //Try and find the required application node.
            for (int i = standard.getChildCount(); i > 0; i--) {
                DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) standard.getChildAt(i - 1);
                if (applicationID == ((ParaProfApplication) defaultMutableTreeNode.getUserObject()).getID())
                    return defaultMutableTreeNode;
            }
            //Required application node was not found, try adding it.
            if (application != null) {
                DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
                application.setDMTN(applicationNode);
                treeModel.insertNodeInto(applicationNode, standard, standard.getChildCount());
                return applicationNode;
            }
            return null;

        case 2:
            //            //Test to see if dbApps is expanded, if not, expand it.
            //            if (!(tree.isExpanded(new TreePath(dbApps.getPath()))))
            //                tree.expandPath(new TreePath(dbApps.getPath()));
            //
            //            //Try and find the required application node.
            //            for (int i = dbApps.getChildCount(); i > 0; i--) {
            //                DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) dbApps.getChildAt(i - 1);
            //                if (applicationID == ((ParaProfApplication) defaultMutableTreeNode.getUserObject()).getID())
            //                    return defaultMutableTreeNode;
            //            }
            //            //Required application node was not found, try adding it.
            //            if (application != null) {
            //                DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
            //                application.setDMTN(applicationNode);
            //                treeModel.insertNodeInto(applicationNode, dbApps, dbApps.getChildCount());
            //                return applicationNode;
            //            }
            //            return null;

            DefaultMutableTreeNode dbNode = null;
            Database db = null;
            try {
                for (int i = 0; i < root.getChildCount(); i++) {
                    DefaultMutableTreeNode node = (DefaultMutableTreeNode) root.getChildAt(i);
                    if (node.getUserObject() == application.getDatabase()) {
                        dbNode = node;
                        db = (Database) node.getUserObject();
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

            //Test to see if dbApps is expanded, if not, expand it.
            if (!(tree.isExpanded(new TreePath(dbNode.getPath()))))
                tree.expandPath(new TreePath(dbNode.getPath()));

            //Try and find the required application node.
            for (int i = dbNode.getChildCount(); i > 0; i--) {
                DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) dbNode.getChildAt(i - 1);
                if (applicationID == ((ParaProfApplication) defaultMutableTreeNode.getUserObject()).getID())
                    return defaultMutableTreeNode;
            }
            //Required application node was not found, try adding it.
            if (application != null) {
                DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
                application.setDMTN(applicationNode);
                treeModel.insertNodeInto(applicationNode, dbNode, dbNode.getChildCount());
                return applicationNode;
            }
            return null;

        default:
            break;
        }
        return null;
    }

    public DefaultMutableTreeNode expandApplicationType(Database database, ParaProfApplication application) {

        if (database == null) {
            //Test to see if standard is expanded, if not, expand it.
            if (!(tree.isExpanded(new TreePath(standard.getPath()))))
                tree.expandPath(new TreePath(standard.getPath()));

            //Try and find the required application node.
            for (int i = standard.getChildCount(); i > 0; i--) {
                DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) standard.getChildAt(i - 1);
                if (application.getID() == ((ParaProfApplication) defaultMutableTreeNode.getUserObject()).getID())
                    return defaultMutableTreeNode;
            }
            //Required application node was not found, try adding it.
            if (application != null) {
                DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
                application.setDMTN(applicationNode);
                treeModel.insertNodeInto(applicationNode, standard, standard.getChildCount());
                return applicationNode;
            }
            return null;

        } else {

            DefaultMutableTreeNode dbNode = null;
            Database db = null;
            for (int i = 0; i < root.getChildCount(); i++) {
                DefaultMutableTreeNode node = (DefaultMutableTreeNode) root.getChildAt(i);
                if (node.getUserObject() == application.getDatabase()) {
                    dbNode = node;
                    db = (Database) node.getUserObject();
                }
            }

            //Test to see if dbApps is expanded, if not, expand it.
            if (!(tree.isExpanded(new TreePath(dbNode.getPath()))))
                tree.expandPath(new TreePath(dbNode.getPath()));

            //Try and find the required application node.
            for (int i = dbNode.getChildCount(); i > 0; i--) {
                DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) dbNode.getChildAt(i - 1);
                if (application.getID() == ((ParaProfApplication) defaultMutableTreeNode.getUserObject()).getID())
                    return defaultMutableTreeNode;
            }
            //Required application node was not found, try adding it.
            if (application != null) {
                DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
                application.setDMTN(applicationNode);
                treeModel.insertNodeInto(applicationNode, dbNode, dbNode.getChildCount());
                return applicationNode;
            }
            return null;

        }
    }

    //Expands the given application
    public DefaultMutableTreeNode expandApplication(int type, ParaProfApplication application, ParaProfExperiment experiment) {
        DefaultMutableTreeNode applicationNode = this.expandApplicationType(type, application.getID(), application);
        if (applicationNode != null) {
            //Expand the application.
            tree.expandPath(new TreePath(applicationNode.getPath()));

            //Try and find the required experiment node.
            tree.expandPath(new TreePath(standard.getPath()));
            for (int i = applicationNode.getChildCount(); i > 0; i--) {
                DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) applicationNode.getChildAt(i - 1);
                if (experiment.getID() == ((ParaProfExperiment) defaultMutableTreeNode.getUserObject()).getID())
                    return defaultMutableTreeNode;
            }
            //Required experiment node was not found, try adding it.
            if (experiment != null) {
                DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(experiment);
                experiment.setDMTN(experimentNode);
                treeModel.insertNodeInto(experimentNode, applicationNode, applicationNode.getChildCount());
                return experimentNode;
            }
            return null;
        }
        return null;
    }

    //Expands the given application
    public DefaultMutableTreeNode expandApplication(ParaProfApplication application, ParaProfExperiment experiment) {
        DefaultMutableTreeNode applicationNode = this.expandApplicationType(application.getDatabase(), application);
        if (applicationNode != null) {
            //Expand the application.
            tree.expandPath(new TreePath(applicationNode.getPath()));

            //Try and find the required experiment node.
            tree.expandPath(new TreePath(standard.getPath()));
            for (int i = applicationNode.getChildCount(); i > 0; i--) {
                DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) applicationNode.getChildAt(i - 1);
                if (experiment.getID() == ((ParaProfExperiment) defaultMutableTreeNode.getUserObject()).getID())
                    return defaultMutableTreeNode;
            }
            //Required experiment node was not found, try adding it.
            if (experiment != null) {
                DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(experiment);
                experiment.setDMTN(experimentNode);
                treeModel.insertNodeInto(experimentNode, applicationNode, applicationNode.getChildCount());
                return experimentNode;
            }
            return null;
        }
        return null;
    }

    public DefaultMutableTreeNode expandExperiment(ParaProfExperiment experiment, ParaProfTrial ppTrial) {
        ParaProfApplication app = (ParaProfApplication) experiment.getApplication();
        if (app == null)
            app = new ParaProfApplication();
        DefaultMutableTreeNode experimentNode = this.expandApplication(app, experiment);
        if (experimentNode != null) {
            //Expand the experiment.
            tree.expandPath(new TreePath(experimentNode.getPath()));

            //Try and find the required trial node.
            for (int i = experimentNode.getChildCount(); i > 0; i--) {
                DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) experimentNode.getChildAt(i - 1);
                if (ppTrial.getID() == ((ParaProfTrial) defaultMutableTreeNode.getUserObject()).getID())
                    return defaultMutableTreeNode;
            }
            //Required trial node was not found, try adding it.
            if (ppTrial != null) {
                DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(ppTrial);
                ppTrial.setDMTN(trialNode);
                treeModel.insertNodeInto(trialNode, experimentNode, experimentNode.getChildCount());
                return trialNode;
            }
            return null;
        }
        return null;
    }

    public DefaultMutableTreeNode expandExperiment(int type, int applicationID, int experimentID, int trialID,
            ParaProfApplication application, ParaProfExperiment experiment, ParaProfTrial ppTrial) {
        DefaultMutableTreeNode experimentNode = this.expandApplication(type, application, experiment);
        if (experimentNode != null) {
            //Expand the experiment.
            tree.expandPath(new TreePath(experimentNode.getPath()));

            //Try and find the required trial node.
            for (int i = experimentNode.getChildCount(); i > 0; i--) {
                DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) experimentNode.getChildAt(i - 1);
                if (trialID == ((ParaProfTrial) defaultMutableTreeNode.getUserObject()).getID())
                    return defaultMutableTreeNode;
            }
            //Required trial node was not found, try adding it.
            if (ppTrial != null) {
                DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(ppTrial);
                ppTrial.setDMTN(trialNode);
                treeModel.insertNodeInto(trialNode, experimentNode, experimentNode.getChildCount());
                return trialNode;
            }
            return null;
        }
        return null;
    }

    public void expandTrial(ParaProfTrial ppTrial) {
        DefaultMutableTreeNode trialNode = this.expandExperiment(ppTrial.getExperiment(), ppTrial);
        //Expand the trial.
        if (trialNode != null) {
            if (tree.isExpanded(new TreePath(trialNode.getPath())))
                tree.collapsePath(new TreePath(trialNode.getPath()));
            tree.expandPath(new TreePath(trialNode.getPath()));
        }
    }

    public void expandTrial(int type, int applicationID, int experimentID, int trialID, ParaProfApplication application,
            ParaProfExperiment experiment, ParaProfTrial ppTrial) {
        DefaultMutableTreeNode trialNode = this.expandExperiment(type, applicationID, experimentID, trialID, application,
                experiment, ppTrial);
        //Expand the trial.
        if (trialNode != null) {
            if (tree.isExpanded(new TreePath(trialNode.getPath())))
                tree.collapsePath(new TreePath(trialNode.getPath()));
            tree.expandPath(new TreePath(trialNode.getPath()));
        }
    }

    //    public ConnectionManager getConnectionManager() {
    //        //Check to see if the user has set configuration information.
    //        if (ParaProf.savedPreferences.getDatabaseConfigurationFile() == null) {
    //            JOptionPane.showMessageDialog(this,
    //                    "Please set the database configuration information (file menu).",
    //                    "DB Configuration Error!", JOptionPane.ERROR_MESSAGE);
    //            return null;
    //        } else {//Test to see if configurataion file exists.
    //            File file = new File(ParaProf.savedPreferences.getDatabaseConfigurationFile());
    //            if (!file.exists()) {
    //                JOptionPane.showMessageDialog(this, "Specified configuration file does not exist.",
    //                        "DB Configuration Error!", JOptionPane.ERROR_MESSAGE);
    //                return null;
    //            }
    //        }
    //        //Basic checks done, try to access the db.
    //        if (ParaProf.savedPreferences.getDatabasePassword() == null) {
    //            try {
    //                return new ConnectionManager(ParaProf.savedPreferences.getDatabaseConfigurationFile(), false);
    //            } catch (Exception e) {
    //            }
    //        } else {
    //            try {
    //                return new ConnectionManager(ParaProf.savedPreferences.getDatabaseConfigurationFile(),
    //                        ParaProf.savedPreferences.getDatabasePassword());
    //            } catch (Exception e) {
    //            }
    //        }
    //        return null;
    //    }

    // private DatabaseAPI dbAPI = null;

    public String getDatabaseName() {
        if (dbDisplayName == null) {
            try {
                if (ParaProf.preferences.getDatabaseConfigurationFile() == null) {
                    dbDisplayName = "";
                }

                ParseConfig parser = new ParseConfig(ParaProf.preferences.getDatabaseConfigurationFile());
                //dbDisplayName = "[" + parser.getDBHost() + " (" + parser.getDBType() + ")]";
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
        return (Database) databases.get(0);
        //ParseConfig config = new ParseConfig(ParaProf.preferences.getDatabaseConfigurationFile());
        //return new Database("default", config);
    }

    public DatabaseAPI getDatabaseAPI(Database database) {
        try {

            if (database == null) {
                database = getDefaultDatabase();
            }

            //Basic checks done, try to access the db.
            DatabaseAPI databaseAPI = new DatabaseAPI();
            databaseAPI.initialize(database);

            if (!metaDataRetrieved) {
                DatabaseAPI defaultDatabaseAPI = new DatabaseAPI();
                defaultDatabaseAPI.initialize(getDefaultDatabase());
                metaDataRetrieved = true;
                for (Iterator it = ParaProf.applicationManager.getApplications().iterator(); it.hasNext();) {
                    ParaProfApplication ppApp = (ParaProfApplication) it.next();
                    if (!ppApp.dBApplication()) {
                        ppApp.setDatabase(getDefaultDatabase());
                        for (Iterator it2 = ppApp.getExperimentList(); it2.hasNext();) {
                            ParaProfExperiment ppExp = (ParaProfExperiment) it2.next();
                            ppExp.setDatabase(getDefaultDatabase());
                            for (Iterator it3 = ppExp.getTrialList(); it3.hasNext();) {
                                ParaProfTrial ppTrial = (ParaProfTrial) it3.next();
                                ppTrial.getTrial().setDatabase(getDefaultDatabase());
                            }
                        }
                    }
                }
            }

            //   dbAPI = databaseAPI;
            return databaseAPI;
        } catch (Exception e) {
            //Try and determine what went wrong, and then popup the help window
            // giving the user some idea of what to do.
            ParaProf.getHelpWindow().setVisible(true);
            //Clear the window first.
            ParaProf.getHelpWindow().clearText();
            ParaProf.getHelpWindow().writeText("There was an error connecting to the database!");
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText(
                    "Please see the help items below to try and resolve this issue."
                            + " If none of those work, send an email to tau-bugs@cs.uoregon.edu"
                            + " including as complete a description of the problem as possible.");
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText("------------------");
            ParaProf.getHelpWindow().writeText("");

            ParaProf.getHelpWindow().writeText(
                    "1) JDBC driver issue:" + " The JDBC driver is required in your classpath. If you ran ParaProf using"
                            + " the shell script provided in tau (paraprof), then the default."
                            + " location used is $LOCATION_OF_TAU_ROOT/$ARCH/lib.");
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText(
                    " If you ran ParaProf manually, make sure that the location of"
                            + " the JDBC driver is in your classpath (you can set this in your."
                            + " environment, or as a commmand line option to java. As an example, PostgreSQL"
                            + " uses postgresql.jar as its JDBC driver name.");
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText(
                    "2) Network connection issue:"
                            + " Check your ability to connect to the database. You might be connecting to the"
                            + " incorrect port (PostgreSQL uses port 5432 by default). Also make sure that"
                            + " if there exists a firewall on you network (or local machine), it is not"
                            + " blocking you connection. Also check your database logs to ensure that you have"
                            + " permission to connect to the server.");
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText(
                    "3) Password issue:" + " Make sure that your password is set correctly. If it is not in the perfdmf"
                            + " configuration file, you can enter it manually by selecting"
                            + "  File -> Database Configuration in the ParaProfManagerWindow window.");
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText("------------------");
            ParaProf.getHelpWindow().writeText("");

            ParaProf.getHelpWindow().writeText("The full error is given below:\n");

            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            if (e instanceof TauRuntimeException) { // unwrap
                ParaProf.getHelpWindow().writeText(((TauRuntimeException) e).getMessage() + "\n\n");
                e = ((TauRuntimeException) e).getActualException();
            }
            e.printStackTrace(pw);
            pw.close();
            ParaProf.getHelpWindow().writeText(sw.toString());

            EventQueue.invokeLater(new Runnable() {
                public void run() {
                    ParaProf.getHelpWindow().getScrollPane().getVerticalScrollBar().setValue(0);
                }
            });

            //Collapse the dBApps node ... makes more sense to the user.
            //tree.collapsePath(new TreePath(dbApps));

            return null;
        }
    }

    //Respond correctly when this window is closed.
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
     * @return the loaded trials
     */
    public Vector getLoadedTrials() {
        return loadedTrials;
    }

}