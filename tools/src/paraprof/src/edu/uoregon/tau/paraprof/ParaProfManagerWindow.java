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
 * <P>CVS $Id: ParaProfManagerWindow.java,v 1.4 2005/01/06 22:49:44 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.4 $
 * @see		ParaProfManagerTableModel
 */



package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;

import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.database.*;
import java.sql.SQLException;

public class ParaProfManagerWindow extends JFrame implements ActionListener, TreeSelectionListener,
        TreeWillExpandListener {

    public ParaProfManagerWindow() {

        //Window Stuff.
        int windowWidth = 800;
        int windowHeight = 515;

        //Grab the screen size.
        Toolkit tk = Toolkit.getDefaultToolkit();
        Dimension screenDimension = tk.getScreenSize();
        int screenHeight = screenDimension.height;
        int screenWidth = screenDimension.width;

        //Find the center position with respect to this window.
        int xPosition = (screenWidth - windowWidth) / 2;
        int yPosition = (screenHeight - windowHeight) / 2;

        //Offset a little so that we do not interfere too much with the
        //main window which comes up in the centre of the screen.
        if (xPosition > 50)
            xPosition = xPosition - 50;
        if (yPosition > 50)
            yPosition = yPosition - 50;

        this.setLocation(xPosition, yPosition);
        setSize(new java.awt.Dimension(windowWidth, windowHeight));
        setTitle("ParaProf Manager");

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });
        //End - Window Stuff.

        setupMenus();

        //####################################
        //Create the tree.
        //####################################
        //Create the root node.
        DefaultMutableTreeNode root = new DefaultMutableTreeNode("Applications");
        standard = new DefaultMutableTreeNode("Standard Applications");
        runtime = new DefaultMutableTreeNode("Runtime Applications");
        dbApps = new DefaultMutableTreeNode("DB Applications");

        root.add(standard);
        root.add(runtime);
        root.add(dbApps);

        treeModel = new DefaultTreeModel(root);
        treeModel.setAsksAllowsChildren(true);
        tree = new JTree(treeModel);
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
                        if ((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0) {
                            if ((selectedNode == standard) || (selectedNode == dbApps)) {
                                clickedOnObject = selectedNode;
                                popup1.show(ParaProfManagerWindow.this, evt.getX(), evt.getY());
                            } else if (userObject instanceof ParaProfApplication) {
                                clickedOnObject = userObject;
                                popup2.show(ParaProfManagerWindow.this, evt.getX(), evt.getY());
                            } else if (userObject instanceof ParaProfExperiment) {
                                clickedOnObject = userObject;
                                popup3.show(ParaProfManagerWindow.this, evt.getX(), evt.getY());
                            } else if (userObject instanceof ParaProfTrial) {
                                clickedOnObject = userObject;
                                popup4.show(ParaProfManagerWindow.this, evt.getX(), evt.getY());
                            }
                        } else {
                            if (evt.getClickCount() == 2) {
                                if (userObject instanceof ParaProfMetric)
                                    metric((ParaProfMetric) userObject, true);
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

        //         (getContentPane()).add(jSplitOuterPane, "Center");

        this.getContentPane().add(jSplitInnerPane, "Center");

        //            this.getContentPane().removeAll();
        //
        //         
        //            jSplitInnerPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, treeScrollPane,
        //                    getPanelHelpMessage(0));
        //            jSplitInnerPane.setContinuousLayout(true);
        //            jSplitInnerPane.setDividerLocation(0.5);
        //            
        //            
        //            jSplitOuterPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, jSplitInnerPane, pPMLPanel);
        //            this.getContentPane().add(jSplitOuterPane, "Center");
        //
        //            jSplitOuterPane.setDividerLocation(0.75);

        //Show before setting dividers.
        //Components have to be realized on the screen before
        //the dividers can be set.

        jSplitInnerPane.setDividerLocation(0.5);

        this.show();

        jSplitInnerPane.setDividerLocation(0.5);

        //            jSplitOuterPane.setDividerLocation(1.0);
        //####################################
        //End - Set up the split pane, and add to content pane.
        //####################################

        ParaProf.incrementNumWindows();
    }

    void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();

        //File menu.
        JMenu fileMenu = new JMenu("File");

        JMenuItem menuItem = new JMenuItem("Open...");
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

        jMenuItem = new JMenuItem("Add Experiment");
        jMenuItem.addActionListener(this);
        popup2.add(jMenuItem);

        jMenuItem = new JMenuItem("Delete");
        jMenuItem.addActionListener(this);
        popup2.add(jMenuItem);

        jMenuItem = new JMenuItem("Add Trial");
        jMenuItem.addActionListener(this);
        popup2.add(jMenuItem);

        jMenuItem = new JMenuItem("Delete");
        jMenuItem.addActionListener(this);
        popup3.add(jMenuItem);

        jMenuItem = new JMenuItem("Add Trial");
        jMenuItem.addActionListener(this);
        popup3.add(jMenuItem);

        jMenuItem = new JMenuItem("Delete");
        jMenuItem.addActionListener(this);
        popup4.add(jMenuItem);

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
                        ParaProfExperiment experiment = addExperiment(false, application);
                        if (experiment != null)
                            (new LoadTrialWindow(this, application, experiment, false)).show();
                    }
                } else if (arg.equals("Close This Window")) {
                    closeThisWindow();
                } else if (arg.equals("Database Configuration")) {
                    (new DBConfiguration(this)).show();

                } else if (arg.equals("Show Derived Metric Panel")) {
                    if (showApplyOperationItem.isSelected()) {

                        this.getContentPane().removeAll();

                        //jSplitInnerPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, treeScrollPane,
                        //       getPanelHelpMessage(0));
                        //jSplitInnerPane.setContinuousLayout(true);

                        jSplitOuterPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, jSplitInnerPane, pPMLPanel);
                        this.getContentPane().add(jSplitOuterPane, "Center");

                        this.validate();
                        jSplitOuterPane.setDividerLocation(0.75);

                    } else {

                        double dividerLocation = jSplitInnerPane.getDividerLocation();
                        this.getContentPane().removeAll();

                        jSplitInnerPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, treeScrollPane,
                                getPanelHelpMessage(0));
                        jSplitInnerPane.setContinuousLayout(true);

                        this.getContentPane().add(jSplitInnerPane, "Center");

                        this.validate();
                        jSplitInnerPane.setDividerLocation(dividerLocation / this.getWidth());

                        //jSplitOuterPane.setDividerLocation(1.00);
                    }
                } else if (arg.equals("About ParaProf")) {
                    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
                } else if (arg.equals("Show Help Window")) {
                    ParaProf.helpWindow.show();
                    //Clear the window first.
                    ParaProf.helpWindow.clearText();
                    ParaProf.helpWindow.writeText("This is ParaProf's manager window!");
                    ParaProf.helpWindow.writeText("");
                    ParaProf.helpWindow.writeText("This window allows you to manage all of ParaProf's data sources,"
                            + " including loading data from local files, or from a database."
                            + " We also support the generation of derived metrics. Please see the"
                            + " items below for more help.");
                    ParaProf.helpWindow.writeText("");
                    ParaProf.helpWindow.writeText("------------------");
                    ParaProf.helpWindow.writeText("");

                    ParaProf.helpWindow.writeText("1) Navigation:"
                            + " The window is split into two halves, the left side gives a tree representation"
                            + " of all data. The right side gives information about items clicked on in the left"
                            + " half. You can also update information in the right half by double clicking in"
                            + " the fields, and entering new data.  This automatically updates the left half."
                            + " Right-clicking on the tree nodes in the left half displays popup menus which"
                            + " allow you to add/delete applications, experiments, or trials.");
                    ParaProf.helpWindow.writeText("");
                    ParaProf.helpWindow.writeText("2) DB Configuration:"
                            + " By default, ParaProf looks in the .ParaProf home directory in your home"
                            + " directory for the database configuration file.  If that file is found, then"
                            + " you are done, and can just expand the DB Applications node.  If there was a"
                            + " problem finding the file, you can enter the location of the file by selecting"
                            + " File -> Database Configuration.  You can also override the configuration file"
                            + " password in the same manner.");
                    ParaProf.helpWindow.writeText("");
                    ParaProf.helpWindow.writeText("3) Deriving new metrics:"
                            + " By selecting Options -> Show Derived Metric Panel, you will display the apply"
                            + " operations window.  Clicking on the metrics of a trial will update the"
                            + " arguments to the selected operation.  Currently, you can only derive metrics"
                            + " from metric in the same trial (thus for example creating floating point"
                            + " operations per second by taking PAPI_FP_INS and dividing it by GET_TIME_OF_DAY)."
                            + " The 2nd argument is a user editable textbox and can be filled in with scalar "
                            + " values using the keyword 'val' (e.g. \"val 1.5\".");
                    ParaProf.helpWindow.writeText("");
                    ParaProf.helpWindow.writeText("------------------");
                    ParaProf.helpWindow.writeText("");
                } else if (arg.equals("Delete")) {
                    if (clickedOnObject instanceof ParaProfApplication) {
                        ParaProfApplication application = (ParaProfApplication) clickedOnObject;
                        if (application.dBApplication()) {

                            DatabaseAPI databaseAPI = this.getDBSession();
                            if (databaseAPI != null) {
                                databaseAPI.deleteApplication(application.getID());
                                databaseAPI.terminate();
                                //Remove any loaded trials associated with this application.
                                for (Enumeration e = loadedTrials.elements(); e.hasMoreElements();) {
                                    ParaProfTrial loadedTrial = (ParaProfTrial) e.nextElement();
                                    if (loadedTrial.getApplicationID() == application.getID())
                                        loadedTrials.remove(loadedTrial);
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

                            DatabaseAPI databaseAPI = this.getDBSession();
                            if (databaseAPI != null) {
                                databaseAPI.deleteExperiment(experiment.getID());
                                databaseAPI.terminate();
                                //Remove any loaded trials associated with this application.
                                for (Enumeration e = loadedTrials.elements(); e.hasMoreElements();) {
                                    ParaProfTrial loadedTrial = (ParaProfTrial) e.nextElement();
                                    if (loadedTrial.getApplicationID() == experiment.getApplicationID()
                                            && loadedTrial.getExperimentID() == experiment.getID())
                                        loadedTrials.remove(loadedTrial);
                                }
                                treeModel.removeNodeFromParent(experiment.getDMTN());
                            }

                        } else {
                            experiment.getApplication().removeExperiment(experiment);
                            treeModel.removeNodeFromParent(experiment.getDMTN());
                        }

                    } else if (clickedOnObject instanceof ParaProfTrial) {
                        ParaProfTrial trial = (ParaProfTrial) clickedOnObject;
                        if (trial.dBTrial()) {

                            DatabaseAPI databaseAPI = this.getDBSession();
                            if (databaseAPI != null) {
                                databaseAPI.deleteTrial(trial.getID());
                                databaseAPI.terminate();
                                //Remove any loaded trials associated with this application.
                                for (Enumeration e = loadedTrials.elements(); e.hasMoreElements();) {
                                    ParaProfTrial loadedTrial = (ParaProfTrial) e.nextElement();
                                    if (loadedTrial.getApplicationID() == trial.getApplicationID()
                                            && loadedTrial.getExperimentID() == trial.getID()
                                            && loadedTrial.getID() == trial.getID())
                                        loadedTrials.remove(loadedTrial);
                                }
                                treeModel.removeNodeFromParent(trial.getDMTN());
                            }
                        } else {
                            trial.getExperiment().removeTrial(trial);
                            treeModel.removeNodeFromParent(trial.getDMTN());
                        }
                    }
                } else if (arg.equals("Add Application")) {
                    if (clickedOnObject == standard) {
                        ParaProfApplication application = addApplication(false, standard);
                        this.expandApplicationType(0, application.getID(), application);
                    } else if (clickedOnObject == dbApps) {
                        ParaProfApplication application = addApplication(true, dbApps);
                        this.expandApplicationType(2, application.getID(), application);
                    }
                } else if (arg.equals("Add Experiment")) {
                    if (clickedOnObject == standard) {
                        ParaProfApplication application = addApplication(false, standard);
                        ParaProfExperiment experiment = addExperiment(false, application);
                        if (application != null || experiment != null)
                            this.expandApplication(0, application.getID(), experiment.getID(), application,
                                    experiment);
                    } else if (clickedOnObject == dbApps) {
                        ParaProfApplication application = addApplication(true, dbApps);
                        ParaProfExperiment experiment = addExperiment(true, application);
                        if (application != null || experiment != null)
                            this.expandApplication(2, application.getID(), experiment.getID(), application,
                                    experiment);
                    }
                    if (clickedOnObject instanceof ParaProfApplication) {
                        ParaProfApplication application = (ParaProfApplication) clickedOnObject;
                        if (application.dBApplication()) {
                            ParaProfExperiment experiment = addExperiment(true, application);
                            if (experiment != null)
                                this.expandApplication(2, application.getID(), experiment.getID(), null,
                                        experiment);
                        } else {
                            ParaProfExperiment experiment = addExperiment(false, application);
                            if (experiment != null)
                                this.expandApplication(0, application.getID(), experiment.getID(), null,
                                        experiment);
                        }
                    }
                } else if (arg.equals("Add Trial")) {
                    if (clickedOnObject == standard) {
                        ParaProfApplication application = addApplication(false, standard);
                        if (application != null) {
                            ParaProfExperiment experiment = addExperiment(false, application);
                            if (experiment != null)
                                (new LoadTrialWindow(this, application, experiment, false)).show();
                        }
                    } else if (clickedOnObject == dbApps) {
                        ParaProfApplication application = addApplication(true, dbApps);
                        if (application != null) {
                            ParaProfExperiment experiment = addExperiment(true, application);
                            if (experiment != null)
                                (new LoadTrialWindow(this, application, experiment, false)).show();
                        }
                    } else if (clickedOnObject instanceof ParaProfApplication) {
                        ParaProfApplication application = (ParaProfApplication) clickedOnObject;
                        if (application.dBApplication()) {
                            ParaProfExperiment experiment = addExperiment(true, application);
                            if (experiment != null)
                                (new LoadTrialWindow(this, null, experiment, false)).show();
                        } else {
                            ParaProfExperiment experiment = addExperiment(false, application);
                            if (experiment != null)
                                (new LoadTrialWindow(this, null, experiment, false)).show();
                        }
                    } else if (clickedOnObject instanceof ParaProfExperiment) {
                        ParaProfExperiment experiment = (ParaProfExperiment) clickedOnObject;
                        if (experiment.dBExperiment()) {
                            (new LoadTrialWindow(this, null, experiment, true)).show();
                        } else
                            (new LoadTrialWindow(this, null, experiment, false)).show();
                    }
                } else if (arg.equals("Upload Trial to DB")) {
                    if (clickedOnObject instanceof ParaProfTrial) {
                        ParaProfTrial clickedOnTrial = (ParaProfTrial) clickedOnObject;
                        int[] array = this.getSelectedDBExperiment();
                        if (array != null) {
                            DatabaseAPI databaseAPI = this.getDBSession();
                            if (databaseAPI != null) {
                                Trial trial = new Trial(0);
                                trial.setDataSource(clickedOnTrial.getDataSource());
                                trial.setName(clickedOnTrial.getName());
                                trial.setExperimentID(array[1]);
                                int[] maxNCT = clickedOnTrial.getMaxNCTNumbers();
                                // 				trial.setNodeCount(maxNCT[0]+1);
                                // 				trial.setNumContextsPerNode(maxNCT[1]+1);
                                // 				trial.setNumThreadsPerContext(maxNCT[2]+1);
                                databaseAPI.saveParaProfTrial(trial, -1);
                                databaseAPI.terminate();
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void valueChanged(TreeSelectionEvent event) {
        try {
            TreePath path = tree.getSelectionPath();
            if (path == null)
                return;
            DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
            DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
            Object userObject = selectedNode.getUserObject();

            if (selectedNode.isRoot()) {
                jSplitInnerPane.setRightComponent(getPanelHelpMessage(0));
                jSplitInnerPane.setDividerLocation(0.5);
            } else if ((parentNode.isRoot())) {
                if (userObject.toString().equals("Standard Applications")) {
                    jSplitInnerPane.setRightComponent(getPanelHelpMessage(1));
                    jSplitInnerPane.setDividerLocation(0.5);
                } else if (userObject.toString().equals("Runtime Applications")) {
                    jSplitInnerPane.setRightComponent(getPanelHelpMessage(2));
                    jSplitInnerPane.setDividerLocation(0.5);
                } else if (userObject.toString().equals("DB Applications")) {
                    jSplitInnerPane.setRightComponent(getPanelHelpMessage(3));
                    jSplitInnerPane.setDividerLocation(0.5);
                }
            } else if (userObject instanceof ParaProfApplication) {
                jSplitInnerPane.setRightComponent(getTable(userObject));
                jSplitInnerPane.setDividerLocation(0.5);
            } else if (userObject instanceof ParaProfExperiment) {
                jSplitInnerPane.setRightComponent(getTable(userObject));
                jSplitInnerPane.setDividerLocation(0.5);
            } else if (userObject instanceof ParaProfTrial) {
                jSplitInnerPane.setRightComponent(getTable(userObject));
                jSplitInnerPane.setDividerLocation(0.5);
            } else if (userObject instanceof ParaProfMetric)
                this.metric((ParaProfMetric) userObject, false);
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void treeWillCollapse(TreeExpansionEvent event) {
        try {
            TreePath path = event.getPath();
            if (path == null)
                return;
            if (UtilFncs.debug)
                System.out.println("In treeWillCollapse - path:" + path.toString());
            DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
            DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
            Object userObject = selectedNode.getUserObject();

            if (selectedNode.isRoot()) {
                int childCount = standard.getChildCount();
                for (int i = 0; i < childCount; i++)
                    this.clearDefaultMutableTreeNodes((DefaultMutableTreeNode) standard.getChildAt(i));
                childCount = runtime.getChildCount();
                for (int i = 0; i < childCount; i++)
                    this.clearDefaultMutableTreeNodes((DefaultMutableTreeNode) runtime.getChildAt(i));
                childCount = dbApps.getChildCount();
                for (int i = 0; i < childCount; i++)
                    this.clearDefaultMutableTreeNodes((DefaultMutableTreeNode) dbApps.getChildAt(i));
            } else if (parentNode.isRoot()) {
                if (userObject.toString().equals("Standard Applications")) {
                    int childCount = standard.getChildCount();
                    for (int i = 0; i < childCount; i++)
                        this.clearDefaultMutableTreeNodes((DefaultMutableTreeNode) standard.getChildAt(i));
                } else if (userObject.toString().equals("Runtime Applications")) {
                    int childCount = runtime.getChildCount();
                    for (int i = 0; i < childCount; i++)
                        this.clearDefaultMutableTreeNodes((DefaultMutableTreeNode) runtime.getChildAt(i));
                } else if (userObject.toString().equals("DB Applications")) {
                    int childCount = dbApps.getChildCount();
                    for (int i = 0; i < childCount; i++)
                        this.clearDefaultMutableTreeNodes((DefaultMutableTreeNode) dbApps.getChildAt(i));
                }
            } else if (userObject instanceof ParaProfTreeNodeUserObject) {
                this.clearDefaultMutableTreeNodes(selectedNode);
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void treeWillExpand(TreeExpansionEvent event) {
        try {
            TreePath path = event.getPath();
            if (path == null)
                return;
            if (UtilFncs.debug)
                System.out.println("In treeWillExpand - path:" + path.toString());
            DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
            DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
            Object userObject = selectedNode.getUserObject();

            if (selectedNode.isRoot()) {
                //Do not need to do anything here.
                return;
            } else if ((parentNode.isRoot())) {
                if (userObject.toString().equals("Standard Applications")) {
                    jSplitInnerPane.setRightComponent(getPanelHelpMessage(1));
                    jSplitInnerPane.setDividerLocation(0.5);
                    //Refresh the application list.
                    for (int i = standard.getChildCount(); i > 0; i--) {
                        treeModel.removeNodeFromParent(((DefaultMutableTreeNode) standard.getChildAt(i - 1)));
                    }
                    ListIterator l = ParaProf.applicationManager.getApplicationList();
                    while (l.hasNext()) {
                        ParaProfApplication application = (ParaProfApplication) l.next();
                        DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
                        application.setDMTN(applicationNode);
                        treeModel.insertNodeInto(applicationNode, standard, standard.getChildCount());
                    }
                    return;
                } else if (userObject.toString().equals("Runtime Applications")) {
                    jSplitInnerPane.setRightComponent(getPanelHelpMessage(2));
                    jSplitInnerPane.setDividerLocation(0.5);
                } else if (userObject.toString().equals("DB Applications")) {
                        jSplitInnerPane.setRightComponent(getPanelHelpMessage(3));
                        jSplitInnerPane.setDividerLocation(0.5);
                        for (int i = dbApps.getChildCount(); i > 0; i--) {
                            treeModel.removeNodeFromParent(((DefaultMutableTreeNode) dbApps.getChildAt(i - 1)));
                        }

                        DatabaseAPI databaseAPI = this.getDBSession();
                        if (databaseAPI != null) {

                            ListIterator l = databaseAPI.getApplicationList();
                            while (l.hasNext()) {
                                ParaProfApplication application = new ParaProfApplication(
                                        (Application) l.next());
                                application.setDBApplication(true);
                                DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
                                application.setDMTN(applicationNode);
                                treeModel.insertNodeInto(applicationNode, dbApps, dbApps.getChildCount());
                            }
                            databaseAPI.terminate();
                        }

                        return;
                }
            } else if (userObject instanceof ParaProfApplication) {
                    ParaProfApplication application = (ParaProfApplication) userObject;
                    if (application.dBApplication()) {
                        //Refresh the experiments list.
                        //System.out.println("Loading experiment list ...");
                        for (int i = selectedNode.getChildCount(); i > 0; i--) {
                            treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i - 1)));
                        }

                        DatabaseAPI databaseAPI = this.getDBSession();
                        if (databaseAPI != null) {
                            databaseAPI.setApplication(application.getID());
                            ListIterator l = databaseAPI.getExperimentList();
                            while (l.hasNext()) {
                                ParaProfExperiment experiment = new ParaProfExperiment((Experiment) l.next());
                                experiment.setDBExperiment(true);
                                experiment.setApplication(application);
                                DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(experiment);
                                experiment.setDMTN(experimentNode);
                                treeModel.insertNodeInto(experimentNode, selectedNode,
                                        selectedNode.getChildCount());
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
                    jSplitInnerPane.setRightComponent(getTable(userObject));
                    jSplitInnerPane.setDividerLocation(0.5);
            } else if (userObject instanceof ParaProfExperiment) {
                    ParaProfExperiment experiment = (ParaProfExperiment) userObject;
                    if (experiment.dBExperiment()) {
                        // refresh the trials list
                        for (int i = selectedNode.getChildCount(); i > 0; i--) {
                            treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i - 1)));
                        }

                        DatabaseAPI databaseAPI = this.getDBSession();
                        if (databaseAPI != null) {
                            databaseAPI.setExperiment(experiment.getID());
                            ListIterator l = databaseAPI.getTrialList();
                            while (l.hasNext()) {
                                ParaProfTrial trial = new ParaProfTrial((Trial) l.next(), 4);
                                trial.setDBTrial(true);
                                trial.setExperiment(experiment);
                                DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(trial);
                                trial.setDMTN(trialNode);
                                treeModel.insertNodeInto(trialNode, selectedNode, selectedNode.getChildCount());
                            }
                            databaseAPI.terminate();
                        }
                    } else {
                        for (int i = selectedNode.getChildCount(); i > 0; i--) {
                            treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i - 1)));
                        }
                        ListIterator l = experiment.getTrialList();
                        while (l.hasNext()) {
                            ParaProfTrial trial = (ParaProfTrial) l.next();
                            DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(trial);
                            trial.setDMTN(trialNode);
                            treeModel.insertNodeInto(trialNode, selectedNode, selectedNode.getChildCount());
                            trial.setTreePath(new TreePath(trialNode.getPath()));
                        }
                    }
                    jSplitInnerPane.setRightComponent(getTable(userObject));
                    jSplitInnerPane.setDividerLocation(0.5);
            } else if (userObject instanceof ParaProfTrial) {
                ParaProfTrial ppTrial = (ParaProfTrial) userObject;
                if (ppTrial.dBTrial()) {
                    //Test to see if trial has been loaded already.
                    boolean loadedExists = false;
                    for (Enumeration e = loadedTrials.elements(); e.hasMoreElements();) {
                        ParaProfTrial loadedTrial = (ParaProfTrial) e.nextElement();
                        if ((ppTrial.getID() == loadedTrial.getID())
                                && (ppTrial.getExperimentID() == loadedTrial.getExperimentID())
                                && (ppTrial.getApplicationID() == loadedTrial.getApplicationID())) {
                            selectedNode.setUserObject(loadedTrial);
                            ppTrial = loadedTrial;
                            loadedExists = true;
                        }
                    }

                    if (!loadedExists) {
                        //Need to load the trial in from the db.
                        //System.out.println("Loading trial ...");
                        ppTrial.setLoading(true);

                        DatabaseAPI databaseAPI = this.getDBSession();
                        if (databaseAPI != null) {
                            databaseAPI.setApplication(ppTrial.getApplicationID());
                            databaseAPI.setExperiment(ppTrial.getExperimentID());
                            databaseAPI.setTrial(ppTrial.getID());

                            DBDataSource dbDataSource = new DBDataSource(databaseAPI);
                            //dBSource.setDebug(UtilFncs.debug);

                            //                        int numNodes=0;
                            //                        int numContextsPerNode=0;
                            //                        int numThreadsPerContext=0;
                            //                        if (ppTrial.getField("node_count") != null) {
                            //                            numNodes = Integer.parseInt(ppTrial.getField("node_count"));
                            //                        }
                            //                        if (ppTrial.getField("contexts_per_node") != null) {
                            //                            numContextsPerNode = Integer.parseInt(ppTrial.getField("contexts_per_node"));
                            //                        }
                            //                        if (ppTrial.getField("threads_per_context") != null) {
                            //                            numThreadsPerContext = Integer.parseInt(ppTrial.getField("threads_per_context"));
                            //                        }
                            //
                            //                        DatabaseAPI.setTotalItems(numNodes*numContextsPerNode*numThreadsPerContext);
                            //                        
                            //                        
                            //                        LoadTrialProgressWindow lpw = new LoadTrialProgressWindow(this, dbDataSource, ppTrial);
                            //                        lpw.show();

                            DataSourceThreadControl dataSourceThreadControl = new DataSourceThreadControl();
                            dataSourceThreadControl.addObserver(ppTrial);
                            dataSourceThreadControl.initialize(dbDataSource);

                            //Add to the list of loaded trials.
                            loadedTrials.add(ppTrial);

                        }
                    }
                }

                //At this point, in both the db and non-db cases, the trial
                //is either loading or not. Check this before displaying.
                if (!ppTrial.loading()) {
                    //System.out.println("Loading metric list ...");
                    for (int i = selectedNode.getChildCount(); i > 0; i--) {
                        treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i - 1)));
                    }
                    ListIterator l = ppTrial.getMetricList();
                    while (l.hasNext()) {
                        ParaProfMetric metric = (ParaProfMetric) l.next();
                        DefaultMutableTreeNode metricNode = new DefaultMutableTreeNode(metric, false);

                        metric.setDMTN(metricNode);
                        treeModel.insertNodeInto(metricNode, selectedNode, selectedNode.getChildCount());
                    }
                    //System.out.println("Done loading metric list.");

                    jSplitInnerPane.setRightComponent(getTable(userObject));
                    jSplitInnerPane.setDividerLocation(0.5);
                } else {
                    //                jSplitInnerPane.setRightComponent(new JScrollPane(
                    //                      this.getLoadingTrialPanel(userObject)));
                    //            jSplitInnerPane.setDividerLocation(0.5);
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    private void metric(ParaProfMetric metric, boolean show) {
        jSplitInnerPane.setRightComponent(getTable(metric));
        jSplitInnerPane.setDividerLocation(0.5);
        this.operand2 = this.operand1;
        pPMLPanel.setArg2Field(pPMLPanel.getArg1Field());
        operand1 = metric;
        pPMLPanel.setArg1Field(metric.getApplicationID() + ":" + metric.getExperimentID() + ":"
                + metric.getTrialID() + ":" + metric.getID() + " - " + metric.getName());
        if (show)
            this.showMetric(metric);
    }

    public ParaProfMetric getOperand1() {
        return operand1;
    }

    public ParaProfMetric getOperand2() {
        return operand2;
    }

    public void uploadMetric(ParaProfMetric metric) {
        if (metric != null) {
            DatabaseAPI databaseAPI = this.getDBSession();
            if (databaseAPI != null) {
                try {
                    databaseAPI.saveParaProfTrial(metric.getTrial(), metric.getID());
                } catch (DatabaseException e) {
                    ParaProfUtils.handleException (e);
                }
                databaseAPI.terminate();
            }
        }
    }

    public void clearDefaultMutableTreeNodes(DefaultMutableTreeNode defaultMutableTreeNode) {
        //Do not want to clear this node's user object's DefaultMutableTreeNode
        // so start the recursion on its children.
        int count = defaultMutableTreeNode.getChildCount();
        for (int i = 0; i < count; i++)
            clearDefaultMutableTreeNodesHelper((DefaultMutableTreeNode) defaultMutableTreeNode.getChildAt(i));
    }

    public void clearDefaultMutableTreeNodesHelper(DefaultMutableTreeNode defaultMutableTreeNode) {
        int count = defaultMutableTreeNode.getChildCount();
        for (int i = 0; i < count; i++)
            this.clearDefaultMutableTreeNodesHelper((DefaultMutableTreeNode) defaultMutableTreeNode.getChildAt(i));
        ((ParaProfTreeNodeUserObject) defaultMutableTreeNode.getUserObject()).clearDefaultMutableTreeNodes();
    }

    public int[] getSelectedDBExperiment() {
        if (ParaProf.savedPreferences.getDatabaseConfigurationFile() == null
                || ParaProf.savedPreferences.getDatabasePassword() == null) {
            // Check to see if the user has set configuration information.
            JOptionPane.showMessageDialog(this,
                    "Please set the database configuration information (file menu).",
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

    //    public int[] getSelectedDBTrial() {
    //        if (ParaProf.savedPreferences.getDatabaseConfigurationFile() == null
    //                || ParaProf.savedPreferences.getDatabasePassword() == null) {
    //            // Check to see if the user has set configuration information.
    //            JOptionPane.showMessageDialog(this,
    //                    "Please set the database configuration information (file menu).",
    //                    "DB Configuration Error!", JOptionPane.ERROR_MESSAGE);
    //            return null;
    //        }
    //
    //        TreePath path = tree.getSelectionPath();
    //        boolean error = false;
    //        if (path == null)
    //            error = true;
    //        else {
    //            DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
    //            Object userObject = selectedNode.getUserObject();
    //            if (userObject instanceof ParaProfTrial) {
    //                ParaProfTrial trial = (ParaProfTrial) userObject;
    //                if (trial.dBTrial()) {
    //                    int[] array = new int[3];
    //                    array[0] = trial.getApplicationID();
    //                    array[1] = trial.getExperimentID();
    //                    array[2] = trial.getID();
    //                    return array;
    //                } else
    //                    error = true;
    //            } else
    //                error = true;
    //        }
    //        if (error)
    //            JOptionPane.showMessageDialog(this, "Please select an db trial first!", "DB Upload Error!",
    //                    JOptionPane.ERROR_MESSAGE);
    //        return null;
    //    }

    private Component getPanelHelpMessage(int type) {
        JTextArea jTextArea = new JTextArea();
        jTextArea.setLineWrap(true);
        jTextArea.setWrapStyleWord(true);
        ///Set the text.
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
                    + "under the default application automatically. Please see the ParaProf documentation for mre details.\n");
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
            jTextArea.append("DB Apps - Another new feature allowing ParaProf to load data from a database.  Again, please see"
                    + " the ParaProf documentation if the options are not clear.\n");
            break;
        default:
            break;
        }
        return (new JScrollPane(jTextArea));
    }

    private Component getTable(Object obj) {
        return (new JScrollPane(new JTable(new ParaProfManagerTableModel(this, obj, treeModel))));
    }

   

    public ParaProfApplication addApplication(boolean dBApplication, DefaultMutableTreeNode treeNode)
            throws SQLException {
        ParaProfApplication application = null;
        if (dBApplication) {

            DatabaseAPI databaseAPI = this.getDBSession();
            if (databaseAPI != null) {
                application = new ParaProfApplication(databaseAPI.db());
                application.setDBApplication(true);
                application.setName("New Application");
                application.setID(databaseAPI.saveApplication(application));
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
            DatabaseAPI databaseAPI = this.getDBSession();
            if (databaseAPI != null) {
                experiment = new ParaProfExperiment(databaseAPI.db());
                experiment.setDBExperiment(true);
                experiment.setApplicationID(application.getID());
                experiment.setName("New Experiment");
                experiment.setID(databaseAPI.saveExperiment(experiment));
                databaseAPI.terminate();

            }
        } else {
            experiment = application.addExperiment();
            experiment.setName("New Experiment");
        }
        return experiment;
    }

    public void addTrial(ParaProfApplication application, ParaProfExperiment experiment, File files[],
            int fileType, boolean fixGprofNames) {
        ParaProfTrial ppTrial = null;
        DataSource dataSource = null;

        try {
            dataSource = UtilFncs.initializeDataSource(files, fileType, fixGprofNames);
        } catch (DataSourceException e) {

            if (files == null || files.length != 0) // We don't output an error message if paraprof was just invoked with no parameters.
                new ParaProfErrorDialog(e);

            return;
        }

        ppTrial = new ParaProfTrial(fileType);

        ppTrial.setExperiment(experiment);
        ppTrial.setApplicationID(experiment.getApplicationID());
        ppTrial.setExperimentID(experiment.getID());
        if (files.length != 0) {
            ppTrial.setPaths(files[0].getPath());
        } else {
            ppTrial.setPaths(System.getProperty("user.dir"));
        }
        ppTrial.setName(ppTrial.getPathReverse());
        ppTrial.setLoading(true);
        if (experiment.dBExperiment()) {
            ppTrial.setUpload(true); //This trial is not set to a db
            // trial until after it has
            // finished loading.
        } else {
            experiment.addTrial(ppTrial);
        }

        LoadTrialProgressWindow lpw = new LoadTrialProgressWindow(this, dataSource, ppTrial);
        lpw.show();

        //                    dataSource.setDebug(UtilFncs.debug);
        //            DataSourceThreadControl dataSourceThreadControl = new DataSourceThreadControl();
        //            dataSourceThreadControl.setDebug(UtilFncs.debug);
        //            dataSourceThreadControl.addObserver(trial);
        //            dataSourceThreadControl.initialize(dataSource, true);

        if (experiment.dBExperiment()) //Check need to occur on the
            // experiment as trial not
            // yet a recognized db trial.
            this.expandTrial(2, ppTrial.getApplicationID(), ppTrial.getExperimentID(), ppTrial.getID(),
                    application, experiment, ppTrial);
        else
            this.expandTrial(0, ppTrial.getApplicationID(), ppTrial.getExperimentID(), ppTrial.getID(),
                    application, experiment, ppTrial);
    }

    private void showMetric(ParaProfMetric metric) {
        try {
            ParaProfTrial ppTrial = metric.getTrial();
            if (ppTrial.getSelectedMetricID() != metric.getID()) {
                ppTrial.setSelectedMetricID(metric.getID());
                ppTrial.getSystemEvents().updateRegisteredObjects("dataEvent");
            }
            ppTrial.showMainWindow();
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void populateTrialMetrics(ParaProfTrial trial) {
        try {
            if (trial.upload()) {
                //Add to the list of loaded trials.
                loadedTrials.add(trial);
                trial.setUpload(false);
            }

            if (trial.dBTrial()) {
                this.expandTrial(2, trial.getApplicationID(), trial.getExperimentID(), trial.getID(), null,
                        null, trial);
            } else {
                this.expandTrial(0, trial.getApplicationID(), trial.getExperimentID(), trial.getID(), null,
                        null, trial);
            }

        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public DefaultMutableTreeNode expandApplicationType(int type, int applicationID,
            ParaProfApplication application) {
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
        case 1:
            break;
        case 2:
            //Test to see if dbApps is expanded, if not, expand it.
            if (!(tree.isExpanded(new TreePath(dbApps.getPath()))))
                tree.expandPath(new TreePath(dbApps.getPath()));

            //Try and find the required application node.
            for (int i = dbApps.getChildCount(); i > 0; i--) {
                DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) dbApps.getChildAt(i - 1);
                if (applicationID == ((ParaProfApplication) defaultMutableTreeNode.getUserObject()).getID())
                    return defaultMutableTreeNode;
            }
            //Required application node was not found, try adding it.
            if (application != null) {
                DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
                application.setDMTN(applicationNode);
                treeModel.insertNodeInto(applicationNode, dbApps, dbApps.getChildCount());
                return applicationNode;
            }
            return null;
        default:
            break;
        }
        return null;
    }

    //Expands the given application
    public DefaultMutableTreeNode expandApplication(int type, int applicationID, int experimentID,
            ParaProfApplication application, ParaProfExperiment experiment) {
        DefaultMutableTreeNode applicationNode = this.expandApplicationType(type, applicationID, application);
        if (applicationNode != null) {
            //Expand the application.
            tree.expandPath(new TreePath(applicationNode.getPath()));

            //Try and find the required experiment node.
            tree.expandPath(new TreePath(standard.getPath()));
            for (int i = applicationNode.getChildCount(); i > 0; i--) {
                DefaultMutableTreeNode defaultMutableTreeNode = (DefaultMutableTreeNode) applicationNode.getChildAt(i - 1);
                if (experimentID == ((ParaProfExperiment) defaultMutableTreeNode.getUserObject()).getID())
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

    public DefaultMutableTreeNode expandExperiment(int type, int applicationID, int experimentID, int trialID,
            ParaProfApplication application, ParaProfExperiment experiment, ParaProfTrial trial) {
        DefaultMutableTreeNode experimentNode = this.expandApplication(type, applicationID, experimentID,
                application, experiment);
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
            if (trial != null) {
                DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(trial);
                trial.setDMTN(trialNode);
                treeModel.insertNodeInto(trialNode, experimentNode, experimentNode.getChildCount());
                return trialNode;
            }
            return null;
        }
        return null;
    }

    public void expandTrial(int type, int applicationID, int experimentID, int trialID,
            ParaProfApplication application, ParaProfExperiment experiment, ParaProfTrial trial) {
        DefaultMutableTreeNode trialNode = this.expandExperiment(type, applicationID, experimentID, trialID,
                application, experiment, trial);
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

    public DatabaseAPI getDBSession() {

        try {
            //Check to see if the user has set configuration information.
            if (ParaProf.savedPreferences.getDatabaseConfigurationFile() == null) {
                JOptionPane.showMessageDialog(this,
                        "Please set the database configuration information (file menu).",
                        "DB Configuration Error!", JOptionPane.ERROR_MESSAGE);
                return null;
            } else {//Test to see if configurataion file exists.
                File file = new File(ParaProf.savedPreferences.getDatabaseConfigurationFile());
                if (!file.exists()) {
                    JOptionPane.showMessageDialog(this, "Specified configuration file does not exist.",
                            "DB Configuration Error!", JOptionPane.ERROR_MESSAGE);
                    return null;
                }
            }
            //Basic checks done, try to access the db.
            DatabaseAPI databaseAPI = new DatabaseAPI();
            if (ParaProf.savedPreferences.getDatabasePassword() == null)
                databaseAPI.initialize(ParaProf.savedPreferences.getDatabaseConfigurationFile(), false);
            else
                databaseAPI.initialize(ParaProf.savedPreferences.getDatabaseConfigurationFile(),
                        ParaProf.savedPreferences.getDatabasePassword());
            return databaseAPI;
        } catch (Exception e) {
            //Try and determine what went wrong, and then popup the help window
            // giving the user some idea
            //what to do.
            ParaProf.helpWindow.show();
            //Clear the window first.
            ParaProf.helpWindow.clearText();
            ParaProf.helpWindow.writeText("There was an error connecting to the database!");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("Please see the help items below to try and resolve this issue."
                    + " If none of those work, send an email to tau-bugs@cs.uoregon.edu"
                    + " including as complete a description of the problem as possible.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("------------------");
            ParaProf.helpWindow.writeText("");

            ParaProf.helpWindow.writeText("1) JDBC driver issue:"
                    + " The JDBC driver is required in your classpath. If you ran ParaProf using"
                    + " the shell script provided in tau (paraprof), then the default."
                    + " location used is $LOCATION_OF_TAU_ROOT/$ARCH/lib.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText(" If you ran ParaProf manually, make sure that the location of"
                    + " the JDBC driver is in your classpath (you can set this in your."
                    + " environment, or as a commmand line option to java. As an example, PostgreSQL"
                    + " uses postgresql.jar as its JDBC driver name.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("2) Network connection issue:"
                    + " Check your ability to connect to the database. You might be connecting to the"
                    + " incorrect port (PostgreSQL uses port 5432 by default). Also make sure that"
                    + " if there exists a firewall on you network (or local machine), it is not"
                    + " blocking you connection. Also check your database logs to ensure that you have"
                    + " permission to connect to the server.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("3) Password issue:"
                    + " Make sure that your password is set correctly. If it is not in the perfdmf"
                    + " configuration file, you can enter it manually by selecting"
                    + "  File -> Database Configuration in the ParaProfManagerWindow window.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("------------------");
            ParaProf.helpWindow.writeText("");

            ParaProf.helpWindow.writeText("The full error is given below:\n");

            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.close();
            ParaProf.helpWindow.writeText(sw.toString());
            
            //Collapse the dBApps node ... makes more sense to the user.
            tree.collapsePath(new TreePath(dbApps));

            return null;
        }
    }


    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    void closeThisWindow() {
        try {
            setVisible(false);
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

    //Instance Data.
    private JTree tree = null;
    private DefaultTreeModel treeModel = null;
    private DefaultMutableTreeNode standard = null;
    private DefaultMutableTreeNode runtime = null;
    private DefaultMutableTreeNode dbApps = null;
    private JSplitPane jSplitInnerPane = null;
    private JSplitPane jSplitOuterPane = null;

    //A reference to the default trial node.
    private DefaultMutableTreeNode defaultParaProfTrialNode = null;

    private JCheckBoxMenuItem showApplyOperationItem = null;
    private DerivedMetricPanel pPMLPanel = new DerivedMetricPanel(this);

    JScrollPane treeScrollPane;

    private Vector loadedTrials = new Vector();

    //Popup menu stuff.
    private JPopupMenu popup1 = new JPopupMenu();
    private JPopupMenu popup2 = new JPopupMenu();
    private JPopupMenu popup3 = new JPopupMenu();
    private JPopupMenu popup4 = new JPopupMenu();
    private JMenuItem uploadMetricItem = null;

    private Object clickedOnObject = null;
    private ParaProfMetric operand1 = null;
    private ParaProfMetric operand2 = null;
}