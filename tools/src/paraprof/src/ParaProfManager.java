/* 
   ParaProfManager.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.text.*;
import javax.swing.border.*;
import javax.swing.event.*;
import javax.swing.tree.*;
import javax.swing.table.*;
import java.sql.*;

public class ParaProfManager extends JFrame implements ActionListener 
{
    public ParaProfManager()
    {
    
	try{
	    //Some window stuff.
	    setLocation(new java.awt.Point(0, 0));
	    setSize(new java.awt.Dimension(800, 600));
	    setTitle("ParaProf Manager");
      
	    //Add some window listener code
	    addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(java.awt.event.WindowEvent evt) {
			thisWindowClosing(evt);
		    }
		});
      
      
	    //******************************
	    //Code to generate the menus.
	    //******************************
	    JMenuBar mainMenu = new JMenuBar();
      
	    //******************************
	    //File menu.
	    //******************************
	    JMenu fileMenu = new JMenu("File");
      
	    //Add a menu item.
	    JMenuItem closeItem = new JMenuItem("ParaProf Manager");
	    closeItem.addActionListener(this);
	    fileMenu.add(closeItem);
      
      
	    //Add a menu item.
	    JMenuItem exitItem = new JMenuItem("Exit ParaProf!");
	    exitItem.addActionListener(this);
	    fileMenu.add(exitItem);
	    //******************************
	    //End - File menu.
	    //******************************
      
	    //******************************
	    //Help menu.
	    //******************************
	    JMenu helpMenu = new JMenu("Help");
      
	    //Add a menu item.
	    JMenuItem aboutItem = new JMenuItem("About Racy");
	    aboutItem.addActionListener(this);
	    helpMenu.add(aboutItem);
      
	    //Add a menu item.
	    JMenuItem showHelpWindowItem = new JMenuItem("Show Help Window");
	    showHelpWindowItem.addActionListener(this);
	    helpMenu.add(showHelpWindowItem);
	    //******************************
	    //End - Help menu.
	    //******************************
      
      
	    //Now, add all the menus to the main menu.
	    mainMenu.add(fileMenu);
	    mainMenu.add(helpMenu);
	    setJMenuBar(mainMenu);
      
	    //******************************
	    //End - Code to generate the menus.
	    //******************************
    
	    JPanel rightInnerPanel = new JPanel();
	    JPanel test1 = new JPanel();
      
	    loweredbev = BorderFactory.createLoweredBevelBorder();
	    raisedbev = BorderFactory.createRaisedBevelBorder();
	    empty = BorderFactory.createEmptyBorder();
      
	    connectDisconnectButton = new JButton("Connect to Database");
	    connectDisconnectButton.setFont(new Font("Times-Roman", Font.PLAIN, 12));
	    refreshButton = new JButton("Reload");
	    refreshButton.setFont(new Font("Times-Roman", Font.PLAIN, 12));
	    refreshExpButton = new JButton("Reload ParaProfExperiment");
	    refreshExpButton.setFont(new Font("Times-Roman", Font.PLAIN, 12));
	    serverField = new JTextField("Please enter server address", 30);
	    usernameField = new JTextField("Username", 30);
	    passwordField = new JPasswordField("Password", 30);

	    String operationStrings[] = {"Add", "Subtract", "Multiply", "Divide"};
	    operation = new JComboBox(operationStrings);
      
	    //******************************
	    //Create the tree.
	    //******************************
	    //Create the root node.
	    DefaultMutableTreeNode root = new DefaultMutableTreeNode("ParaProfApplications");
      
	    DefaultMutableTreeNode standard = new DefaultMutableTreeNode(new ParaProfApplicationType("Standard ParaProfApplications"));
	    //Populate this node.
	    this.populateStandardParaProfApplications(standard);
	    root.add(standard);
      
	    DefaultMutableTreeNode runtime = new DefaultMutableTreeNode(new ParaProfApplicationType("Runtime ParaProfApplications"));
	    root.add(runtime);
      
	    dbApps = new DefaultMutableTreeNode(new ParaProfApplicationType("DB ParaProfApplications"));
	    root.add(dbApps);
      
	    treeModel = new DefaultTreeModel(root);
	    treeModel.setAsksAllowsChildren(true);
	    tree = new JTree(treeModel);
	    tree.setRootVisible(false);
	    tree.getSelectionModel().setSelectionMode(TreeSelectionModel.SINGLE_TREE_SELECTION);
	    JRacyTreeCellRenderer renderer = new JRacyTreeCellRenderer();
	    tree.setCellRenderer(renderer);
      
	    //Add tree listeners.
	    tree.addTreeSelectionListener(new TreeSelectionListener(){
		    public void valueChanged(TreeSelectionEvent event){
			treeSelectionEventHandler(event);
		    }
		});
      
	    tree.addTreeExpansionListener(new TreeExpansionListener(){
		    public void treeCollapsed(TreeExpansionEvent event){
			treeCollapsedEventHandler(event);
		    }
		    public void treeExpanded(TreeExpansionEvent event){
			treeExpansionEventHandler(event);
		    }
		}); 
      
	    //Bung it in a scroll pane.
	    JScrollPane treeScrollPane = new JScrollPane(tree); 
	    //******************************
	    //End - Create the tree.
	    //******************************
          
      
	    //Set up the split panes.
	    innerPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, treeScrollPane, getWelcomeUpperRight());
	    innerPane.setContinuousLayout(true);
	    innerPane.setOneTouchExpandable(true);
	    innerPane.setDividerLocation(0.5);
      
	    innerPane2 = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, test1, getApplyOperationComponent());
	    innerPane2.setContinuousLayout(true);
	    innerPane2.setOneTouchExpandable(true);
	    innerPane2.setDividerLocation(0.5);
      
	    outerPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, innerPane, innerPane2);
	    outerPane.setContinuousLayout(true);
	    outerPane.setOneTouchExpandable(true);
      
	    (getContentPane()).add(outerPane, "Center");
      
      
	    connectDisconnectButton.addActionListener(new ActionListener(){
		    public void actionPerformed(ActionEvent evt)
		    {
                        connectDisconnectButtonFunction();
		    }
		});
                    
	    refreshButton.addActionListener(new ActionListener(){
		    public void actionPerformed(ActionEvent evt)
		    {
                        refreshButton();
		    }
		});
                    
	    refreshExpButton.addActionListener(new ActionListener(){
		    public void actionPerformed(ActionEvent evt)
		    {
                        refreshParaProfExperimentsButton();
		    }
		});
    
    
    
	    //Show before setting sliders.
	    //Components have to be realized on the screen before
	    //the sliders can be set.
	    this.show();
	    innerPane.setDividerLocation(0.5);
	    innerPane2.setDividerLocation(0.5);
	    outerPane.setDividerLocation(0.667);
    
	}
	catch(Exception e){
	    e.printStackTrace();
	    ParaProf.systemError(e, null, "jRM01");
	} 
    }
  
    //******************************
    //Tree event handlers.
    //******************************
    private void treeSelectionEventHandler(TreeSelectionEvent event){
	TreePath path = tree.getSelectionPath();
	if(path == null)
	    return;
	DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();            
	Object userObject = selectedNode.getUserObject();
	if(selectedNode.isRoot()){
	    //We are at the root node.  Display some helpful information.
	    innerPane.setRightComponent(getRootUpperRight());
	    innerPane2.setLeftComponent(new JPanel());
	    setJSPDividers(innerPane, innerPane2, 0.5, 0.5);
	}
	else if(userObject instanceof ParaProfApplicationType){
	    String tmpString = userObject.toString();
	    if(parentNode != null){
		if(tmpString.equals("Standard ParaProfApplications")){
		    innerPane.setRightComponent(getSAppUpperRight());
		    innerPane2.setLeftComponent(getAppTypeNodeLowerLeft());
		    setJSPDividers(innerPane, innerPane2, 0.5, 0.5);
		    tree.expandPath(path);
		}
		else if(tmpString.equals("Runtime ParaProfApplications")){
		    innerPane.setRightComponent(getRAppUpperRight());
		    innerPane2.setLeftComponent(getAppTypeNodeLowerLeft());
		    setJSPDividers(innerPane, innerPane2, 0.5, 0.5);
		}
		else{
		    innerPane.setRightComponent(getDBAppUpperRight());
		    innerPane2.setLeftComponent(getDBAppTypeNodeLowerLeft());
		    setJSPDividers(innerPane, innerPane2, 0.5, 0.5);
		}
	    }
	}
	else if(userObject instanceof ParaProfApplication){
	    innerPane.setRightComponent(getAppNodeUpperRight((ParaProfApplication) userObject));
	    innerPane2.setLeftComponent(getBlank());
	    setJSPDividers(innerPane, innerPane2, 0.5, 0.5);
	}
	else if((parentNode.getUserObject()) instanceof ParaProfApplication){
	    innerPane.setRightComponent(getExpTypeNodeUpperRight());
	    innerPane2.setLeftComponent(getExpTypeNodeLowerLeft());
	    setJSPDividers(innerPane, innerPane2, 0.5, 0.5);
	}
	else if((parentNode.getUserObject()) instanceof ParaProfExperiment){
	    innerPane.setRightComponent(getParaProfTrialTypeNodeUpperRight());
	    innerPane2.setLeftComponent(getParaProfTrialTypeNodeLowerLeft());
	    setJSPDividers(innerPane, innerPane2, 0.5, 0.5);
	}
	else if(userObject instanceof ParaProfExperiment){
	    innerPane.setRightComponent(getExpNodeUpperRight());
	    innerPane2.setLeftComponent(getBlank());
	    setJSPDividers(innerPane, innerPane2, 0.5, 0.5);
	}
	else if(userObject instanceof ParaProfTrial){
	    //Would like this node expanded.  Makes more sense.
	    tree.expandPath(path);
	    String tmpString = userObject.toString();
	    //Here the actual clicked on node is an instance of ParaProfTrial (unlike the above
	    //check on ParaProfTrial where it was the parent node).
	    innerPane.setRightComponent(getParaProfTrialNodeUpperRight());
	    innerPane2.setLeftComponent(getBlank());
	    setJSPDividers(innerPane, innerPane2, 0.5, 0.5);
	}
	else if(userObject instanceof Metric){
	    innerPane.setRightComponent(getMetricUpperRight());
	    //Note that the parent user object should be an intance of trial.
	    //Check though!
	    Object tmpObject =  parentNode.getUserObject();
	    if(tmpObject instanceof ParaProfTrial)
		showMetric((ParaProfTrial) tmpObject, (Metric) userObject);
	    else
		ParaProf.systemError(null, null, "jRM02 - Logic Error");
	    innerPane2.setLeftComponent(getBlank());
	    setJSPDividers(innerPane, innerPane2, 0.5, 0.5);
	}
    }
  
    private void treeExpansionEventHandler(TreeExpansionEvent event){
	//Want to automatically expand the experiments and trials place holders.
	//This just makes it easier for the user.
	TreePath path = event.getPath();
	if(ParaProf.debugIsOn)
	    System.out.println(path);
	if(path == null)
	    return;
	DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();            
	Object userObject = selectedNode.getUserObject();
	if((userObject instanceof ParaProfApplication)||(userObject instanceof ParaProfExperiment)){
	    if(selectedNode.getChildCount() > 0){
		tree.expandPath(path.pathByAddingChild(selectedNode.getChildAt(0)));
        
	    }
	}
    }
  
    private void treeCollapsedEventHandler(TreeExpansionEvent event){
    }
  
    //******************************
    //End - Tree event handlers.
    //******************************
  
    //******************************
    //Component functions.
    //******************************
  
    private Component getWelcomeUpperRight(){
	JTextArea tmpJTA = new JTextArea();
	tmpJTA.setLineWrap(true);
	tmpJTA.setWrapStyleWord(true);
	///Set the text.
	tmpJTA.append("ParaProf Manager\n\n");
	tmpJTA.append("This window allows you to manage all of ParaProf's loaded data.\n");
	tmpJTA.append("Data can be static (ie, not updated at runtime),"
		      + " and loaded either remotely or locally.  You can also specify data to be uploaded at runtime.\n\n");
	return (new JScrollPane(tmpJTA));
    }
  
    private Component getSAppUpperRight(){
	JTextArea tmpJTA = new JTextArea();
	tmpJTA.setLineWrap(true);
	tmpJTA.setWrapStyleWord(true);
	///Set the text.
	tmpJTA.append("ParaProf Manager\n\n");
	tmpJTA.append("This is the Standard application section:\n\n");
	tmpJTA.append("Standard - The classic ParaProf mode.  Data sets that are loaded at startup are placed"
		      + "under the default application automatically. Please see the ParaProf documentation for mre details.\n");
	return (new JScrollPane(tmpJTA));
    }
  
    private Component getRAppUpperRight(){
	JTextArea tmpJTA = new JTextArea();
	tmpJTA.setLineWrap(true);
	tmpJTA.setWrapStyleWord(true);
	///Set the text.
	tmpJTA.append("ParaProf Manager\n\n");
	tmpJTA.append("This is the Runtime application section:\n\n");
	tmpJTA.append("Runtime - A new feature allowing ParaProf to update data at runtime.  Please see"
		      + " the ParaProf documentation if the options are not clear.\n");
	return (new JScrollPane(tmpJTA));
    }
  
    private Component getDBAppUpperRight(){
	JTextArea tmpJTA = new JTextArea();
	tmpJTA.setLineWrap(true);
	tmpJTA.setWrapStyleWord(true);
	///Set the text.
	tmpJTA.append("ParaProf Manager\n\n");
	tmpJTA.append("This is the DB Apps application section:\n\n");
	tmpJTA.append("DB Apps - Another new feature allowing ParaProf to load data from a database.  Again, please see"
		      + " the ParaProf documentation if the options are not clear.\n");
	return (new JScrollPane(tmpJTA));
    }
  
    private Component  getExpTypeNodeUpperRight(){
	JTextArea tmpJTA = new JTextArea();
	tmpJTA.setLineWrap(true);
	tmpJTA.setWrapStyleWord(true);
	///Set the text.
	tmpJTA.append("ParaProf Manager\n\n");
	tmpJTA.append("The button in the lower left allows you to add an experiment. Other features will be added here soon.\n");
	return (new JScrollPane(tmpJTA));
    }
  
  
    private Component  getParaProfTrialTypeNodeUpperRight(){
	JTextArea tmpJTA = new JTextArea();
	tmpJTA.setLineWrap(true);
	tmpJTA.setWrapStyleWord(true);
	///Set the text.
	tmpJTA.append("ParaProf Manager\n\n");
	tmpJTA.append("The buttons in the lower left allow you to add trials to this experiment.  You can either add"
		      + " a trial with a single counter (top button) or you can select to add a trial with multiple counters\n"
		      + " (bottom button).  You will be prompted for a name for the trial, and then you can select either a \n"
		      + " pprof dump file, or a directory, depending on which button you clicked on.\n");
	return (new JScrollPane(tmpJTA));
    }
  
    private Component  getExpNodeUpperRight(){
	JTextArea tmpJTA = new JTextArea();
	tmpJTA.setLineWrap(true);
	tmpJTA.setWrapStyleWord(true);
	///Set the text.
	tmpJTA.append("ParaProf Manager\n\n");
	tmpJTA.append("If this node is not expanded, you may double click on it to see the trials in this experiment.\n");
	return (new JScrollPane(tmpJTA));
    }
  
    private Component  getParaProfTrialNodeUpperRight(){
	JTextArea tmpJTA = new JTextArea();
	tmpJTA.setLineWrap(true);
	tmpJTA.setWrapStyleWord(true);
	///Set the text.
	tmpJTA.append("ParaProf Manager\n\n");
	tmpJTA.append("If this node is not expanded, you may double click on it to see the metrics in this trial.\n");
	return (new JScrollPane(tmpJTA));
    }
  
    private Component getAppTypeNodeLowerLeft(){
	JButton tmpJButton = null;
	JPanel tmpJPanel = new JPanel();
	tmpJButton = new JButton("Add ParaProfApplication");
	tmpJButton.addActionListener(new ActionListener(){
                public void actionPerformed(ActionEvent evt){
		    newParaProfApplicationButton();}
	    });
	tmpJPanel.add(tmpJButton);
	return tmpJPanel;
    }
  
    private Component getExpTypeNodeLowerLeft(){
	JButton tmpJButton = null;
	JPanel tmpJPanel = new JPanel();
	tmpJButton = new JButton("Add ParaProfExperiment");
	tmpJButton.addActionListener(new ActionListener(){
                public void actionPerformed(ActionEvent evt){
		    addParaProfExperimentButtonFunction();}
	    });
	tmpJPanel.add(tmpJButton);
	return tmpJPanel;
    }
  
    private Component getParaProfTrialTypeNodeLowerLeft(){
	JButton tmpJButton = null;
	JPanel tmpJPanel = new JPanel();
	tmpJButton = new JButton("Add Single Metric Trail");
	tmpJButton.addActionListener(new ActionListener(){
                public void actionPerformed(ActionEvent evt){
		    addParaProfTrialSVTButtonFunction();}
	    });
	tmpJPanel.add(tmpJButton);
      
	tmpJButton = new JButton("Add Multiple Metric Trail");
	tmpJButton.addActionListener(new ActionListener(){
                public void actionPerformed(ActionEvent evt){
		    addParaProfTrialMVTButtonFunction();}
	    });
	tmpJPanel.add(tmpJButton);
      
	return tmpJPanel;
    }
  
    private Component getDBAppTypeNodeLowerLeft(){
	JButton tmpJButton = null;
	JPanel tmpJPanel = new JPanel();
	tmpJButton = new JButton("Add ParaProfApplication");
	tmpJButton.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent evt){
		    newParaProfApplicationButton();}
	    });

	JLabel serverFieldLabel = new JLabel("Server Address:");
	JLabel usernameFieldLabel = new JLabel("Password:");
	JLabel passwordFieldLabel = new JLabel("Username:");
    
    
	//Now add the components to the panel.
	GridBagLayout gbl = new GridBagLayout();
	tmpJPanel.setLayout(gbl);
	GridBagConstraints gbc = new GridBagConstraints();
	gbc.insets = new Insets(5, 5, 5, 5);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, tmpJButton, gbc, 0, 0, 1, 1);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.EAST;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, serverFieldLabel, gbc, 0, 1, 1, 1);
    
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 100;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, serverField, gbc, 1, 1, 1, 1);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.EAST;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, usernameFieldLabel, gbc, 0, 2, 1, 1);
    
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 100;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, usernameField, gbc, 1, 2, 1, 1);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.EAST;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, passwordFieldLabel, gbc, 0, 3, 1, 1);
    
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 100;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, passwordField, gbc, 1, 3, 1, 1);
    
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, connectDisconnectButton, gbc, 0, 4, 1, 1);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, refreshButton, gbc, 1, 4, 1, 1);
    
	return tmpJPanel;
    }
  
    private Component getRootUpperRight(){
	JTextArea tmpJTA = new JTextArea();
	tmpJTA.setLineWrap(true);
	tmpJTA.setWrapStyleWord(true);
	///Set the text.
	tmpJTA.append("ParaProf Manager\n\n");
	tmpJTA.append("There are three application types:\n\n");
	tmpJTA.append("Standard - The classic ParaProf mode.  At present, there is only a \"default\" set"
		      + " which is set for you automatically.  This maintains compatability with older versions where the"
		      + " application set was not definied.  This grouping will be expanded in time. For now, you may add"
		      + " experiments under the default set only.\n\n");
	tmpJTA.append("Runtime - A new feature allowing ParaProf to update data at runtime.  Please see"
		      + " the ParaProf documentation if the options are not clear."
		      + " As in the Standard set, there is only section.\n\n");
	tmpJTA.append("DB Apps - Another new feature allowing ParaProf to load data from a database.  Again, please see"
		      + " the ParaProf documentation if the options are not clear.\n\n");
	return (new JScrollPane(tmpJTA));
    }
  
    private Component getAppNodeUpperRight(ParaProfApplication inApp){
	return (new JScrollPane(new JTable(new JRacyTableModel(inApp))));
    }
  
    private Component getMetricUpperRight(){
	JTextArea tmpJTA = new JTextArea();
	tmpJTA.setLineWrap(true);
	tmpJTA.setWrapStyleWord(true);
	///Set the text.
	tmpJTA.append("ParaProf Manager\n\n");
	tmpJTA.append("Clicking on different metrics causes ParaProf to display the clicked on metric.\n\n");
	tmpJTA.append("The sub-window below allow you to generate new metrics based on those that were"
		      + " gathered during the run.  The operand number options for Operand A and B correspond the numbers prefixing the metrics.\n");
	return (new JScrollPane(tmpJTA));
    }
  
    private Component getAppNodeLowerLeft(ParaProfApplication inApp){
	JButton tmpJButton = null;
	JPanel tmpJPanel = new JPanel();
      
	tmpJButton = new JButton("Update ParaProfApplication");
	tmpJButton.addActionListener(new ActionListener(){
                public void actionPerformed(ActionEvent evt){
		    updateParaProfApplicationButton();}
	    });
	tmpJPanel.add(tmpJButton);
      
	tmpJButton = new JButton("Add ParaProfExperiment");
	tmpJButton.addActionListener(new ActionListener(){
                public void actionPerformed(ActionEvent evt){
                }
	    });
	tmpJPanel.add(tmpJButton);
	return tmpJPanel;
    }
  
    private Component getApplyOperationComponent(){
	JPanel tmpJPanel = new JPanel();
    
	JButton tmpJButton = new JButton("Apply Operation");
	tmpJButton.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent evt){
		    applyOperationButtonFunction();}
	    });
    
	//Now add the components to the panel.
	GridBagLayout gbl = new GridBagLayout();
	tmpJPanel.setLayout(gbl);
	GridBagConstraints gbc = new GridBagConstraints();
	gbc.insets = new Insets(5, 5, 5, 5);
    
	JLabel opALabel = new JLabel("Op A");
	JLabel opBLabel = new JLabel("Op B:");
	JLabel opInstrLabel = new JLabel("Apply operations here!");
	JLabel opLabel = new JLabel("Operation");
    
	opA = new JTextField("XXXX - XXXX...            ", 30);
	opB = new JTextField("XXXX _ XXXX...            ", 30);
    
	gbc.fill = GridBagConstraints.VERTICAL;
	gbc.anchor = GridBagConstraints.NORTH;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, opInstrLabel, gbc, 0, 0, 1, 1);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, opALabel, gbc, 0, 1, 1, 1);
    
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 1;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, opA, gbc, 1, 1, 2, 1);
    
	gbc.fill = GridBagConstraints.CENTER;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, opBLabel, gbc, 0, 2, 1, 1);
    
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 1;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, opB, gbc, 1, 2, 2, 1);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, opLabel, gbc, 0, 3, 1, 1);
    
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 1;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, operation, gbc, 1, 3, 2, 1);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.SOUTH;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, tmpJButton, gbc, 0, 4, 3, 1);
    
	return tmpJPanel;
    }
  
    private Component getBlank(){
	return new JPanel();
    }
  
    //******************************
    //End - Component functions.
    //******************************
  
    private void setJSPDividers(JSplitPane inJSP1, JSplitPane inJSP2, double inDouble1, double inDouble2){
	inJSP1.setDividerLocation(inDouble1);
	inJSP2.setDividerLocation(inDouble2);
    }
  
    void addParaProfExperimentButtonFunction(){
    
	JOptionPane.showMessageDialog(this, "Only the default experiment allowed in this release!", "Warning!"
                                      ,JOptionPane.ERROR_MESSAGE);
	return;
    }
    void addParaProfTrialSVTButtonFunction()
    {
    
	try{
	    ParaProfTrial trial = null;
	    String tmpString1 = null;
	    String tmpString2 = null;
	    String tmpString3 = null;
      
	    //Get the selected trial placeholder, and then its parent experiment node.
	    TreePath path = tree.getSelectionPath();
	    if(path == null){
		System.out.println("Error adding trial ... aborted.");
		return;}
	    DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	    DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
	    if(parentNode == null){
		System.out.println("Error adding trial ... aborted.");
		return;}
	    Object userObject = parentNode.getUserObject();
	    if(!(userObject instanceof ParaProfExperiment)){
		System.out.println("Error adding trial ... aborted.");
		return;}
        
	    ParaProfExperiment exp = (ParaProfExperiment) userObject;
      
	    //First get the name of the new run.
	    String newTrailName = JOptionPane.showInputDialog(this, "Please enter a new trial name, click ok, and then select a pprof dump file!");
	    if((newTrailName == null) || "".equals(newTrailName)){
		JOptionPane.showMessageDialog(this, "You must enter a name!", "Error!"
					      ,JOptionPane.ERROR_MESSAGE);
		return;
	    }
	    //Create a file chooser to allow the user to select the pprof dump file.
	    JFileChooser pprofDumpFileChooser = new JFileChooser();
	    //Set the directory to the current directory.
	    pprofDumpFileChooser.setCurrentDirectory(null);
	    //Bring up the file chooser.
	    int resultValue = pprofDumpFileChooser.showOpenDialog(this);
      
	    if(resultValue == JFileChooser.APPROVE_OPTION)
		{
		    //Try and get the file name.
		    File file = pprofDumpFileChooser.getSelectedFile();
        
		    //Test to see if valid.
		    if(file != null)
			{ 
			    tmpString1 = file.getCanonicalPath();
			    tmpString2 = ParaProf.applicationManager.getPathReverse(tmpString1);
			    tmpString3 = newTrailName + " : " + tmpString2;
          
			    //Pop up the dialog if there is already an experiment with this name.
			    if(exp.isParaProfTrialNamePresent(tmpString3)){
				JOptionPane.showMessageDialog(this, "A run already exists with that name!", "Warning!"
							      ,JOptionPane.ERROR_MESSAGE);
				return;
			    }
          
			    trial = exp.addParaProfTrial();
			    DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(trial);
			    trial.setDMTN(trialNode);
			    trial.setProfilePathName(tmpString1);
			    trial.setParaProfTrialName(tmpString3);
			    trial.buildStaticData(file);
          
			    //Now update the tree.
			    //Populate the metrics list for this trial.
			    for(Enumeration e = (trial.getMetrics()).elements(); e.hasMoreElements() ;){
				Metric metric = (Metric) e.nextElement();
				DefaultMutableTreeNode metricNode = new DefaultMutableTreeNode(metric);
				metric.setDMTN(metricNode);
				metricNode.setAllowsChildren(false);
				trialNode.add(metricNode);
			    }
			    treeModel.insertNodeInto(trialNode, selectedNode, selectedNode.getChildCount());
			}
		    else
			{
			    System.out.println("There was some sort of internal error!");
			    return;
			}
		}
	}
	catch(Exception e)
	    {
      
		ParaProf.systemError(e, null, "ELM01");
	    }
    }
  
    void addParaProfTrialMVTButtonFunction()
    {
    
	try{
	    ParaProfTrial trial = null;
	    DefaultMutableTreeNode trialNode = null;
	    String tmpString1 = null;
	    String tmpString2 = null;
	    String tmpString3 = null;
      
	    //Get the selected trial placeholder, and then its parent experiment node.
	    TreePath path = tree.getSelectionPath();
	    if(path == null){
		System.out.println("Error adding trial ... aborted.");
		return;}
	    DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	    DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
	    if(parentNode == null){
		System.out.println("Error adding trial ... aborted.");
		return;}
	    Object userObject = parentNode.getUserObject();
	    if(!(userObject instanceof ParaProfExperiment)){
		System.out.println("Error adding trial ... aborted.");
		return;}
        
	    ParaProfExperiment exp = (ParaProfExperiment) userObject;
      
	    //First get the name of the new run.
	    String newTrailName = JOptionPane.showInputDialog(this, "Please enter a new trial name, click ok, and then select a pprof dump file!");
	    if((newTrailName == null) || "".equals(newTrailName)){
		JOptionPane.showMessageDialog(this, "You must enter a name!", "Error!"
					      ,JOptionPane.ERROR_MESSAGE);
		return;
	    }
	    //Create a file chooser to allow the user to select the pprof dump file.
	    JFileChooser pprofDumpDirectoryChooser = new JFileChooser();
	    pprofDumpDirectoryChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
	    //Set the directory to the current directory.
	    pprofDumpDirectoryChooser.setCurrentDirectory(null);
	    //Bring up the file chooser.
	    int resultValue = pprofDumpDirectoryChooser.showOpenDialog(this);
      
	    if(resultValue == JFileChooser.APPROVE_OPTION)
		{
		    boolean foundSomething = false;
		    //Try and get the file name.
		    File file = pprofDumpDirectoryChooser.getSelectedFile();
        
		    //Test to see if valid.
		    if(file != null)
			{ 
			    String filePath = file.getCanonicalPath();
			    File [] list = file.listFiles();
			    for(int i = 0; i < list.length; i++)
				{
				    File tmpFile = (File) list[i];
				    if(tmpFile != null){
					String tmpString = tmpFile.getName();
              
					if(tmpString.indexOf("MULTI__") != -1){
					    String newString = filePath + "/" + tmpString + "/pprof.dat";
					    File testFile = new File(newString);
                
					    if(testFile.exists()){
						if(!foundSomething){
						    System.out.println("Found pprof.dat ... loading");
                    
						    tmpString1 = filePath;
						    tmpString2 = ParaProf.applicationManager.getPathReverse(tmpString1);
						    tmpString3 = newTrailName + " : " + tmpString2;
                                                
						    trial = exp.addParaProfTrial();
						    trialNode = new DefaultMutableTreeNode(trial);
						    trial.setDMTN(trialNode);
                        
						    trial.setProfilePathName(tmpString1);
						    trial.setParaProfTrialName(tmpString3);
						    trial.buildStaticData(testFile);
                    
						    System.out.println("Found: " + newString);
                    
						    foundSomething = true;
						}
						else{
						    trial.buildStaticData(testFile);
						} 
					    }
					}
				    }
				}
          
			    //Now update the tree.
			    //Populate the metrics list for this trial.
			    for(Enumeration e = (trial.getMetrics()).elements(); e.hasMoreElements() ;){
				Metric metric = (Metric) e.nextElement();
				DefaultMutableTreeNode metricNode = new DefaultMutableTreeNode(metric);
				metric.setDMTN(metricNode);
				metricNode.setAllowsChildren(false);
				trialNode.add(metricNode);
			    }
			    treeModel.insertNodeInto(trialNode, selectedNode, selectedNode.getChildCount());
			}
		    else
			{
			    System.out.println("There was some sort of internal error!");
			    return;
			}
		}
	}
	catch(Exception e)
	    {
      
		ParaProf.systemError(e, null, "ELM01");
	    }
    }
  
    public void connectDisconnectButtonFunction(){
    
	if(!ParaProf.dbSupport){
	    JOptionPane.showMessageDialog(this, "Sorry, in this release db support is not turned on!", "Warning!"
					  ,JOptionPane.ERROR_MESSAGE);
	    return;
	}
    
	try{
	    //Try making a connection to the database.
	    if(connectDisconnectButton.getText().equals("Connect to Database")){
		String serverAddress = serverField.getText().trim();
		String username = usernameField.getText().trim();
		String password = new String(passwordField.getPassword());
		password.trim();
		//ConnectionManager.connect(serverAddress, username, password);
		connectDisconnectButton.setText("Disconnect from Database");
	    }
	    else{
		//ConnectionManager.dbclose();
		connectDisconnectButton.setText("Connect to Database");
	    }
	}
	catch(Exception e)
	    {
      
		ParaProf.systemError(e, null, "ELM03");
	    }
    }
  
    public void refreshButton(){  
	if(!ParaProf.dbSupport){
	    JOptionPane.showMessageDialog(this, "Sorry, in this release db support is not turned on!", "Warning!"
					  ,JOptionPane.ERROR_MESSAGE);
	    return;
	}
    
	/*
	  try{
	  //Clear the node before adding anything else.
	  //Works better than the enumeration ... well, just works.
	  for(int i=dbApps.getChildCount(); i>0; i--){
	  treeModel.removeNodeFromParent(((DefaultMutableTreeNode) dbApps.getChildAt(i-1)));
	  }
      
	  ResultSet result;
	  ResultSet resultExp;
      
	  //Excecute query.
	  String query = new String("select * from applications; ");
	  //result = ConnectionManager.getDB().executeQuery(query);
      
	  //int columnNum = result.getMetaData().getColumnCount();
	  int counter = 0;
	  while (result.next()){
	  ParaProfApplication tmpApp = new ParaProfApplication();
	  tmpApp.setDBParaProfApplication(true);
	  for(int i=1;i<=columnNum; i++){
          String returnString = result.getString(i);
          switch(i){
          case(1):
	  tmpApp.setParaProfApplicationID(Integer.parseInt(returnString));
	  break;
          case(2):
	  tmpApp.setParaProfApplicationName(returnString);
	  break;
          case(3):
	  tmpApp.setVersion(returnString);
	  break;
          case(4):
	  tmpApp.setDescription(returnString);
	  break;
          case(5):
	  tmpApp.setLanguage(returnString);
	  break;
          case(6):
	  tmpApp.setPara_diag(returnString);
	  break;
          case(7):
	  tmpApp.setUsage(returnString);
	  break;
          case(8):
	  tmpApp.setExe_opt(returnString);
	  break;
          default:
	  System.out.println("Error in application information parsing" + i);
          }
	  }
	  //Now add the application to the node.
	  treeModel.insertNodeInto(new DefaultMutableTreeNode(tmpApp), dbApps, dbApps.getChildCount());
	  }
	  result.close();
	  return;
	  }
	  catch(Exception e){
	  ParaProf.systemError(e, null, "ELM03");
	  }
	*/
    }
  
  
    public void refreshParaProfExperimentsButton(){
	/*try{
    
	TreePath path = tree.getSelectionPath();
	if(path == null)
	return;
	DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	Object userObject = selectedNode.getUserObject();
      
	if(userObject instanceof ParaProfApplication){
        int applicationID = ((ParaProfApplication) userObject).getParaProfApplicationID(); 
        
        //Clear the node before adding anything else.
        //for(int i=selectedNode.getChildCount(); i>0; i--){
	//treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i-1)));
        //}
      
        ResultSet result;
        //Execute query.
        String query = new String("select * from experiments where appID = '" + applicationID + "';");
        //result = ConnectionManager.getDB().executeQuery(query);
        
        //int columnNum = result.getMetaData().getColumnCount();
        int counter = 0;
        
        while (result.next()){
	ParaProfApplication tmpApp = new ParaProfApplication();
	System.out.println("@@@@@@@@@@@@@ParaProfApplication: " + counter++);
	for(int i=1;i<=1; i++){
	String returnString = result.getString(i);
	System.out.println("ParaProfExperiment: " + returnString);
	}
        }
        
        result.close();
        return;
        
	}
	catch(Exception e){
	ParaProf.systemError(e, null, "ELM03");
	}
	*/
    }
  
  
    void applyOperationButtonFunction()
    {
    
	try{      
      
	    int tmpInt1 = -1;
	    int tmpInt2 = -1;
      
	    String tmpString1 = opA.getText().trim();
	    String tmpString2 = opB.getText().trim();
      
	    if(!((tmpString1.substring(0,3)).equals(tmpString2.substring(0,3)))){
		JOptionPane.showMessageDialog(this, "Sorry, in this release, please select from the same trial!", "Warning!"
					      ,JOptionPane.ERROR_MESSAGE);
		return;
	    }
      
	    //Get the application.
	    tmpInt1 = Character.getNumericValue(tmpString1.charAt(0));
	    ParaProfApplication tmpApp = (ParaProfApplication) ParaProf.applicationManager.getApplications().elementAt(tmpInt1);
	    //Get the experiment.
	    tmpInt1 = Character.getNumericValue(tmpString1.charAt(1));
	    ParaProfExperiment tmpExp = (ParaProfExperiment) tmpApp.getExperiments().elementAt(tmpInt1);
	    //Get the trial.
	    tmpInt1 = Character.getNumericValue(tmpString1.charAt(2));
	    ParaProfTrial tmpParaProfTrial = (ParaProfTrial) tmpExp.getTrials().elementAt(tmpInt1);
      
	    //Get Metrics.
	    tmpInt1 = Character.getNumericValue(tmpString1.charAt(3));
	    Metric tmpMetric1 = (Metric) tmpParaProfTrial.getMetrics().elementAt(tmpInt1);
	    tmpInt2 = Character.getNumericValue(tmpString2.charAt(3));
	    Metric tmpMetric2 = (Metric) tmpParaProfTrial.getMetrics().elementAt(tmpInt2);
      
	    tmpString1 = tmpString1.substring((tmpString1.indexOf('-')) + 2);
	    tmpString2 = tmpString2.substring((tmpString2.indexOf('-')) + 2);
      
	    String tmpString3 = (String) operation.getSelectedItem();
      
	    Metric newMetric = PPML.applyOperation(tmpParaProfTrial, tmpMetric1.getName(), tmpMetric2.getName(), tmpString3);
	    DefaultMutableTreeNode metricNode = new DefaultMutableTreeNode(newMetric);
	    newMetric.setDMTN(metricNode);
      
	    DefaultMutableTreeNode parentNode = tmpParaProfTrial.getDMTN();
      
      
	    metricNode.setAllowsChildren(false);
	    treeModel.insertNodeInto(metricNode, parentNode, parentNode.getChildCount());
      
	    tmpParaProfTrial.getSystemEvents().updateRegisteredObjects("dataEvent");
	}
	catch(Exception e)
	    {
      
		ParaProf.systemError(e, null, "ELM03");
	    }
    }
  
    void showMetric(ParaProfTrial inParaProfTrial, Metric inMetric)
    {
	try{
	    //Update the operation text fields.  Makes it easier for the user.
	    opB.setText(opA.getText().trim());
	    opA.setText(inMetric.toString());
	    inParaProfTrial.setCurValLoc(inMetric.getID());
	    inParaProfTrial.getSystemEvents().updateRegisteredObjects("dataEvent");
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "jRM04");
	}
    }
  
    void populateStandardParaProfApplications(DefaultMutableTreeNode inNode){
    
	DefaultMutableTreeNode applicationNode = null;
	DefaultMutableTreeNode placeHolderParaProfExperiments = null;
	DefaultMutableTreeNode experimentNode = null;
	DefaultMutableTreeNode placeHolderParaProfTrials = null;
	DefaultMutableTreeNode trialNode = null;
	DefaultMutableTreeNode metricNode = null;
    
    
	int cnt1 = 0;
	int cnt2 = 0;
	int cnt3 = 0;
	int cnt4 = 0;
	for(Enumeration e1 = (ParaProf.applicationManager.getApplications()).elements(); e1.hasMoreElements() ;){
	    ParaProfApplication application = (ParaProfApplication) e1.nextElement();
	    applicationNode = new DefaultMutableTreeNode(application);
	    application.setDMTN(applicationNode);
      
	    placeHolderParaProfExperiments = new DefaultMutableTreeNode("ParaProfExperiments");
	    for(Enumeration e2 = (application.getExperiments()).elements(); e2.hasMoreElements() ;){  
		ParaProfExperiment exp = (ParaProfExperiment) e2.nextElement();
		experimentNode = new DefaultMutableTreeNode(exp);
		exp.setDMTN(experimentNode);
        
		//Populate the trials for this experiemnt.
		placeHolderParaProfTrials = new DefaultMutableTreeNode("ParaProfTrials");
		for(Enumeration e3 = (exp.getTrials()).elements(); e3.hasMoreElements() ;){
		    ParaProfTrial trial = (ParaProfTrial) e3.nextElement();
		    trialNode = new DefaultMutableTreeNode(trial);
		    trial.setDMTN(trialNode);
          
		    //Populate the metrics list for this trial.
		    for(Enumeration e4 = (trial.getMetrics()).elements(); e4.hasMoreElements() ;){
			Metric metric = (Metric) e4.nextElement();
			metricNode = new DefaultMutableTreeNode(metric);
			metric.setDMTN(metricNode);
			metricNode.setAllowsChildren(false);
			trialNode.add(metricNode);
			cnt4++;
		    }
		    placeHolderParaProfTrials.add(trialNode);
		    if(cnt3 == 0)
			defaultParaProfTrialNode = trialNode;
		    cnt3++;
		}
		experimentNode.add(placeHolderParaProfTrials);
		placeHolderParaProfExperiments.add(experimentNode);
		cnt2++;
	    }
	    applicationNode.add(placeHolderParaProfExperiments);
	    inNode.add(applicationNode);
	    cnt1++;
	}
    } 
  
    public void expandDefaultParaProfTrialNode(){
	if(defaultParaProfTrialNode != null)
	    tree.expandPath(new TreePath(defaultParaProfTrialNode.getPath()));
    }
  
  
    //******************************
    //Manage the applications.
    //******************************
    public void newParaProfApplicationButton()
    {
	JOptionPane.showMessageDialog(this, "Only the default application allowed in this release!", "Warning!"
                                      ,JOptionPane.ERROR_MESSAGE);
	return;
    
	/*
	  TreePath path = tree.getSelectionPath();
	  if(path == null)
	  return;
	  DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	  DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();            
	  Object userObject = selectedNode.getUserObject();
    
	  //Ok, now check to make sure that the parent node is the root.
	  if(parentNode != null){
	  //Not null, therfore, can continue.
	  if(parentNode.isRoot()){
	  String tmpString = userObject.toString();
	  if(tmpString.equals("Standard ParaProfApplications")){
          treeModel.insertNodeInto(new DefaultMutableTreeNode(ParaProf.applicationManager.addParaProfApplication()),
	  selectedNode, selectedNode.getChildCount());
	  }
	  else if(tmpString.equals("Runtime ParaProfApplications")){
	  }
	  else{
	  }
	  }
	  }*/
    }
  
    public void updateParaProfApplicationButton()
    {
	TreePath path = tree.getSelectionPath();
	if(path == null)
	    return;
	DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();            
	Object userObject = selectedNode.getUserObject();
    
	//Ok, now check to make sure that the parent node is the root.
	if(parentNode != null){
	    //Not null, therfore, can continue.
	    if(parentNode.isRoot()){
		String tmpString = userObject.toString();
		if(tmpString.equals("Standard ParaProfApplications")){
		    treeModel.insertNodeInto(new DefaultMutableTreeNode(ParaProf.applicationManager.addParaProfApplication()),
					     selectedNode, selectedNode.getChildCount());
		}
		else if(tmpString.equals("Runtime ParaProfApplications")){
		}
		else{
		}
	    }
	}
    }
  
  
  
  
  
    //******************************
    //Event listener code!!
    //******************************
  
    //ActionListener code.
    public void actionPerformed(ActionEvent evt)
    {
	try{
	    Object EventSrc = evt.getSource();
      
	    if(EventSrc instanceof JMenuItem)
		{
		    String arg = evt.getActionCommand();
        
		    if(arg.equals("Exit ParaProf!"))
			{
			    setVisible(false);
			    dispose();
			    System.exit(0);
			} 
		    else if(arg.equals("ParaProf Manager"))
			{
			    if(!(ParaProf.runHasBeenOpened)){
				setVisible(false);
				dispose();
				System.out.println("Quiting ParaProf!");
				System.exit(0);
			    }
			    else{
				dispose();
			    }
			}
		    else if(arg.equals("About Racy"))
			{
			    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
			}
		    else if(arg.equals("Show Help Window"))
			{
			    //Show the ParaProf help window.
			    ParaProf.helpWindow.clearText();
			    ParaProf.helpWindow.show();
          
			    ParaProf.helpWindow.writeText("This is the experiment manager window.");
			    ParaProf.helpWindow.writeText("");
			    ParaProf.helpWindow.writeText("You can create an experiment, and then add separate runs,.");
			    ParaProf.helpWindow.writeText("which may contain one or more metrics (gettimeofday, cache misses, etc.");
			    ParaProf.helpWindow.writeText("You can also derive new metrics in this window.");
			    ParaProf.helpWindow.writeText("");
			    ParaProf.helpWindow.writeText("Please see ParaProf's documentation for more information.");
			}
		}
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "ELM05");
	    }
    }
  
    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e)
    {
	closeThisWindow();
    }
  
    void closeThisWindow()
    { 
	try
	    {
		if(ParaProf.debugIsOn)
		    {
			System.out.println("------------------------");
			System.out.println("ParaProfExperiment List Manager Window is closing!");
			System.out.println("Clearing resourses for this window.");
		    }
      
		setVisible(false);
		dispose();
	    }
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "ELM06");
	    }
    }
  
    private void panelAdd(JPanel inJPanel, Component c, GridBagConstraints gbc, int x, int y, int w, int h)
    {
	gbc.gridx = x;
	gbc.gridy = y;
	gbc.gridwidth = w;
	gbc.gridheight = h;
    
	inJPanel.add(c, gbc);
    }
  
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h)
    {
	gbc.gridx = x;
	gbc.gridy = y;
	gbc.gridwidth = w;
	gbc.gridheight = h;
    
	getContentPane().add(c, gbc);
    }
  
    //Instance data.
    JTree tree = null;
    DefaultTreeModel treeModel = null;
    DefaultMutableTreeNode dbApps = null;
    JSplitPane innerPane = null;
    JSplitPane innerPane2 = null;
    JSplitPane outerPane = null;
  
    //A reference to the default trial node.
    DefaultMutableTreeNode defaultParaProfTrialNode = null;
  
    private JButton connectDisconnectButton = null;
    private JButton refreshButton = null;
    private JButton refreshExpButton = null;
    private JTextField serverField = null;
    private JTextField usernameField = null;
    private JPasswordField passwordField = null;
  
    private JComboBox operation = null;
    private JTextField opA = null;
    private JTextField opB = null;
  
    private Border loweredbev = null;
    private Border raisedbev = null;
    private Border empty = null;
}

class JRacyTreeCellRenderer extends DefaultTreeCellRenderer{
    public Component getTreeCellRendererComponent(JTree tree,
						  Object value,
						  boolean selected,
						  boolean expanded,
						  boolean leaf,
						  int row,
						  boolean hasFocus){
	super.getTreeCellRendererComponent(tree,value,selected,expanded,leaf,row,hasFocus);
	DefaultMutableTreeNode node = (DefaultMutableTreeNode) value;
	Object userObject = node.getUserObject();
	if(userObject instanceof ParaProfApplicationType){
	    this.setIcon(new ImageIcon("red-ball.gif"));
	}
	else if(userObject instanceof String){
	    String userObjectString = userObject.toString();
	    if(userObjectString.equals("ParaProfExperiments")){
		this.setIcon(new ImageIcon("blue-ball.gif"));
	    }
	    else if(userObjectString.equals("ParaProfTrials")){
		this.setIcon(new ImageIcon("yellow-ball.gif"));
	    }
	}
	else if(userObject instanceof Metric){
	    this.setIcon(new ImageIcon("green-ball.gif"));
	}
    
	return this;
    }
}

class JRacyTableModel extends AbstractTableModel{
  
    public JRacyTableModel(ParaProfApplication inApp){
	super();
	app = inApp;
    }
  
    public int getColumnCount(){
	return 2;
    }
  
    public int getRowCount(){
	return 7;
    }
  
    public String getColumnName(int c){
	return columnNames[c];
    }
  
    public Object getValueAt(int r, int c){
	Object returnObject = null;
	if(c==0){
	    switch(r){
	    case(0):
		returnObject = "Name";
		break;
	    case(1):
		returnObject = "ID";
		break;
	    case(2):
		returnObject = "Language";
		break;
	    case(3):
		returnObject = "Para_diag";
		break;
	    case(4):
		returnObject = "Usage";
		break;
	    case(5):
		returnObject = "Executable Options";
		break;
	    case(6):
		returnObject = "Description";
		break;
	    }
	}
	else{
	    switch(r){
	    case(0):
		returnObject = app.getParaProfApplicationName();
		break;
	    case(1):
		returnObject = new Integer(app.getID());
		break;
	    case(2):
		returnObject = app.getLanguage();
		break;
	    case(3):
		returnObject = app.getParaDiag();
		break;
	    case(4):
		returnObject = app.getUsage();
		break;
	    case(5):
		returnObject = app.getExecutableOptions();
		break;
	    case(6):
		returnObject = app.getDescription();
		break;
	    }
	}
    
	return returnObject; 
          
    }
  
    public boolean isCellEditable(int r, int c){
	boolean tmpBoolean = false;
	if(c==1 && r!=1)
	    tmpBoolean = true;
	return tmpBoolean;
    }
  
    public void setValueAt(Object obj, int r, int c){
	//Should be getting a string I think.
	if(obj instanceof String){
	    String tmpString = (String) obj;
	    if(c==1){
		switch(r){
		case(0):
		    app.setName(tmpString);
		    break;
		case(1):
		    app.setID(Integer.parseInt(tmpString));
		    break;
		case(2):
		    app.setLanguage(tmpString);
		    break;
		case(3):
		    app.setParaDiag(tmpString);
		    break;
		case(4):
		    app.setUsage(tmpString);
		    break;
		case(5):
		    app.setExecutableOptions(tmpString);
		    break;
		case(6):
		    app.setDescription(tmpString);
		    break;
		}
	    }
	}
    }
  
    private ParaProfApplication app = null;
    String[] columnNames = {
	"Field", "Value"
    };
  
}

class ParaProfApplicationType{
    public ParaProfApplicationType(){}
  
    public ParaProfApplicationType(String inString){
	name = inString;}
  
    public String toString(){
	return name;}
  
    String name = "Name Not Set";
}
