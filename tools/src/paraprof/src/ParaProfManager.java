
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
import dms.dss.*;
import java.net.*;

public class ParaProfManager extends JFrame implements ActionListener{
    public ParaProfManager(){
	
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
	    JMenuItem aboutItem = new JMenuItem("About ParaProf");
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
      
	    String operationStrings[] = {"Add", "Subtract", "Multiply", "Divide"};
	    operation = new JComboBox(operationStrings);
      
	    //******************************
	    //Create the tree.
	    //******************************
	    //Create the root node.
	    DefaultMutableTreeNode root = new DefaultMutableTreeNode("Applications");
      
	    DefaultMutableTreeNode standard = new DefaultMutableTreeNode(new PlaceHolder("Standard Applications", 0, 0));
	    //Populate this node.
	    this.populateStandardApplications(standard);
	    root.add(standard);
      
	    DefaultMutableTreeNode runtime = new DefaultMutableTreeNode(new PlaceHolder("Runtime Applications", 0, 1));
	    root.add(runtime);
      
	    dbApps = new DefaultMutableTreeNode(new PlaceHolder("DB Applications", 0, 2));
	    root.add(dbApps);
      
	    treeModel = new DefaultTreeModel(root);
	    treeModel.setAsksAllowsChildren(true);
	    tree = new JTree(treeModel);
	    tree.setRootVisible(false);
	    tree.getSelectionModel().setSelectionMode(TreeSelectionModel.SINGLE_TREE_SELECTION);
	    ParaProfTreeCellRenderer renderer = new ParaProfTreeCellRenderer();
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
	    innerPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, treeScrollPane, getPanelHelpMessage(0));
	    innerPane.setContinuousLayout(true);
	    
	    outerPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, innerPane, new JPanel());
	    outerPane.setContinuousLayout(true);
	          
	    (getContentPane()).add(outerPane, "Center");
      

	    //Add listener to connectDisconnect button.  Since this button's text
	    //is updated, must 
	    connectDisconnect.addActionListener(new ActionListener(){
		    public void actionPerformed(ActionEvent evt){
                        connectDisconnect();}});
      
	    //Show before setting sliders.
	    //Components have to be realized on the screen before
	    //the sliders can be set.
	    this.show();
	    innerPane.setDividerLocation(0.5);
	    outerPane.setDividerLocation(1.0);
    
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
	if(!(userObject instanceof PlaceHolder))
	    tree.expandPath(path);
	if(selectedNode.isRoot()){
	    //We are at the root node.  Display some helpful information.
	    innerPane.setRightComponent(getPanelHelpMessage(1));
	    outerPane.setDividerLocation(0.5);
	}
	else if(userObject instanceof PlaceHolder){
	    PlaceHolder ph = (PlaceHolder) userObject; 
	    if(parentNode != null){
		switch(ph.getApplicationExperimentTrial()){
		case 0:
		    switch(ph.getStandardRuntimeDB()){
		    case 0:
			outerPane.setRightComponent(getApplicationPanel(0));
			innerPane.setRightComponent(getPanelHelpMessage(2));
			outerPane.setDividerLocation(0.8);
			innerPane.setDividerLocation(0.5);
			break;
		    case 1:
			outerPane.setRightComponent(getApplicationPanel(1));
			innerPane.setRightComponent(getPanelHelpMessage(3));
			outerPane.setDividerLocation(0.8);
			innerPane.setDividerLocation(0.5);
			break;
		    case 2:
			outerPane.setRightComponent(getApplicationPanel(2));
			innerPane.setRightComponent(getPanelHelpMessage(4));
			outerPane.setDividerLocation(0.67);
			innerPane.setDividerLocation(0.5);
			break;
		    default:
			break;
		    }
		    break;
		case 1:
		    switch(ph.getStandardRuntimeDB()){
		    case 0:
			outerPane.setRightComponent(getExperimentPanel(0));
			innerPane.setRightComponent(getPanelHelpMessage(5));
			outerPane.setDividerLocation(0.8);
			innerPane.setDividerLocation(0.5);
			break;
		    case 1:
			outerPane.setRightComponent(getExperimentPanel(1));
			innerPane.setRightComponent(getPanelHelpMessage(5));
			outerPane.setDividerLocation(0.8);
			innerPane.setDividerLocation(0.5);
			break;
		    case 2:
			outerPane.setRightComponent(getExperimentPanel(2));
			innerPane.setRightComponent(getPanelHelpMessage(5));
			outerPane.setDividerLocation(0.8);
			innerPane.setDividerLocation(0.5);
			break;
		    default:
			break;
		    }
		    break;
		case 2:
		    switch(ph.getStandardRuntimeDB()){
		    case 0:
			outerPane.setRightComponent(getTrialPanel(0));
			innerPane.setRightComponent(getPanelHelpMessage(6));
			outerPane.setDividerLocation(0.8);
			innerPane.setDividerLocation(0.5);
			break;
		    case 1:
			outerPane.setRightComponent(getTrialPanel(1));
			innerPane.setRightComponent(getPanelHelpMessage(6));
			outerPane.setDividerLocation(0.8);
			innerPane.setDividerLocation(0.5);
			break;
		    case 2:
			outerPane.setRightComponent(getTrialPanel(2));
			innerPane.setRightComponent(getPanelHelpMessage(6));
			outerPane.setDividerLocation(0.8);
			innerPane.setDividerLocation(0.5);
			break;
		    default:
			break;
		    }
		    break;
		default:
		    break;
		}
	    }
	}
	else if(userObject instanceof ParaProfApplication){
	    outerPane.setRightComponent(new JPanel());
	    innerPane.setRightComponent(getApplicationTable((ParaProfApplication) userObject));
	    outerPane.setDividerLocation(1.0);
	    innerPane.setDividerLocation(0.5);
	}
	else if(userObject instanceof ParaProfExperiment){
	    outerPane.setRightComponent(new JPanel());
	    innerPane.setRightComponent(getPanelHelpMessage(8));
	    outerPane.setDividerLocation(1.0);
	    innerPane.setDividerLocation(0.5);
	}
	else if(userObject instanceof ParaProfTrial){
	    String tmpString = userObject.toString();
	    //Here the actual clicked on node is an instance of ParaProfTrial (unlike the above
	    //check on ParaProfTrial where it was the parent node).
	    outerPane.setRightComponent(new JPanel());
	    innerPane.setRightComponent(getPanelHelpMessage(8));
	    outerPane.setDividerLocation(1.0);
	    innerPane.setDividerLocation(0.5);
	}
	else if(userObject instanceof Metric){
	    innerPane.setRightComponent(getPanelHelpMessage(7));
	    innerPane.setDividerLocation(0.5);
	    //Note that the parent user object should be an intance of trial.
	    //Check though!
	    Object tmpObject =  parentNode.getUserObject();
	    if(tmpObject instanceof ParaProfTrial)
		showMetric((ParaProfTrial) tmpObject, (Metric) userObject);
	    else
		ParaProf.systemError(null, null, "jRM02 - Logic Error");
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
  
    private Component getPanelHelpMessage(int type){
	JTextArea jTextArea = new JTextArea();
	jTextArea.setLineWrap(true);
	jTextArea.setWrapStyleWord(true);
	///Set the text.
	switch(type){
	case 0:
	    jTextArea.append("ParaProf Manager\n\n");
	    jTextArea.append("This window allows you to manage all of ParaProf's loaded data.\n");
	    jTextArea.append("Data can be static (ie, not updated at runtime),"
			     + " and loaded either remotely or locally.  You can also specify data to be uploaded at runtime.\n\n");
	    break;
	case 1:
	    jTextArea.append("ParaProf Manager\n\n");
	    jTextArea.append("There are three application types:\n\n");
	    jTextArea.append("Standard - The classic ParaProf mode.  At present, there is only a \"default\" set"
			     + " which is set for you automatically.  This maintains compatability with older versions where the"
			     + " application set was not definied.  This grouping will be expanded in time. For now, you may add"
			     + " experiments under the default set only.\n\n");
	    jTextArea.append("Runtime - A new feature allowing ParaProf to update data at runtime.  Please see"
			     + " the ParaProf documentation if the options are not clear."
			     + " As in the Standard set, there is only section.\n\n");
	    jTextArea.append("DB Apps - Another new feature allowing ParaProf to load data from a database.  Again, please see"
			     + " the ParaProf documentation if the options are not clear.\n\n");
	    
	    break;
	case 2:
	    jTextArea.append("ParaProf Manager\n\n");
	    jTextArea.append("This is the Standard application section:\n\n");
	    jTextArea.append("Standard - The classic ParaProf mode.  Data sets that are loaded at startup are placed"
			  + "under the default application automatically. Please see the ParaProf documentation for mre details.\n");
	    break;
	case 3:
	    jTextArea.append("ParaProf Manager\n\n");
	    jTextArea.append("This is the Runtime application section:\n\n");
	    jTextArea.append("Runtime - A new feature allowing ParaProf to update data at runtime.  Please see"
			  + " the ParaProf documentation if the options are not clear.\n");
	    break;
	case 4:
	    jTextArea.append("ParaProf Manager\n\n");
	    jTextArea.append("This is the DB Apps application section:\n\n");
	    jTextArea.append("DB Apps - Another new feature allowing ParaProf to load data from a database.  Again, please see"
			  + " the ParaProf documentation if the options are not clear.\n");
	    break;
	case 5:
	    jTextArea.append("ParaProf Manager\n\n");
	    jTextArea.append("Click the add experiment button below to add an experiment.\n");
	    break;
	case 6:
	    jTextArea.append("ParaProf Manager\n\n");
	    jTextArea.append("Click the add trial button below to add a trial.\n");
	    break;
	case 7:
	    jTextArea.append("ParaProf Manager\n\n");
	    jTextArea.append("Clicking on different metrics causes ParaProf to display the clicked on metric.\n\n");
	    jTextArea.append("The sub-window below allow you to generate new metrics based on those that were"
			     + " gathered during the run.  The operand number options for Operand A and B correspond the numbers prefixing the metrics.\n");
	    break;
	case 8:
	    //Don't set any text.
	    break;
	default:
	    break;
	}
	return (new JScrollPane(jTextArea));
    }

    private Component getApplicationTable(ParaProfApplication application){
	return (new JScrollPane(new JTable(new ParaProfManagerTableModel(application, treeModel))));}
    
    private Component getApplicationPanel(int type){
	JPanel jPanel = new JPanel();
	JButton jButton = new JButton("Add Application");
	jButton.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent evt){
		    addApplicationButton();}});
    
	//Now add the components to the panel.
	GridBagLayout gbl = new GridBagLayout();
	jPanel.setLayout(gbl);
	GridBagConstraints gbc = new GridBagConstraints();
	gbc.insets = new Insets(5, 5, 5, 5);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.WEST;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(jPanel, jButton, gbc, 0, 0, 2, 1);

	if(type == 2){
	    jButton = new JButton("Refresh Applications");
	    jButton.addActionListener(new ActionListener(){
		    public void actionPerformed(ActionEvent evt){
                        refreshApplications();}});
	    
	    gbc.fill = GridBagConstraints.NONE;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    panelAdd(jPanel, new JLabel("Username:"), gbc, 0, 1, 1, 1);
	    
	    gbc.fill = GridBagConstraints.NONE;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 100;
	    gbc.weighty = 0;
	    panelAdd(jPanel, usernameField, gbc, 1, 1, 1, 1);
	    
	    gbc.fill = GridBagConstraints.NONE;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    panelAdd(jPanel, new JLabel("Password:"), gbc, 0, 2, 1, 1);

	    gbc.fill = GridBagConstraints.NONE;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 100;
	    gbc.weighty = 0;
	    panelAdd(jPanel, passwordField, gbc, 1, 2, 1, 1);

	    gbc.fill = GridBagConstraints.NONE;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    panelAdd(jPanel, new JLabel("Config File:"), gbc, 0, 3, 1, 1);
	    
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 100;
	    gbc.weighty = 0;
	    panelAdd(jPanel, configFileField, gbc, 1, 3, 3, 1);
	    
	    gbc.fill = GridBagConstraints.NONE;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    panelAdd(jPanel, connectDisconnect, gbc, 0, 4, 2, 1);
	    
	    gbc.fill = GridBagConstraints.NONE;
	    gbc.anchor = GridBagConstraints.EAST;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    panelAdd(jPanel, jButton, gbc, 2, 4, 2, 1);
	}
	
	return jPanel;
    }
 
    private Component getExperimentPanel(int type){
	JPanel jPanel = new JPanel();
	JButton jButton = new JButton("Add Experiment");
	jButton.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent evt){
		    addExperiment();}});
    
	//Now add the components to the panel.
	GridBagLayout gbl = new GridBagLayout();
	jPanel.setLayout(gbl);
	GridBagConstraints gbc = new GridBagConstraints();
	gbc.insets = new Insets(5, 5, 5, 5);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(jPanel, jButton, gbc, 0, 0, 1, 1);

	if(type == 2){
	    jButton = new JButton("Refresh Experiments");
	    jButton.addActionListener(new ActionListener(){
		    public void actionPerformed(ActionEvent evt){
                        refreshExperiments();}});

	    
	    gbc.fill = GridBagConstraints.NONE;
	    gbc.anchor = GridBagConstraints.EAST;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    panelAdd(jPanel, jButton, gbc, 1, 0, 1, 1);
	}
	
	return jPanel;
    }

    private Component getTrialPanel(int type){
	JPanel jPanel = new JPanel();

	JButton jButton = new JButton("Add Trail");
	jButton.addActionListener(new ActionListener(){
                public void actionPerformed(ActionEvent evt){
		    addTrial();}
	    });

	String trialTypeStrings[] = {"Pprof -d File", "Tau Output", "Dynaprof", "other-2"};
	trialType = new JComboBox(trialTypeStrings);
	
	JLabel jLabel = new JLabel("Trial Type");//, SwingConstants.RIGHT);

	//Now add the components to the panel.
	GridBagLayout gbl = new GridBagLayout();
	jPanel.setLayout(gbl);
	GridBagConstraints gbc = new GridBagConstraints();
	gbc.insets = new Insets(5, 5, 5, 5);

	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(jPanel, jButton, gbc, 0, 0, 1, 1);
	
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(jPanel, jLabel, gbc, 1, 0, 1, 1);

	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.WEST;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(jPanel, trialType, gbc, 2, 0, 1, 1);

	if(type == 2){
	    jButton = new JButton("Refresh Trials");
	    jButton.addActionListener(new ActionListener(){
		    public void actionPerformed(ActionEvent evt){
                        refreshTrials();}});

	    
	    gbc.fill = GridBagConstraints.NONE;
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    panelAdd(jPanel, jButton, gbc, 1, 1, 1, 1);
	}

	return jPanel;
    }
    //******************************
    //End - Component functions.
    //******************************
    void addExperiment(){
	JOptionPane.showMessageDialog(this, "Only the default experiment allowed in this release!", "Warning!"
                                      ,JOptionPane.ERROR_MESSAGE);
	return;
    }

    //Adds a trial to the given experiment. If the given experiment is null it tries to determine if
    //an experiment is clicked on in the display, and uses that.
    //If prompt is set to true, the user is prompted for a trial name, otherwise, a default name is created.
    //Returns the added trial or null if no trial was added.
    public void addTrial(){
    	try{
	    ParaProfTrial trial = null;
	    String trialName = null;
	    boolean dataAdded = false;
	    ParaProfExperiment experiment;
	    String string1 = null;
	    String string2 = null;
	    String string3 = null;

	    //Get the selected trial placeholder, and then its parent experiment node.
	    TreePath path = tree.getSelectionPath();
	    DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	    DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
	    Object userObject = parentNode.getUserObject();
	    experiment = (ParaProfExperiment) userObject;
      
	    //Get the name of the new trial.
	    trialName = JOptionPane.showInputDialog(this, "Enter trial name, then make your selection.");
	    if((trialName == null) || "".equals(trialName)){
		JOptionPane.showMessageDialog(this, "You must enter a name!", "Error!"
					      ,JOptionPane.ERROR_MESSAGE);
		System.out.println("Error adding trial ... aborted.");
		return;
	    }

	    if(experiment.isTrialPresent(trialName)){
		System.out.println("Trial with name: " + trialName + "exists. Not adding!");
		return;
	    }

	    int type = -1;
	    String s = (String) trialType.getSelectedItem();
	    if(s.equals("Pprof -d File"))
		type = 0;
	    else if(s.equals("Tau Output"))
		type = 1;
	    else if(s.equals("Dynaprof"))
		type = 2;
	    else if(s.equals("other-2"))
		type = 3;

	    System.out.println("trial type: " + type);

	    //Create the trial.
	    trial = new ParaProfTrial(null, type);
	    trial.setName(trialName);
	    
	    FileList fl = new FileList();
	    Vector v = fl.getFileList(null, this, type,null,ParaProf.debugIsOn);

	    trial.initialize(v);
	    
	    experiment.addTrial(trial);
	    
	    DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(trial);
	    trial.setDMTN(trialNode);
	    
	    //Update the tree.
	    for(Enumeration e2 = (trial.getMetrics()).elements(); e2.hasMoreElements() ;){
		Metric metric = (Metric) e2.nextElement();
		DefaultMutableTreeNode metricNode = new DefaultMutableTreeNode(metric);
		metric.setDMTN(metricNode);
		metricNode.setAllowsChildren(false);
		trialNode.add(metricNode);
	    }
	    treeModel.insertNodeInto(trialNode, selectedNode, selectedNode.getChildCount());
	}
	catch(Exception e){
	    System.out.println("Error adding trial ... aborted.");
	    System.out.println("Location - ParaProfManager.addTrial(...)");
	    if(ParaProf.debugIsOn)
		e.printStackTrace();
	    return;
	}
    }

    public void connectDisconnect(){
	try{
	    //Try making a connection to the database.
	    if(connectDisconnect.getText().equals("Connect to Database")){
		String configFile = configFileField.getText().trim();
		String username = usernameField.getText().trim();
		String password = new String(passwordField.getPassword());
		if(password==null)
		    password = "";
		else
		    password.trim();
		perfDBSession = new PerfDBSession();
		perfDBSession.initialize(configFile, password);
		this.connectDisconnect.setText("Disconnect from Database");
	    }
	    else{
		perfDBSession.terminate();
		perfDBSession = null;
		connectDisconnect.setText("Connect to Database");
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "ELM03");
	}
    }

    public void refreshApplications(){  
	  try{
	      DefaultMutableTreeNode applicationNode = null;
	      DefaultMutableTreeNode experimentsPlaceHolder = null;

	      //Clear node.
	      for(int i=dbApps.getChildCount(); i>0; i--){
		  treeModel.removeNodeFromParent(((DefaultMutableTreeNode) dbApps.getChildAt(i-1)));
	      }
 	      ListIterator l = perfDBSession.getApplicationList();
	      while (l.hasNext()){
		  ParaProfApplication application = new ParaProfApplication((Application)l.next());
		  applicationNode = new DefaultMutableTreeNode(application);
		  application.setDBApplication(true);
		  application.setDMTN(applicationNode);
		  //Create and add the experiment place holder.
		  experimentsPlaceHolder = new DefaultMutableTreeNode(new PlaceHolder("Experiments", 1, 2));
		  applicationNode.add(experimentsPlaceHolder);
		  //Now insert the application into the node into the tree.
		  treeModel.insertNodeInto(applicationNode, dbApps, dbApps.getChildCount());
	      }
	      return;
	  }
	  catch(Exception e){
	      ParaProf.systemError(e, null, "ELM03");
	  }
    }

    public void refreshExperiments(){
	try{
	    DefaultMutableTreeNode experimentNode = null;
	    DefaultMutableTreeNode trialsPlaceHolder = null;

	    TreePath path = tree.getSelectionPath();
	    DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	    DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
	    for(int i=selectedNode.getChildCount(); i>0; i--){
		treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i-1)));
	    }

	    ParaProfApplication application = (ParaProfApplication) parentNode.getUserObject();
	    
	    perfDBSession.setApplication(application.getID());
	    ListIterator l = perfDBSession.getExperimentList();
	    while (l.hasNext()){
		ParaProfExperiment experiment = new ParaProfExperiment((Experiment)l.next());
		experiment.setApplication(application);
		experimentNode = new DefaultMutableTreeNode(experiment);
		experiment.setDMTN(experimentNode);
		//Create and add the trials place holder.
		trialsPlaceHolder = new DefaultMutableTreeNode(new PlaceHolder("Trials", 2, 2));
		experimentNode.add(trialsPlaceHolder);
		//Now insert the experimentinto the node into the tree.
		treeModel.insertNodeInto(experimentNode, selectedNode, selectedNode.getChildCount());
	    }
	    return;
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "ELM03");
	}
    }

    public void refreshTrials(){
	try{
	    DefaultMutableTreeNode trialNode = null;
	    DefaultMutableTreeNode metricNode = null;

	    TreePath path = tree.getSelectionPath();
	    DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	    DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();
	    for(int i=selectedNode.getChildCount(); i>0; i--){
		treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i-1)));
	    }

	    ParaProfExperiment experiment = (ParaProfExperiment) parentNode.getUserObject();
	    ParaProfApplication application = experiment.getApplication();
	    
	    perfDBSession.setApplication(application.getID());
	    perfDBSession.setExperiment(experiment.getID());
	    ListIterator l = perfDBSession.getTrialList();
	    while (l.hasNext()){
		ParaProfTrial trial = new ParaProfTrial((Trial)l.next(), 4);
		trial.setExperiment(experiment);
		trialNode = new DefaultMutableTreeNode(trial);
		trial.setDMTN(trialNode);
		//Populate the metrics list for this trial.
		for(Enumeration e = (trial.getMetrics()).elements(); e.hasMoreElements() ;){
		    Metric metric = (Metric) e.nextElement();
		    metricNode = new DefaultMutableTreeNode(metric);
		    metric.setDMTN(metricNode);
		    metric.setDBMetric(true);
		    metricNode.setAllowsChildren(false);
		    trialNode.add(metricNode);
		}
		treeModel.insertNodeInto(trialNode, selectedNode, selectedNode.getChildCount());
	    }
	    return;
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "ELM03");
	}
    }
  
    void applyOperation(){
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
	catch(Exception e){
	    ParaProf.systemError(e, null, "ELM03");
	}
    }
  
    void showMetric(ParaProfTrial trial, Metric metric){
	try{
	    if(metric.getDBMetric()){
		perfDBSession.setApplication(trial.getExperiment().getApplication().getID());
		perfDBSession.setExperiment(trial.getExperiment().getID());
		perfDBSession.setTrial(trial.getID());
		ListIterator l = perfDBSession.getFunctionData();
		int metricID = metric.getID();
		while(l.hasNext()){
		    FunctionDataObject f = (FunctionDataObject) l.next();
		    System.out.println("###############################");
		    System.out.println("getNode: "+f.getNode());
		    System.out.println("getContext: "+f.getContext());
		    System.out.println("getThread: "+f.getThread());
		    System.out.println("getFunctionIndexID: "+f.getFunctionIndexID());
		    System.out.println("getInclusivePercentage: "+f.getInclusivePercentage(metricID));
		    System.out.println("getInclusive: "+f.getInclusive(metricID));
		    System.out.println("getExclusivePercentage: "+f.getExclusivePercentage(metricID));
		    System.out.println("getExclusive: "+f.getExclusive(metricID));
		    System.out.println("getInclusivePerCall: "+f.getInclusivePerCall(metricID));
		    System.out.println("getNumCalls: "+f.getNumCalls());
		    System.out.println("getNumSubroutines: "+f.getNumSubroutines());
		    System.out.println("###############################");
		}
	    }
	    else{
		trial.setCurValLoc(metric.getID());
	    trial.getSystemEvents().updateRegisteredObjects("dataEvent");
	    trial.showMainWindow();
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "jRM04");
	}
    }
  
    void populateStandardApplications(DefaultMutableTreeNode inNode){
    
	DefaultMutableTreeNode applicationNode = null;
	DefaultMutableTreeNode experimentsPlaceHolder = null;
	DefaultMutableTreeNode experimentNode = null;
	DefaultMutableTreeNode trialsPlaceHolder = null;
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
      
	    experimentsPlaceHolder = new DefaultMutableTreeNode(new PlaceHolder("Experiments", 1, 0));
	    for(Enumeration e2 = (application.getExperiments()).elements(); e2.hasMoreElements() ;){  
		ParaProfExperiment exp = (ParaProfExperiment) e2.nextElement();
		experimentNode = new DefaultMutableTreeNode(exp);
		exp.setDMTN(experimentNode);
        
		//Populate the trials for this experiemnt.
		trialsPlaceHolder = new DefaultMutableTreeNode(new PlaceHolder("Trials", 2, 0));
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
		    trialsPlaceHolder.add(trialNode);
		    if(cnt3 == 0)
			defaultParaProfTrialNode = trialNode;
		    cnt3++;
		}
		experimentNode.add(trialsPlaceHolder);
		experimentsPlaceHolder.add(experimentNode);
		cnt2++;
	    }
	    applicationNode.add(experimentsPlaceHolder);
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
    public void addApplicationButton(){
	TreePath path = tree.getSelectionPath();
	if(path == null)
	    return;
	DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	Object userObject = selectedNode.getUserObject();
    
	if(userObject instanceof PlaceHolder){
	    ParaProfApplication application = null;
	    DefaultMutableTreeNode treeNode = null;
	    PlaceHolder pl = (PlaceHolder) userObject;
	    String name = pl.getName();
	    //Just for sanity, check that we got an application place holder.
	    if(pl.getApplicationExperimentTrial()==0){
		switch(pl.getStandardRuntimeDB()){
		case 0:
		    application = ParaProf.applicationManager.addApplication();
		    treeNode = new DefaultMutableTreeNode(application);
		    application.setDMTN(treeNode);
		    treeModel.insertNodeInto(treeNode, selectedNode, selectedNode.getChildCount());
		    break;
		case 1:
		    application = ParaProf.applicationManager.addApplication();
		    treeNode = new DefaultMutableTreeNode(application);
		    application.setDMTN(treeNode);
		    treeModel.insertNodeInto(treeNode, selectedNode, selectedNode.getChildCount());
		    break;
		case 2:
		    break;
		default:
		    break;
		}
	    }
	}
    }
  
    public void updateParaProfApplicationButton(){
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
		    treeModel.insertNodeInto(new DefaultMutableTreeNode(ParaProf.applicationManager.addApplication()),
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
	    if(EventSrc instanceof JMenuItem){
		    String arg = evt.getActionCommand();
		    if(arg.equals("Exit ParaProf!")){
			setVisible(false);
			dispose();
			System.exit(0);
		    } 
		    else if(arg.equals("ParaProf Manager")){
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
		    else if(arg.equals("About ParaProf")){
			JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
		    }
		    else if(arg.equals("Show Help Window")){
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
	catch(Exception e){
	    ParaProf.systemError(e, null, "ELM05");
	}
    }
  
    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e){
	closeThisWindow();
    }
    
    void closeThisWindow(){ 
	try{
	    if(ParaProf.debugIsOn){
		System.out.println("------------------------");
		System.out.println("ParaProfExperiment List Manager Window is closing!");
		System.out.println("Clearing resourses for this window.");
	    }
	    setVisible(false);
	    dispose();
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "ELM06");
	}
    }
  
    private void panelAdd(JPanel inJPanel, Component c, GridBagConstraints gbc, int x, int y, int w, int h){
	gbc.gridx = x;
	gbc.gridy = y;
	gbc.gridwidth = w;
	gbc.gridheight = h;
	inJPanel.add(c, gbc);
    }
  
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h){
	gbc.gridx = x;
	gbc.gridy = y;
	gbc.gridwidth = w;
	gbc.gridheight = h;
    	getContentPane().add(c, gbc);
    }
  
    //####################################
    //Instance Data.
    //####################################
    JTree tree = null;
    DefaultTreeModel treeModel = null;
    DefaultMutableTreeNode dbApps = null;
    JSplitPane innerPane = null;
    JSplitPane outerPane = null;
  
    //A reference to the default trial node.
    DefaultMutableTreeNode defaultParaProfTrialNode = null;

    private PerfDBSession perfDBSession = null;
    //Must keep a reference as button's text is updated.
    private JButton connectDisconnect = new JButton("Connect to Database");
    private JTextField configFileField = new JTextField(System.getProperty("user.dir"), 30);
    private JTextField usernameField = new JTextField("username", 20);
    private JPasswordField passwordField = new JPasswordField("password",20);

    private JComboBox trialType = null;
  
    private JComboBox operation = null;
    private JTextField opA = null;
    private JTextField opB = null;
  
    private Border loweredbev = null;
    private Border raisedbev = null;
    private Border empty = null;
    //####################################
    //End - Instance Data.
    //####################################
}

class ParaProfTreeCellRenderer extends DefaultTreeCellRenderer{
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
	if(userObject instanceof PlaceHolder){
	    PlaceHolder ph = (PlaceHolder) userObject; 
	    URL url = null;
	    switch(ph.getApplicationExperimentTrial()){
	    case 0:
		url = ParaProfTreeCellRenderer.class.getResource("red-ball.gif");
		this.setIcon(new ImageIcon(url));
		break;
	    case 1:
		url = ParaProfTreeCellRenderer.class.getResource("blue-ball.gif");
		this.setIcon(new ImageIcon(url));
		break;
	    case 2:
		url = ParaProfTreeCellRenderer.class.getResource("green-ball.gif");
		this.setIcon(new ImageIcon(url));
		break;
	    default:
		break;
	    }
	}
	else if(userObject instanceof Metric){
	    this.setIcon(new ImageIcon("green-ball.gif"));
	}
	return this;
    }
}

//######
//Acts as a simple place holder for applications, experiments, and trials.
//Chose to make it a class to make it easier to determine the click location
//in the tree.  Checking on strings runs the risk of name conflicts if
//one does not check the parent node (which is a pain).
//######
class PlaceHolder{
    public PlaceHolder(String name, int applicationExperimentTrial, int standardRuntimeDB){
	this.name = name;
	this.applicationExperimentTrial = applicationExperimentTrial;
	this.standardRuntimeDB = standardRuntimeDB;
    }
  
    public String getName(){
	return name;}

    public int getApplicationExperimentTrial(){
	return applicationExperimentTrial;}

    public int getStandardRuntimeDB(){
	return standardRuntimeDB;}
    
    public String toString(){
	return name;}
  
    String name = "Name Not Set";
    int applicationExperimentTrial = -1; //0-Application,1-Experiment,2-Trial.
    int standardRuntimeDB = -1; //0-Standard,1-Runtime,2-DB.
}
