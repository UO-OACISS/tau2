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

public class ParaProfManager extends JFrame implements ActionListener, TreeSelectionListener, TreeExpansionListener{
    public ParaProfManager(){
	
	try{
	    //####################################
	    //Window Stuff.
	    //####################################
	    int windowWidth = 800;
	    int windowHeight = 500;
	    
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
	    if(xPosition>50)
		xPosition = xPosition-50;
	    if(yPosition>50)
		yPosition = yPosition-50;
	    
	    this.setLocation(xPosition, yPosition);
	    setSize(new java.awt.Dimension(windowWidth, windowHeight));
	    setTitle("ParaProf Manager");
	    
	    //Add some window listener code
	    addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(java.awt.event.WindowEvent evt) {
			thisWindowClosing(evt);
		    }
		});
	    //####################################
	    //End - Window Stuff.
	    //####################################
      
	    //####################################
	    //Code to generate the menus.
	    //####################################
	    JMenuBar mainMenu = new JMenuBar();
      
	    //######
	    //File menu.
	    //######
	    JMenu fileMenu = new JMenu("File");

	    //Add a menu item.
	    JMenuItem dbItem = new JMenuItem("Database Configuration");
	    dbItem.addActionListener(this);
	    fileMenu.add(dbItem);
      
	    //Add a menu item.
	    JMenuItem closeItem = new JMenuItem("Close This Window");
	    closeItem.addActionListener(this);
	    fileMenu.add(closeItem);
      
	    //Add a menu item.
	    JMenuItem exitItem = new JMenuItem("Exit ParaProf!");
	    exitItem.addActionListener(this);
	    fileMenu.add(exitItem);
	    //######
	    //End - File menu.
	    //######

	    //######
	    //Options menu.
	    //######
	    JMenu optionsMenu = new JMenu("Options");
	    
	    JMenuItem applyOperationItem = new JMenuItem("Apply Operation");
	    applyOperationItem.addActionListener(this);
	    optionsMenu.add(applyOperationItem);
	    //######
	    //End - Options menu.
	    //######

	    //######
	    //Help menu.
	    //######
	    JMenu helpMenu = new JMenu("Help");
      
	    //Add a menu item.
	    JMenuItem aboutItem = new JMenuItem("About ParaProf");
	    aboutItem.addActionListener(this);
	    helpMenu.add(aboutItem);
      
	    //Add a menu item.
	    JMenuItem showHelpWindowItem = new JMenuItem("Show Help Window");
	    showHelpWindowItem.addActionListener(this);
	    helpMenu.add(showHelpWindowItem);
	    //######
	    //End - Help menu.
	    //######
       
	    //Now, add all the menus to the main menu.
	    mainMenu.add(fileMenu);
	    mainMenu.add(optionsMenu);
	    mainMenu.add(helpMenu);
	    setJMenuBar(mainMenu);
	    //####################################
	    //End - Code to generate the menus.
	    //####################################
   
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
	    tree.setRootVisible(false);
	    tree.getSelectionModel().setSelectionMode(TreeSelectionModel.SINGLE_TREE_SELECTION);
	    ParaProfTreeCellRenderer renderer = new ParaProfTreeCellRenderer();
	    tree.setCellRenderer(renderer);
      
	    //######
	    //Add tree listeners.
	    tree.addTreeSelectionListener(this);
	    tree.addTreeExpansionListener(this);
	    //######
	          
	    //Bung it in a scroll pane.
	    JScrollPane treeScrollPane = new JScrollPane(tree);
	    //####################################
	    //End - Create the tree.
	    //####################################


	    //####################################
	    //Set up the split pane, and add to content pane.
	    //####################################
	    jSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, treeScrollPane, getPanelHelpMessage(0));
	    jSplitPane.setContinuousLayout(true);
	    (getContentPane()).add(jSplitPane, "Center");
      
	    //Show before setting dividers.
	    //Components have to be realized on the screen before
	    //the dividers can be set.
	    this.show();
	    jSplitPane.setDividerLocation(0.5);
	    //####################################
	    //End - Set up the split pane, and add to content pane.
	    //####################################
	}
	catch(Exception e){
	    e.printStackTrace();
	    UtilFncs.systemError(e, null, "PPM01");
	}
    }

    //####################################
    //Interface code.
    //####################################
    
    //######
    //ActionListener.
    //######
    public void actionPerformed(ActionEvent evt){
	try{
	    Object EventSrc = evt.getSource();
	    if(EventSrc instanceof JMenuItem){
		    String arg = evt.getActionCommand();
		    if(arg.equals("Exit ParaProf!")){
			setVisible(false);
			dispose();
			System.exit(0);
		    } 
		    else if(arg.equals("Close This Window")){
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
		    else if(arg.equals("Database Configuration")){
			(new DBConfiguration(this)).show();}
		    else if(arg.equals("Apply Operation")){
			(new PPMLWindow(this)).show();}
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
	    UtilFncs.systemError(e, null, "ELM05");
	}
    }
    //######
    //End - ActionListener.
    //######

    //######
    //TreeSelectionListener
    //######
    public void valueChanged(TreeSelectionEvent event){
	TreePath path = tree.getSelectionPath();
	if(path == null)
	    return;
	DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) selectedNode.getParent();            
	Object userObject = selectedNode.getUserObject();

	if((parentNode.isRoot())){
	    if(userObject.toString().equals("Standard Applications")){
		jSplitPane.setRightComponent(getPanelHelpMessage(1));
		jSplitPane.setDividerLocation(0.5);
		//Refresh the application list.
		System.out.println("Loading application list ...");
		for(int i=standard.getChildCount(); i>0; i--){
		    treeModel.removeNodeFromParent(((DefaultMutableTreeNode) standard.getChildAt(i-1)));
		}
		ListIterator l = ParaProf.applicationManager.getApplicationList();
		while (l.hasNext()){
		    ParaProfApplication application = (ParaProfApplication)l.next();
		    DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
		    application.setDMTN(applicationNode);
		    treeModel.insertNodeInto(applicationNode, standard, standard.getChildCount());
		}
		System.out.println("Done loading application list.");
		tree.expandPath(path);
		return;
	    }
	    else if(userObject.toString().equals("Runtime Applications")){
		jSplitPane.setRightComponent(getPanelHelpMessage(2));
		jSplitPane.setDividerLocation(0.5);
	    }
	    else if(userObject.toString().equals("DB Applications")){
		try{
		    jSplitPane.setRightComponent(getPanelHelpMessage(3));
		    jSplitPane.setDividerLocation(0.5);

		    if(configFile==null||password==null){//Check to see if the user has set configuration information.
			JOptionPane.showMessageDialog(this, "Please set the database configuration information (file menu).",
						      "DB Configuration Error!",
						      JOptionPane.ERROR_MESSAGE);
			return;
		    }
		    else{//Test to see if configurataion file exists.
			File file = new File(configFile);
			if(!file.exists()){
			    JOptionPane.showMessageDialog(this, "Specified configuration file does not exist.",
							  "DB Configuration Error!",
							  JOptionPane.ERROR_MESSAGE);
			    return;
			}
		    }
		    //Basic checks done, try to access the db.
		    //Refresh the application list.
		    System.out.println("Loading application list ...");
		    for(int i=dbApps.getChildCount(); i>0; i--){
			treeModel.removeNodeFromParent(((DefaultMutableTreeNode) dbApps.getChildAt(i-1)));
		    }
		    PerfDBSession perfDBSession = new PerfDBSession(); 
		    perfDBSession.initialize(configFile, password);
		    ListIterator l = perfDBSession.getApplicationList();
		    while (l.hasNext()){
			ParaProfApplication application = new ParaProfApplication((Application)l.next());
			application.setDBApplication(true);
			DefaultMutableTreeNode applicationNode = new DefaultMutableTreeNode(application);
			application.setDMTN(applicationNode);
			treeModel.insertNodeInto(applicationNode, dbApps, dbApps.getChildCount());
		    }
		    perfDBSession.terminate();
		    System.out.println("Done loading application list.");
		    tree.expandPath(path);
		    return;
		}
		catch(Exception e){
		    e.printStackTrace();
		}
	    }
	}
	else if(userObject instanceof ParaProfApplication){
	    try{
		ParaProfApplication application = (ParaProfApplication) userObject;
		if(application.dBApplication()){
		    //Refresh the experiments list.
		    System.out.println("Loading experiment list ...");
		    for(int i=selectedNode.getChildCount(); i>0; i--){
			treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i-1)));
		    }
		    PerfDBSession perfDBSession = new PerfDBSession(); 
		    perfDBSession.initialize(configFile, password);
		    //Set the application.
		    perfDBSession.setApplication(application.getID());
		    ListIterator l = perfDBSession.getExperimentList();
		    while (l.hasNext()){
			ParaProfExperiment experiment = new ParaProfExperiment((Experiment)l.next());
			experiment.setDBExperiment(true);
			experiment.setApplication(application);
			DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(experiment);
			experiment.setDMTN(experimentNode);
			treeModel.insertNodeInto(experimentNode, selectedNode, selectedNode.getChildCount());
		    }
		    perfDBSession.terminate();
		    System.out.println("Done loading experiment list.");
		}
		else{
		    System.out.println("Loading experiment list ...");
		    for(int i=selectedNode.getChildCount(); i>0; i--){
			treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i-1)));
		    }
		    ListIterator l = application.getExperimentList();
		    while (l.hasNext()){
			ParaProfExperiment experiment = (ParaProfExperiment)l.next();
			DefaultMutableTreeNode experimentNode = new DefaultMutableTreeNode(experiment);
			experiment.setDMTN(experimentNode);
			treeModel.insertNodeInto(experimentNode, selectedNode, selectedNode.getChildCount());
		    }
		    System.out.println("Done loading experiment list.");
		}
		tree.expandPath(path);
		jSplitPane.setRightComponent(getTable(userObject));
		jSplitPane.setDividerLocation(0.5);
	    }
	    catch(Exception e){
		e.printStackTrace();
	    }
	}
	else if(userObject instanceof ParaProfExperiment){
	    ParaProfExperiment experiment = (ParaProfExperiment) userObject;
	    if(experiment.dBExperiment()){
		//Refresh the trials list.
		System.out.println("Loading trial list ...");
		for(int i=selectedNode.getChildCount(); i>0; i--){
		    treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i-1)));
		}
		//Set the application and experiment.
		PerfDBSession perfDBSession = new PerfDBSession(); 
		perfDBSession.initialize(configFile, password);
		perfDBSession.setApplication(experiment.getApplicationID());
		perfDBSession.setExperiment(experiment.getID());
		ListIterator l = perfDBSession.getTrialList();
		while (l.hasNext()){
		    ParaProfTrial trial = new ParaProfTrial((Trial)l.next(), 4);
		    trial.setExperiment(experiment);
		    trial.setDBTrial(true);
		    DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(trial);
		    trial.setDMTN(trialNode);
		    treeModel.insertNodeInto(trialNode, selectedNode, selectedNode.getChildCount());
		}
		perfDBSession.terminate();
		System.out.println("Done loading trial list.");
	    }
	    else{
		System.out.println("Loading trial list ...");
		for(int i=selectedNode.getChildCount(); i>0; i--){
		    treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i-1)));
		}
		ListIterator l = experiment.getTrialList();
		while (l.hasNext()){
		    ParaProfTrial trial = (ParaProfTrial)l.next();
		    DefaultMutableTreeNode trialNode = new DefaultMutableTreeNode(trial);
		    trial.setDMTN(trialNode);
		    treeModel.insertNodeInto(trialNode, selectedNode, selectedNode.getChildCount());
		}
		System.out.println("Done loading trial list.");
	    }

	    tree.expandPath(path);
	    jSplitPane.setRightComponent(getTable(userObject));
	    jSplitPane.setDividerLocation(0.5);
	}
	else if(userObject instanceof ParaProfTrial){
	    ParaProfTrial trial = (ParaProfTrial) userObject;
	    if(trial.dBTrial()){
		//Test to see if trial has been loaded already.
		boolean loadedExists = false;
		for(Enumeration e = loadedTrials.elements(); e.hasMoreElements() ;){
		    ParaProfTrial loadedTrial = (ParaProfTrial) e.nextElement();
		    if((trial.getID()==loadedTrial.getID())&&
		       (trial.getExperimentID()==loadedTrial.getExperimentID())
		       &&(trial.getApplicationID()==loadedTrial.getApplicationID())){
			selectedNode.setUserObject(loadedTrial);
			trial = loadedTrial;
			loadedExists = true;
		    }
		}
		if(!loadedExists){
		    //Need to load the trial in from the db.
		    System.out.println("Loading trial ...");
		    PerfDBSession perfDBSession = new PerfDBSession(); 
		    perfDBSession.initialize(configFile, password);
		    perfDBSession.setApplication(trial.getApplicationID());
		    perfDBSession.setExperiment(trial.getExperimentID());
		    perfDBSession.setTrial(trial.getID());
		    trial.initialize(perfDBSession);
		    //Add to the list of loaded trials.
		    loadedTrials.add(trial);
		}
	    }

	    //At this point, in both the db and non-db cases, the trial
	    //is either loading or not.  Check this before displaying.
	    if(!trial.loading()){
		System.out.println("Loading metric list ...");
		for(int i=selectedNode.getChildCount(); i>0; i--){
		    treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i-1)));
		}
		ListIterator l = trial.getMetricList();
		while (l.hasNext()){
		    Metric metric = (Metric)l.next();
		    DefaultMutableTreeNode metricNode = new DefaultMutableTreeNode(metric);
		    metric.setDMTN(metricNode);
		    treeModel.insertNodeInto(metricNode, selectedNode, selectedNode.getChildCount());
		}
		System.out.println("Done loading metric list.");

		tree.expandPath(path);
		jSplitPane.setRightComponent(getTable(userObject));
		jSplitPane.setDividerLocation(0.5);
	    }
	    else{
		tree.expandPath(path);
		jSplitPane.setRightComponent(new JScrollPane(this.getLoadingTrialPanel(userObject)));
		jSplitPane.setDividerLocation(0.5);
	    }	    
	}
	else if(userObject instanceof Metric){
	    ParaProfTrial paraProfTrial  =  (ParaProfTrial) parentNode.getUserObject();
	    Metric metric = (Metric) userObject;
	    jSplitPane.setRightComponent(getTable(userObject));
	    jSplitPane.setDividerLocation(0.5);
	    this.showMetric(paraProfTrial, metric);
	}
    }
    //######
    //End - TreeSelectionListener
    //######

    //######
    //TreeExpansionListener
    //######
    public void treeCollapsed(TreeExpansionEvent event){
	System.out.println("Tree collapsed");
    }
    public void treeExpanded(TreeExpansionEvent event){}
    //######
    //End - TreeSelectionListener
    //######

    //####################################
    //End - Interface code.
    //####################################

    public void addMetricTreeNodes(ParaProfTrial trial){
	//Refresh the metrics list.
	DefaultMutableTreeNode defaultMutableTreeNode = trial.getDMTN();
	for(int i=defaultMutableTreeNode.getChildCount(); i>0; i--){
	    treeModel.removeNodeFromParent(((DefaultMutableTreeNode) defaultMutableTreeNode.getChildAt(i-1)));
	}
	
	for(Enumeration e = (trial.getMetrics()).elements(); e.hasMoreElements() ;){
	    Metric metric = (Metric) e.nextElement();
	    metric.setDBMetric(true);
	    DefaultMutableTreeNode metricNode = new DefaultMutableTreeNode(metric);
	    metric.setDMTN(metricNode);
	    metricNode.setAllowsChildren(false);
	    treeModel.insertNodeInto(metricNode, defaultMutableTreeNode, defaultMutableTreeNode.getChildCount());
	}

	jSplitPane.setRightComponent(getTable(trial));
	jSplitPane.setDividerLocation(0.5);
    }

    public void setDatabasePassword(String password){
	this.password = password;}

    public String getDatabasePassword(){
	return password;}

    public void setDatabaseConfigurationFile(String configFile){
	this.configFile = configFile;}

    public String getDatabaseConfigurationFile(){
	return configFile;}

    //####################################
    //Component functions.
    //####################################
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

    private Component getTable(Object obj){
	return (new JScrollPane(new JTable(new ParaProfManagerTableModel(obj, treeModel))));}

    private Component getLoadingTrialPanel(Object obj){
	JPanel jPanel = new JPanel();
	GridBagLayout gbl = new GridBagLayout();
	jPanel.setLayout(gbl);
	GridBagConstraints gbc = new GridBagConstraints();
	gbc.insets = new Insets(0, 0, 0, 0);

	JLabel jLabel = new JLabel("Trial loading ... Please wait!");

 	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.NORTH;
	gbc.weightx = 0;
	gbc.weighty = 0;
	gbc.gridx = 0;
	gbc.gridy = 0;
	gbc.gridwidth = 1;
	gbc.gridheight = 1;
	jPanel.add(this.getTable(obj),gbc);

	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	gbc.gridx = 0;
	gbc.gridy = 1;
	gbc.gridwidth = 1;
	gbc.gridheight = 1;
	jPanel.add(jLabel,gbc);

	return jPanel;
    }

    //####################################
    //End - Component functions.
    //####################################

    private void addExperiment(){
	JOptionPane.showMessageDialog(this, "Only the default experiment allowed in this release!", "Warning!"
                                      ,JOptionPane.ERROR_MESSAGE);
	return;
    }

    //Adds a trial to the given experiment. If the given experiment is null it tries to determine if
    //an experiment is clicked on in the display, and uses that.
    //If prompt is set to true, the user is prompted for a trial name, otherwise, a default name is created.
    //Returns the added trial or null if no trial was added.
    private void addTrial(){
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
	    String s = (String) "";
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
	    Vector v = fl.getFileList(null, this, type,null,UtilFncs.debug);

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
	    if(UtilFncs.debug)
		e.printStackTrace();
	    return;
	}
    }

    public void insertMetric(Metric metric){
	DefaultMutableTreeNode metricNode = new DefaultMutableTreeNode(metric);
	metric.setDMTN(metricNode);
	metricNode.setAllowsChildren(false);

	ParaProfTrial trial = metric.getTrial();
	DefaultMutableTreeNode trialNode = trial.getDMTN();

	treeModel.insertNodeInto(metricNode, trialNode, trialNode.getChildCount());
    }
  
    private void showMetric(ParaProfTrial trial, Metric metric){
	try{
	    trial.setSelectedMetricID(metric.getID());
	    trial.getSystemEvents().updateRegisteredObjects("dataEvent");
	    trial.showMainWindow();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "jRM04");
	}
    }
  
    public void expandDefaultParaProfTrialNode(){
	if(defaultParaProfTrialNode != null)
	    tree.expandPath(new TreePath(defaultParaProfTrialNode.getPath()));
    }
    
    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e){
	closeThisWindow();
    }
    
    void closeThisWindow(){ 
	try{
	    if(UtilFncs.debug){
		System.out.println("------------------------");
		System.out.println("ParaProfExperiment List Manager Window is closing!");
		System.out.println("Clearing resourses for this window.");
	    }
	    setVisible(false);
	    dispose();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "ELM06");
	}
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
    DefaultMutableTreeNode standard = null;
    DefaultMutableTreeNode runtime = null;
    DefaultMutableTreeNode dbApps = null;
    JSplitPane jSplitPane = null;
  
    //A reference to the default trial node.
    DefaultMutableTreeNode defaultParaProfTrialNode = null;

    private String password = null;
    private String configFile = null;//"/home/bertie/Programming/data/perfdb.cfg";
    private Vector loadedTrials = new Vector();
    //####################################
    //End - Instance Data.
    //####################################
}
