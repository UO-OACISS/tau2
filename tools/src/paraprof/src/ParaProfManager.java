
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
    
	    //******************************
	    //Create the tree.
	    //******************************
	    //Create the root node.
	    DefaultMutableTreeNode root = new DefaultMutableTreeNode("Applications");
  	    DefaultMutableTreeNode standard = new DefaultMutableTreeNode("Standard Applications");
	    DefaultMutableTreeNode runtime = new DefaultMutableTreeNode("Runtime Applications");
	    dbApps = new DefaultMutableTreeNode("DB Applications");

	    //Populate this node.
	    this.populateStandardApplications(standard);
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
	    //Set up the split pane.
	    jSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, treeScrollPane, getPanelHelpMessage(0));
	    jSplitPane.setContinuousLayout(true);
	    
	    (getContentPane()).add(jSplitPane, "Center");
      
	    //Show before setting dividers.
	    //Components have to be realized on the screen before
	    //the dividers can be set.
	    this.show();
	    jSplitPane.setDividerLocation(0.5);
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

	if((parentNode.isRoot())&&(userObject.toString().equals("DB Applications"))){
	    try{
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
		ParaProf.systemError(e, null, "ELM03");
	    }
	}
	else if(userObject instanceof ParaProfApplication){
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
	    tree.expandPath(path);
	    jSplitPane.setRightComponent(getApplicationTable((ParaProfApplication) userObject));
	    jSplitPane.setDividerLocation(0.5);
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
	    tree.expandPath(path);
	    jSplitPane.setRightComponent(getPanelHelpMessage(-1));
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
		    perfDBSession.terminate();
		    //Add to the list of loaded trials.
		    loadedTrials.add(trial);
		    System.out.println("Done loading trial.");
		}
		
		//The trial should now have been setup.  Therefore, safe to add metrics.
		//Refresh the metrics list.
		for(int i=selectedNode.getChildCount(); i>0; i--){
		    treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i-1)));
		}
		
		for(Enumeration e = (trial.getMetrics()).elements(); e.hasMoreElements() ;){
			Metric metric = (Metric) e.nextElement();
			metric.setDBMetric(true);
			DefaultMutableTreeNode metricNode = new DefaultMutableTreeNode(metric);
			metric.setDMTN(metricNode);
			metricNode.setAllowsChildren(false);
			treeModel.insertNodeInto(metricNode, selectedNode, selectedNode.getChildCount());
		}
	    }
	    tree.expandPath(path);
	    jSplitPane.setRightComponent(getPanelHelpMessage(-1));
	    jSplitPane.setDividerLocation(0.5);
	}
	else if(userObject instanceof Metric){
	    jSplitPane.setRightComponent(getPanelHelpMessage(-1));
	    jSplitPane.setDividerLocation(0.5);
	    //Note that the parent user object should be an intance of trial.
	    //Check though!
	    Object tmpObject =  parentNode.getUserObject();
	    if(tmpObject instanceof ParaProfTrial)
		showMetric((ParaProfTrial) tmpObject, (Metric) userObject);
	    else
		ParaProf.systemError(null, null, "jRM02 - Logic Error");
	}
    }
  
    private void treeExpansionEventHandler(TreeExpansionEvent event){}
    private void treeCollapsedEventHandler(TreeExpansionEvent event){}
  
    //******************************
    //End - Tree event handlers.
    //******************************
  
    //******************************
    //Component functions.
    //******************************
  
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

    private Component getApplicationTable(ParaProfApplication application){
	return (new JScrollPane(new JTable(new ParaProfManagerTableModel(application, treeModel))));}
    
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
  
    void showMetric(ParaProfTrial trial, Metric metric){
	try{
	    trial.setCurValLoc(metric.getID());
	    trial.getSystemEvents().updateRegisteredObjects("dataEvent");
	    trial.showMainWindow();
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "jRM04");
	}
    }
  
    void populateStandardApplications(DefaultMutableTreeNode inNode){
	DefaultMutableTreeNode applicationNode = null;
	DefaultMutableTreeNode experimentNode = null;
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
	    for(Enumeration e2 = (application.getExperiments()).elements(); e2.hasMoreElements() ;){  
		ParaProfExperiment experiment = (ParaProfExperiment) e2.nextElement();
		experimentNode = new DefaultMutableTreeNode(experiment);
		experiment.setDMTN(experimentNode);
 		//Populate the trials for this experiemnt.
		for(Enumeration e3 = (experiment.getTrials()).elements(); e3.hasMoreElements() ;){
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
		    experimentNode.add(trialNode);
		    if(cnt3 == 0)
			defaultParaProfTrialNode = trialNode;
		    cnt3++;
		}
		cnt2++;
	    }
	    applicationNode.add(experimentNode);
	    inNode.add(applicationNode);
	    cnt1++;
	}
    } 
  
    public void expandDefaultParaProfTrialNode(){
	if(defaultParaProfTrialNode != null)
	    tree.expandPath(new TreePath(defaultParaProfTrialNode.getPath()));
    }

    //******************************
    //Event listener code!!
    //******************************
  
    //ActionListener code.
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
    JSplitPane jSplitPane = null;
  
    //A reference to the default trial node.
    DefaultMutableTreeNode defaultParaProfTrialNode = null;

    private String password = "";
    private String configFile = "/Users/bertie/Desktop/Programming/data/pi/perfdb.cfg";
    private Vector loadedTrials = new Vector();
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
	DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode) node.getParent();
	Object userObject = node.getUserObject();
	
	if(parentNode.isRoot()){
	    URL url = ParaProfTreeCellRenderer.class.getResource("red-ball.gif");
	    this.setIcon(new ImageIcon(url));
	}
	else if(userObject instanceof Metric){
	    URL url = ParaProfTreeCellRenderer.class.getResource("green-ball.gif");
	    this.setIcon(new ImageIcon(url));
	}
	return this;
    }
}
