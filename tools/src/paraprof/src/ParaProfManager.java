
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
      
	    DefaultMutableTreeNode standard = new DefaultMutableTreeNode(new ParaProfApplicationType("Standard Applications"));
	    //Populate this node.
	    this.populateStandardApplications(standard);
	    root.add(standard);
      
	    DefaultMutableTreeNode runtime = new DefaultMutableTreeNode(new ParaProfApplicationType("Runtime Applications"));
	    root.add(runtime);
      
	    dbApps = new DefaultMutableTreeNode(new ParaProfApplicationType("DB Applications"));
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
      

	    //Add listeners to buttons.
	    connectDisconnect.addActionListener(new ActionListener(){
		    public void actionPerformed(ActionEvent evt){
                        connectDisconnect();}});
	    refreshApplications.addActionListener(new ActionListener(){
		    public void actionPerformed(ActionEvent evt){
                        refreshApplications();}});
	    refreshExperiments.addActionListener(new ActionListener(){
		    public void actionPerformed(ActionEvent evt){
                        refreshExperiments();}});
      
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
		if(tmpString.equals("Standard Applications")){
		    innerPane.setRightComponent(getSAppUpperRight());
		    innerPane2.setLeftComponent(getAppTypeNodeLowerLeft());
		    setJSPDividers(innerPane, innerPane2, 0.5, 0.5);
		    tree.expandPath(path);
		}
		else if(tmpString.equals("Runtime Applications")){
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
	tmpJButton = new JButton("Add Application");
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
	tmpJButton = new JButton("Add Experiment");
	tmpJButton.addActionListener(new ActionListener(){
                public void actionPerformed(ActionEvent evt){
		    addExperiment();}
	    });
	tmpJPanel.add(tmpJButton);
	return tmpJPanel;
    }
  
    private Component getParaProfTrialTypeNodeLowerLeft(){
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
	gbc.anchor = GridBagConstraints.WEST;
	gbc.weightx = 100;
	gbc.weighty = 0;
	panelAdd(jPanel, jButton, gbc, 0, 0, 2, 1);
	
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(jPanel, jLabel, gbc, 0, 1, 1, 1);

	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.WEST;
	gbc.weightx = 100;
	gbc.weighty = 0;
	panelAdd(jPanel, trialType, gbc, 1, 1, 1, 1);

	return jPanel;
    }
  
    private Component getDBAppTypeNodeLowerLeft(){
	JButton tmpJButton = null;
	JPanel tmpJPanel = new JPanel();
	tmpJButton = new JButton("Add ParaProfApplication");
	tmpJButton.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent evt){
		    newParaProfApplicationButton();}
	    });

	JLabel configFileLabel = new JLabel("Config File:");
	JLabel usernameFieldLabel = new JLabel("Username:");
	JLabel passwordFieldLabel = new JLabel("Password:");
    
    
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
	panelAdd(tmpJPanel, configFileLabel, gbc, 0, 1, 1, 1);
    
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 100;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, configFileField, gbc, 1, 1, 1, 1);
    
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
	panelAdd(tmpJPanel, connectDisconnect, gbc, 0, 4, 1, 1);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 0;
	gbc.weighty = 0;
	panelAdd(tmpJPanel, refreshApplications, gbc, 1, 4, 1, 1);
    
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
	return (new JScrollPane(new JTable(new ParaProfTableModel(inApp))));
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
		    applyOperation();}
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
	    trial = new ParaProfTrial(type);
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
	      //Clear node.
	      for(int i=dbApps.getChildCount(); i>0; i--){
		  treeModel.removeNodeFromParent(((DefaultMutableTreeNode) dbApps.getChildAt(i-1)));
	      }
 	      ListIterator l = perfDBSession.getApplicationList();
	      while (l.hasNext()){
		  ParaProfApplication paraProfApplication = new ParaProfApplication((Application)l.next());
		  paraProfApplication.setDBParaProfApplication(true);
		  //Now add the application to the node.
		  treeModel.insertNodeInto(new DefaultMutableTreeNode(paraProfApplication), dbApps, dbApps.getChildCount());
	      }
	      return;
	  }
	  catch(Exception e){
	      ParaProf.systemError(e, null, "ELM03");
	  }
    }
  
    public void refreshExperiments(){
	try{
	    TreePath path = tree.getSelectionPath();
	    if(path == null)
		return;
	    DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
	    Object userObject = selectedNode.getUserObject();
	    
	    if(userObject instanceof ParaProfApplication){
		//Clear the node before adding anything else.
		for(int i=selectedNode.getChildCount(); i>0; i--){
		    treeModel.removeNodeFromParent(((DefaultMutableTreeNode) selectedNode.getChildAt(i-1)));
		}
		
		int applicationID = ((ParaProfApplication) userObject).getID();
		perfDBSession.setApplication(applicationID);
		ListIterator l = perfDBSession.getExperimentList();
		while (l.hasNext()){
		    ParaProfExperiment paraProfExperiment = new ParaProfExperiment((Experiment)l.next());
		    paraProfExperiment.setApplication((ParaProfApplication) userObject);
		    treeModel.insertNodeInto(new DefaultMutableTreeNode(paraProfExperiment), dbApps, dbApps.getChildCount());
		}
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
  
    void showMetric(ParaProfTrial inParaProfTrial, Metric inMetric){
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
  
    void populateStandardApplications(DefaultMutableTreeNode inNode){
    
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
    JSplitPane innerPane2 = null;
    JSplitPane outerPane = null;
  
    //A reference to the default trial node.
    DefaultMutableTreeNode defaultParaProfTrialNode = null;

    private PerfDBSession perfDBSession = null;
    private JButton connectDisconnect = new JButton("Connect to Database");
    private JButton refreshApplications = new JButton("Refresh Applications");
    private JButton refreshExperiments = new JButton("Refresh Experiments");
    private JTextField configFileField = new JTextField(System.getProperty("user.dir"), 30);
    private JTextField usernameField = new JTextField("username", 30);
    private JPasswordField passwordField = new JPasswordField("password", 30);

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

class ParaProfTableModel extends AbstractTableModel{
    public ParaProfTableModel(Application application){
	super();
	this.application = application;
    }
  
    public int getColumnCount(){
	return 2;}
  
    public int getRowCount(){
	return 7;}
  
    public String getColumnName(int c){
	return columnNames[c];}
  
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
		returnObject = application.getName();
		break;
	    case(1):
		returnObject = new Integer(application.getID());
		break;
	    case(2):
		returnObject = application.getLanguage();
		break;
	    case(3):
		returnObject = application.getParaDiag();
		break;
	    case(4):
		returnObject = application.getUsage();
		break;
	    case(5):
		returnObject = application.getExecutableOptions();
		break;
	    case(6):
		returnObject = application.getDescription();
		break;
	    }
	}
    
	return returnObject; 
          
    }
  
    public boolean isCellEditable(int r, int c){
	if(c==1 && r!=1)
	    return true;
	else
	    return false;
    }
  
    public void setValueAt(Object obj, int r, int c){
	//Should be getting a string I think.
	if(obj instanceof String){
	    String tmpString = (String) obj;
	    if(c==1){
		switch(r){
		case(0):
		    application.setName(tmpString);
		    break;
		case(1):
		    application.setID(Integer.parseInt(tmpString));
		    break;
		case(2):
		    application.setLanguage(tmpString);
		    break;
		case(3):
		    application.setParaDiag(tmpString);
		    break;
		case(4):
		    application.setUsage(tmpString);
		    break;
		case(5):
		    application.setExecutableOptions(tmpString);
		    break;
		case(6):
		    application.setDescription(tmpString);
		    break;
		}
	    }
	}
    }
  
    private Application application = null;
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
