/*
 * Created on Jul 16, 2003
 * This class shows the main window.
 *
 */
 
package edu.uoregon.tau.viewer.GUI;

import javax.swing.*;
import javax.swing.tree.*;
import javax.swing.table.*;
import javax.swing.event.*;
import java.net.URL;
import java.io.IOException;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.io.*;

import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.analysis.*;
import edu.uoregon.tau.viewer.apiext.*;
import edu.uoregon.tau.viewer.perfcomparison.*;
import edu.uoregon.tau.viewer.iowithMatlab.*;


/**
 * @author Li Li
 *
 */
public class MainWindow extends JFrame implements ActionListener{

	/* (non-Javadoc)
	 * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
	 */
	private DataSession dbSession;
	private String dbConfigFile;
	private boolean playWithLineStyle = true;
	private String lineStyle = "Angled";

	// Three primary components of main window.

	//tree view of perfdb
	private JTree DBTree;
	private DefaultTreeModel DBTreeModel; 
	private DefaultMutableTreeNode root;

	//db content table and performance data table
	private JTable dbTable, profTable; 
	private JLabel dbLabel, profLabel;

	// defalut staff
	private String mainRoutineName = "main";
	private String defaultMetric = "time";

	// matlab engine.
	private Engine matlabEngine = new Engine();;
	private String matlabCmd = "matlab -nosplash -nojvm";

	public MainWindow(DataSession session, String dbConfigFile){
		super("Main Window");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		// set menu bar.
		setJMenuBar(createMenuBar());

		this.dbSession = session;				
		this.dbConfigFile = dbConfigFile;

		//Create the tree nodes.
		root = new DefaultMutableTreeNode("PerfDMF");
		buildNodes(root);
				
		//Create the tree.
		DBTreeModel = new DefaultTreeModel(root);
		DBTree = new JTree(DBTreeModel);
		
		DBTree.getSelectionModel().setSelectionMode(TreeSelectionModel.DISCONTIGUOUS_TREE_SELECTION);

		if (playWithLineStyle) {
		    DBTree.putClientProperty("JTree.lineStyle", lineStyle);
		}

		//Create the scroll pane and add the tree into it. 
		JScrollPane treeView = new JScrollPane(DBTree);	
				
		//Create the database content viewing pane.		
		JPanel dbView = new JPanel(new BorderLayout());

		dbTable = new JTable();
 		profTable = new JTable();		
		dbLabel = new JLabel();
		dbLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
		profLabel = new JLabel();
		profLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
				
		dbView.setLayout(new BoxLayout(dbView, BoxLayout.Y_AXIS));		
		dbView.add(Box.createVerticalStrut(20));
		dbView.add(dbLabel);		
		dbTable.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		JScrollPane dbPane = new JScrollPane(dbTable);
		//dbView.add(dbTable.getTableHeader());
		//dbView.add(dbTable);
		dbView.add(dbPane);		

		profTable.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		JScrollPane profPane = new JScrollPane(profTable);
		dbView.add(profLabel);
		dbView.add(profPane); 
		
		//Create a split pane for these two panes.
		JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
		splitPane.setLeftComponent(treeView);
		splitPane.setRightComponent(dbView);
	
		Dimension minimumSize = new Dimension(200, 100);
		dbView.setMinimumSize(minimumSize);
		treeView.setMinimumSize(minimumSize);
		splitPane.setDividerLocation(300); 
		splitPane.setPreferredSize(new Dimension(1000, 600));

		//Add the split pane to this frame.
		getContentPane().add(splitPane, BorderLayout.CENTER);	

		// Listen for when the selection changes.
		DBTree.addTreeSelectionListener(new TreeSelectionListener() { // action performed when one node is seleted.
			public void valueChanged(TreeSelectionEvent e) {
				DefaultMutableTreeNode node = (DefaultMutableTreeNode) 
				DBTree.getLastSelectedPathComponent();		      

				DBContent dbContent = null, profData = null;

				if (node == null) return;

				if (!node.isRoot()) { // PerfDMF node is NOT selected.
				    Wrapper nodeInfo = (Wrapper)node.getUserObject();
				
					// IF THE USER SELECTED AN APPLICATION NODE,
					// THEN POPULATE THE RIGHT WINDOW WITH APPLICATION
					// DATA!

				    if (nodeInfo.app != null){	// application node				
						//Generate a table for this application .
						Application anApp = nodeInfo.app;
						String[] columnNames = {"Application-Name", "Version", "Description", "Language", "Parallel-Paradigm", "Usage", "Execution-options"};
						Object[][] rows = { { anApp.getName(), anApp.getVersion(), anApp.getDescription(), anApp.getLanguage(), anApp.getParaDiag(), anApp.getUsage(), anApp.getExecutableOptions()} };
						dbContent = new DBContent(rows, columnNames); // db table model.
						dbTable.setModel(dbContent); // set db table.
						dbLabel.setVisible(true);
						dbLabel.setText("Application information");
						for (int i=0; i<dbTable.getColumnCount(); i++) {
							dbTable.getColumnModel().getColumn(i).setPreferredWidth(150);
						}
						profLabel.setVisible(false);
						profTable.setModel(new DefaultTableModel());
				    }

					// IF THE USER SELECTED AN EXPERIMENT NODE,
					// THEN POPULATE THE RIGHT WINDOW WITH
					// EXPERIMENT DATA!

				    else if (nodeInfo.exp !=null){ // experiment node
						Experiment anExp = nodeInfo.exp;
						String[] columnNames = {"Name", 
											"System name", "System machine type", " Processor architecture", "O/S", "Memory size", "Number of processors", "L1 cache size", "L2 cache size", 
											"TAU config -prefix", "TAU config -arch", "TAU config -CPP", "TAU config -CC", "TAU config -java", "TAU config -profile", 
											"CPP compiler name", "CPP compiler version", "CC compiler Name", "CC compiler version", "Java location", "Java version"};
						Object[][] rows = {{anExp.getName(), anExp.getSystemName(), anExp.getSystemMachineType(), anExp.getSystemArch(), anExp.getSystemOS(), anExp.getSystemMemorySize(), anExp.getSystemProcessorAmount(), anExp.getSystemL1CacheSize(), anExp.getSystemL2CacheSize(), anExp.getConfigPrefix(), anExp.getConfigArchitecture(), anExp.getConfigCpp(), anExp.getConfigCc(), anExp.getConfigJdk(), anExp.getConfigProfile(), anExp.getCompilerCppName(), anExp.getCompilerCppVersion(), anExp.getCompilerCcName(), anExp.getCompilerCcVersion(), anExp.getCompilerJavaDirpath(), anExp.getCompilerJavaVersion()} };
						dbContent = new DBContent(rows, columnNames); // db table model.
						dbTable.setModel(dbContent); // set db table.
						dbLabel.setVisible(true);
						dbLabel.setText("Experiment information");
						for (int i=0; i<dbTable.getColumnCount(); i++) {
							dbTable.getColumnModel().getColumn(i).setPreferredWidth(150);
						}
						profLabel.setVisible(false);
						profTable.setModel(new DefaultTableModel());
				    }

					// IF THE USER SELECTED A TRIAL NODE, THEN POPULATE
					// THE RIGHT WINDOW WITH TRIAL DATA!

 				    else if (nodeInfo.trial != null){ // trial node
						Trial aTrial = nodeInfo.trial;							
										
						// reset perfdbsession 
						dbSession.reset();
						dbSession.setTrial(aTrial.getID());

						// get metric list
						String metricList = "";
						ListIterator metrics = new DataSessionIterator(dbSession.getMetrics());
						while (metrics.hasNext()) {
							Metric m = (Metric)metrics.next();
							metricList += m.getName();
							if (metrics.hasNext()) metricList += "; ";
						}
					
						// get default metric, usually, execution time.
						Metric currentMetric = findADefaultMetric(aTrial, false);
						dbSession.setMetric(currentMetric);				

						// get the function data
						ListIterator funcData = dbSession.getIntervalEventData();
						String[] profColumnNames = {"IntervalEvent-name", "Node", "Context", "Thread","inclusive%", "inclusive  ", "exclusive%", "exclusive  ", "#Call", "#Subrs", "Inclusive/Call"};
						Vector rowsVec = new Vector();
						double maxMainTime = 0.0;

						while (funcData.hasNext()){
					    	IntervalLocationProfile aFunc = (IntervalLocationProfile)funcData.next();
					    	IntervalEventProfile aFuncObject = new IntervalEventProfile(dbSession, aFunc);

					    	if (aFuncObject.getInclusivePercentage() == 100.0)
							if (aFuncObject.getInclusive() > maxMainTime)
						    	maxMainTime = aFuncObject.getInclusive();
					    
					    	Vector aRow = new Vector();					    					    
					    	aRow.add(aFuncObject.getName());
					    	aRow.add(new Integer(aFuncObject.getNode()));
					    	aRow.add(new Integer(aFuncObject.getContext()));
					    	aRow.add(new Integer(aFuncObject.getThread()));
					    	aRow.add(new Double(aFuncObject.getInclusivePercentage()));
					    	aRow.add(new Double(aFuncObject.getInclusive()));
					    	aRow.add(new Double(aFuncObject.getExclusivePercentage()));
					    	aRow.add(new Double(aFuncObject.getExclusive()));
					    	aRow.add(new Integer(aFuncObject.getNumCalls()));
					    	aRow.add(new Integer(aFuncObject.getNumSubroutines()));
					    	aRow.add(new Double(aFuncObject.getInclusivePerCall()));
					    	rowsVec.add(aRow);
						}										
					
						profData = new DBContent(rowsVec, profColumnNames);
						profTable.setModel(profData);
						profData.addMouseListenerToTableHeader(profTable);
						profTable.getColumnModel().getColumn(0).setPreferredWidth(300);
						profLabel.setVisible(true);
						profLabel.setText("Performance data ( "+currentMetric.getName()+" ) for the trial");

						// write down trial information.
						String[] columnNames = {"Date of the trial", "Problem definition", "#Node", "#Context", "#Thread", "Execution time", "Available Counter(s)"};
					
						Object[][] rows = { {aTrial.getTime(), aTrial.getProblemDefinition(), new Integer(aTrial.getNodeCount()), new Integer(aTrial.getNumContextsPerNode()), new Integer(aTrial.getNumThreadsPerContext()), usec2String(maxMainTime), metricList}};
						dbContent = new DBContent(rows, columnNames); // db table model.
						dbTable.setModel(dbContent); // db table.
						for (int i=0; i<dbTable.getColumnCount(); i++)
							dbTable.getColumnModel().getColumn(i).setPreferredWidth(150);
						dbLabel.setVisible(true);
						dbLabel.setText("Trial information");							
				    }

					// IF THE USER SELECTED A FUNCTION NODE, THEN POPULATE
					// THE RIGHT WINDOW WITH FUNCTION DATA!

				    else { // function node
					
						IntervalEvent aIntervalEvent = nodeInfo.func;
						DefaultMutableTreeNode trialNode = (DefaultMutableTreeNode)(node.getParent());
						Trial fatherTrial = ((Wrapper)(trialNode.getUserObject())).trial;
						String[] columnNames = {"Date of the trial", "Problem definition", "#Node", "#Context", "#Thread", "Available Counter(s)"};
						String tmpStr="";
						Metric metricForTheTrial = findADefaultMetric(fatherTrial, false);
					
						String metricList = "";
						ListIterator metrics = new DataSessionIterator(fatherTrial.getMetrics());
						while (metrics.hasNext()) {
							Metric m = (Metric)metrics.next();
							tmpStr += m.getName();
							if (metrics.hasNext()) tmpStr += "; ";
						}
					
						Object[][] rows = { {fatherTrial.getTime(), fatherTrial.getProblemDefinition(), new Integer(fatherTrial.getNodeCount()), new Integer(fatherTrial.getNumContextsPerNode()), new Integer(fatherTrial.getNumThreadsPerContext()), tmpStr}};
						dbContent = new DBContent(rows, columnNames); // db table model.
						dbTable.setModel(dbContent); // db table.
						dbLabel.setVisible(true);
						dbLabel.setText("Trial information");	
						for (int i=0; i<dbTable.getColumnCount(); i++)
							dbTable.getColumnModel().getColumn(i).setPreferredWidth(150);		

						// reset perfdbsession 
						dbSession.reset();
						dbSession.setTrial(fatherTrial.getID());
						dbSession.setMetric(metricForTheTrial);
						dbSession.setIntervalEvent(aIntervalEvent);
					
						ListIterator intervalEventData = dbSession.getIntervalEventData();
						String[] profColumns = {"Node", "Context", "Thread","inclusive%", "inclusive  ", "exclusive%", "exclusive ", "#Call", "#Subroutines", "Inclusive/Call"};
					
						Vector profRows = new Vector();
						IntervalEventProfile aFunc;
						while (intervalEventData.hasNext()){
					    	aFunc = new IntervalEventProfile(dbSession, (IntervalLocationProfile)intervalEventData.next());
					    	Vector oneRow = new Vector();					    
					    	oneRow.add(new Integer(aFunc.getNode()));
					    	oneRow.add(new Integer(aFunc.getContext()));
					    	oneRow.add(new Integer(aFunc.getThread()));
					    	oneRow.add(new Double(aFunc.getInclusivePercentage()));
					    	oneRow.add(new Double(aFunc.getInclusive()));
					    	oneRow.add(new Double(aFunc.getExclusivePercentage()));
					    	oneRow.add(new Double(aFunc.getExclusive()));
					    	oneRow.add(new Integer(aFunc.getNumCalls()));
					    	oneRow.add(new Integer(aFunc.getNumSubroutines()));
					    	oneRow.add(new Double(aFunc.getInclusivePerCall()));
					    	profRows.add(oneRow);
						} 
					
						profData = new DBContent(profRows, profColumns);
						profTable.setModel(profData);
						profData.addMouseListenerToTableHeader(profTable);
						profLabel.setText("Performance data ( "+metricForTheTrial.getName()+" ) for "+aIntervalEvent.getName());
						profLabel.setVisible(true);
				    }
				}
			}
		});						
	}

	// this method creates all menus and submenus of the main window.
	private JMenuBar createMenuBar(){
		JMenuBar menuBar = new JMenuBar();
	   
		// first menu.
		JMenu dbMenu = new JMenu("Database");
		menuBar.add(dbMenu);

		JMenuItem dbItem2 = new JMenuItem("Load application");
		dbItem2.addActionListener(this);
		dbMenu.add(dbItem2);
		dbMenu.addSeparator();

		JMenuItem dbItem3 = new JMenuItem("Load experiment");
		dbItem3.addActionListener(this);
		dbMenu.add(dbItem3);
		dbMenu.addSeparator();

		JMenuItem dbItem4 = new JMenuItem("Load trial");
		dbItem4.addActionListener(this);
		dbMenu.add(dbItem4);
		dbMenu.addSeparator();

		JMenuItem dbItem5 = new JMenuItem("Load hw performance counter");
		dbItem5.addActionListener(this);
		dbMenu.add(dbItem5);
		dbMenu.addSeparator();

		/***** some todos
		JMenuItem dbItem5 = new JMenuItem("Delete application");
		dbItem5.addActionListener(this);
		dbMenu.add(dbItem5);
		dbMenu.addSeparator();

		JMenuItem dbItem6 = new JMenuItem("Delete experiment");
		dbItem6.addActionListener(this);
		dbMenu.add(dbItem6);
		dbMenu.addSeparator();

		JMenuItem dbItem7 = new JMenuItem("Delete trial");
		dbItem7.addActionListener(this);
		dbMenu.add(dbItem7);
		dbMenu.addSeparator();

		JMenuItem dbItem8 = new JMenuItem("Open another database");
		dbItem8.addActionListener(this);
		dbMenu.add(dbItem8);
		dbMenu.addSeparator();
		******/

		JMenuItem dbItem1 = new JMenuItem("Exit main window");
		dbItem1.addActionListener(this);
		dbMenu.add(dbItem1); 
	   
		// second menu.
		JMenu opMenu = new JMenu("Operations");
		menuBar.add(opMenu);

		// submanu 1.
		JMenu opItem1 = new JMenu("Compare");

		JMenuItem compareItem1 = new JMenuItem("Trials");
		compareItem1.addActionListener(this);
		opItem1.add(compareItem1);
		opItem1.addSeparator();

		JMenuItem compareItem2 = new JMenuItem("IntervalEvents");
		compareItem2.addActionListener(this);
		opItem1.add(compareItem2);

		opMenu.add(opItem1);

		JMenu opItem2 = new JMenu("Scalability");

		JMenuItem speedupItem = new JMenuItem("Speedup");
		speedupItem.addActionListener(this);
		//speedupItem.setEnabled(false);
		opItem2.add(speedupItem);
		opItem2.addSeparator();

		JMenuItem exeItem = new JMenuItem("Execution time");
		exeItem.addActionListener(this);
		opItem2.add(exeItem);
		opItem2.addSeparator();

		JMenuItem commItem = new JMenuItem("MPI Communication time");
		//commItem.setEnabled(false);
		commItem.addActionListener(this);
		opItem2.add(commItem);
	   
		/*opItem2.addSeparator();
		JMenuItem collectiveItem = new JMenuItem("Collective communication time");
		JMenuItem p2pItem = new JMenuItem("P2p communication time");*/
	   
		opMenu.add(opItem2);
	   	   	   
		// option menu
		JMenu optionMenu = new JMenu("Options");
		menuBar.add(optionMenu);

		JMenuItem optionItem1 = new JMenuItem("show mean statistics");
		optionItem1.addActionListener(this);
		optionMenu.add(optionItem1);
		optionMenu.addSeparator();

		JMenuItem optionItem2 = new JMenuItem("show total statistics");
		optionItem2.addActionListener(this);
		optionMenu.add(optionItem2);
		optionMenu.addSeparator();

		JMenuItem optionItem4 = new JMenuItem("show user-defined events");
		optionItem4.addActionListener(this);
		optionMenu.add(optionItem4);
		optionMenu.addSeparator();

		JMenuItem optionItem3 = new JMenuItem("show counter");
		optionItem3.addActionListener(this);
		optionMenu.add(optionItem3);	   

		// help menu
		JMenu helpMenu = new JMenu("Help");
		menuBar.add(helpMenu);

		return menuBar;
	}
	 
	// this method create node hierarchy of perfdb tree
   	private void buildNodes(DefaultMutableTreeNode root){
		// create children nodes for root, i.e. applications
		ListIterator applications = dbSession.getApplicationList(), experiments, trials, functions;
		Application oneApp = null;
		Experiment oneExp;
		Trial oneTrial;
		IntervalEvent oneIntervalEvent;
		DefaultMutableTreeNode appNode, expNode, trialNode, funcNode;
		
		while (applications.hasNext()){
			// Get one application.
			oneApp = (Application)applications.next();

			// Create a node for it.
			appNode = new DefaultMutableTreeNode(new Wrapper(oneApp));
			root.add(appNode);
			
			// Get experiment list associated with this application.
			dbSession.setApplication(oneApp);
			experiments = dbSession.getExperimentList();
			oneExp = null;
			
			while (experiments.hasNext()){
				oneExp = (Experiment)experiments.next();
				expNode = new DefaultMutableTreeNode(new Wrapper(oneExp));
				appNode.add(expNode);
				
				// Get trial list associated with this experiment.
				dbSession.setExperiment(oneExp);
				trials = dbSession.getTrialList();
				oneTrial = null;
				
				while (trials.hasNext()){
					oneTrial = (Trial)trials.next();
					trialNode = new DefaultMutableTreeNode(new Wrapper(oneTrial));
					expNode.add(trialNode);
					
					// Get function list associated with this trial.
					dbSession.setTrial(oneTrial);
					functions = dbSession.getIntervalEvents();
					oneIntervalEvent = null;
					
					while (functions.hasNext()){
					    oneIntervalEvent = (IntervalEvent)functions.next();
					    funcNode = new DefaultMutableTreeNode(new Wrapper(oneIntervalEvent));
					    trialNode.add(funcNode);
					}					
				}
			}			
		}
	}    
	
	//this mathod is called by AppLoadDialog class to load an applicaiton
	public void loadApplication(String fileName){
/*
		LoadApplication loadManager = new LoadApplication(dbConfigFile);
		loadManager.getConnector().connect();	
		String appID = loadManager.storeApp(fileName);
		loadManager.getConnector().dbclose();
		Application newApp = dbSession.setApplication(Integer.parseInt(appID));
		// show the new application on tree view.
		DefaultMutableTreeNode newAppNode =  new DefaultMutableTreeNode(new Wrapper(newApp));
		DBTreeModel.insertNodeInto(newAppNode, root, root.getChildCount());
		TreePath newTreePath = new TreePath(newAppNode.getPath());
		DBTree.scrollPathToVisible(newTreePath);
		DBTree.setSelectionPath(newTreePath);
*/
 	}
   
    // this method is called by ExpLoadDialog class to load an experiment.
	public void loadExperiment(DefaultMutableTreeNode selectedAppNode, String fileName){
/*
		LoadExperiment loadManager = new LoadExperiment(dbConfigFile);
		Wrapper nodeInfo = (Wrapper)selectedAppNode.getUserObject();
		String appID = String.valueOf(nodeInfo.app.getID());
		loadManager.getConnector().connect();
		String expID = loadManager.storeExp(appID, fileName);
		loadManager.getConnector().dbclose();
		Experiment newExp = dbSession.setExperiment(Integer.parseInt(expID));
		// show the new experiment on tree view.
		DefaultMutableTreeNode newExpNode =  new DefaultMutableTreeNode(new Wrapper(newExp));
		DBTreeModel.insertNodeInto(newExpNode, selectedAppNode, selectedAppNode.getChildCount());
		TreePath newTreePath = new TreePath(newExpNode.getPath());
		DBTree.scrollPathToVisible(newTreePath);
		DBTree.setSelectionPath(newTreePath);
*/
    }

	//	this method is called by TrialLoadDialog class to load a trial.
    public void loadTrial(DefaultMutableTreeNode selectedExpNode, String pprofdatFile, String probSize){
/*
		// grab application id and experiment id.
		Wrapper nodeInfo = (Wrapper)selectedExpNode.getUserObject();
		String expID = String.valueOf(nodeInfo.exp.getID());
		String appID = String.valueOf(nodeInfo.exp.getApplicationID());

		// translate pprof.dat into pprof.xml file
		translator.Translator trans = new translator.Translator(dbConfigFile, pprofdatFile, "pprof.xml");
		trans.buildPprof();
		trans.writeXmlFiles(appID, expID, "");
	
		// load the pprof.xml file into database.
		perfdb.loadxml.Main loadManager = new perfdb.loadxml.Main(dbConfigFile);
		loadManager.getConnector().connect();

		String trialID = loadManager.storeDocument("pprof.xml", Integer.toString(0), probSize);

		loadManager.getConnector().dbclose();
		File xmlFile = new File("pprof.xml");
		xmlFile.delete();

		// insert the new trial node into DB tree.
		dbSession.setApplication(Integer.parseInt(appID));
		dbSession.setExperiment(Integer.parseInt(expID));
		Trial newTrial = dbSession.setTrial(Integer.parseInt(trialID));
		
		DefaultMutableTreeNode newTrialNode =  new DefaultMutableTreeNode(new Wrapper(newTrial));
		DBTreeModel.insertNodeInto(newTrialNode, selectedExpNode, selectedExpNode.getChildCount());

		// insert functions associated with this trial.
		IntervalEvent nullFunc = null;
		dbSession.setIntervalEvent(nullFunc);

		ListIterator functions = dbSession.getIntervalEvents();
		IntervalEvent oneIntervalEvent;
		DefaultMutableTreeNode funcNode;

		while (functions.hasNext()){
	    	oneIntervalEvent = (IntervalEvent)functions.next();
	    	funcNode = new DefaultMutableTreeNode(new Wrapper(oneIntervalEvent));
	    	DBTreeModel.insertNodeInto(funcNode, newTrialNode, newTrialNode.getChildCount());
		}	

		//// Make sure the user can see the new node.
		TreePath newTreePath = new TreePath(newTrialNode.getPath());
		DBTree.scrollPathToVisible(newTreePath);
		DBTree.setSelectionPath(newTreePath);
*/
    }
	
	public void loadHWCounterData(DefaultMutableTreeNode selectedTrialNode, String pprofdatFileName){
/*
		// grab trial node, experiment ID, application ID 
		Wrapper nodeInfo = (Wrapper)selectedTrialNode.getUserObject();
		String expID = String.valueOf(nodeInfo.trial.getExperimentID());
		DefaultMutableTreeNode parentExpNode = (DefaultMutableTreeNode) (selectedTrialNode.getParent());
		Experiment parentExp = ((Wrapper)(parentExpNode.getUserObject())).exp;
		String appID = String.valueOf(parentExp.getApplicationID());
			    
		// translate pprof.dat file into pprof.xml
		translator.Translator trans = new translator.Translator(dbConfigFile, pprofdatFileName, "pprof.xml");
		trans.buildPprof();
		trans.writeXmlFiles(appID, expID, "");
			    
		// load the pprof.xml file into database.
		perfdb.loadxml.Main loadManager = new perfdb.loadxml.Main(dbConfigFile);
		loadManager.getConnector().connect();

		loadManager.storeDocument("pprof.xml", Integer.toString(nodeInfo.trial.getID()), null);
		// update the trial node.
		Trial updatedTrial = dbSession.setTrial(nodeInfo.trial.getID());
		selectedTrialNode.setUserObject(new Wrapper(updatedTrial));

		File xmlFile = new File("pprof.xml");
		xmlFile.delete();	
*/
	}
		 
	public void actionPerformed(ActionEvent evt) {
	    Object eventObj = evt.getSource();
      
	    if (eventObj instanceof JMenuItem){ // if this event is fired by a menu item.
		String command = evt.getActionCommand();
		
		if (command.equals("Exit main window")){
		    if (matlabEngine.isOpen())
				try{ matlabEngine.close();
				} catch (Exception e){ }

		    dbSession.terminate();
		    setVisible(false);
		    dispose();		    
		    System.exit(0);
		}
		
		else if (command.equals("Load application")){
		    // pop up a dialog asking for the location of xml file
		    AppLoadDialog aDialog = new AppLoadDialog(this);
		}

		else if (command.equals("Load experiment")){
		    TreePath[] selectedTreePaths = DBTree.getSelectionPaths();
		    if (selectedTreePaths.length > 1){
				JOptionPane.showMessageDialog(this, "Please select only one application group", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    }
		    else if (selectedTreePaths.length == 0){ 
				JOptionPane.showMessageDialog(this, "Please first select the application that the new experiment belongs to", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    }
		    else {
				DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode)selectedTreePaths[0].getLastPathComponent();
				Wrapper nodeInfo = (Wrapper)selectedNode.getUserObject();
				if (nodeInfo.app == null){
			    	JOptionPane.showMessageDialog(this, "Wrong node is selected for loading experiment.", "Warning!", JOptionPane.ERROR_MESSAGE);
			    	return;
				}
				else{
			    	// pop up a dialog asking for experiment information			    
			    	ExpLoadDialog aDialog = new ExpLoadDialog(selectedNode, this);			    				    
				}
		    }
		}
		
		else if (command.equals("Load trial")){
		    TreePath[] selectedTreePaths = DBTree.getSelectionPaths();
		    if (selectedTreePaths.length > 1){
				JOptionPane.showMessageDialog(this, "Please select only one experiment group", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    }
		    else if (selectedTreePaths.length == 0){ 
				JOptionPane.showMessageDialog(this, "Please first select the expriment group that the new trial belongs to", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    }
		    else {
				DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode)selectedTreePaths[0].getLastPathComponent();
				Wrapper nodeInfo = (Wrapper)selectedNode.getUserObject();
				if (nodeInfo.exp == null){
			    	JOptionPane.showMessageDialog(this, "Wrong node is selected for loading trial.", "Warning!", JOptionPane.ERROR_MESSAGE);
			    	return;
				}
				else{
			    	// pop up a dialog asking for the trial information
			    	TrialLoadDialog aDialog = new TrialLoadDialog(selectedNode, this);
				}
		    }
		}

		else if (command.equals("Load hw performance counter")){
		    TreePath[] selectedTreePaths = DBTree.getSelectionPaths();
		    if ((selectedTreePaths.length > 1) || (selectedTreePaths.length == 0)){
				JOptionPane.showMessageDialog(this, "Please select only one trial group", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    }
		    else {
				DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode)selectedTreePaths[0].getLastPathComponent();
				Wrapper nodeInfo = (Wrapper)selectedNode.getUserObject();
				if (nodeInfo.trial == null){
			    	JOptionPane.showMessageDialog(this, "Wrong node is selected for loading hw performance data.", "Warning!", JOptionPane.ERROR_MESSAGE);
				    
			    	return;
				}
				else{
			 		MultiCounterLoadDialog adialog = new MultiCounterLoadDialog(selectedNode, this);   
				}		    
		    }
		}

		/*else if (command.equals("Delete application")){}
		else if (command.equals("Delete experiment")){}
		else if (command.equals("Delete trial")){}
		else if (command.equals("Open another database")){
		    String dbName = JOptionPane.showInputDialog(this, "Please input database name", 
								"Open another database", 
								JOptionPane.INFORMATION_MESSAGE);
		    // start to open another db
		    
		   
		}*/

		else if (command.equals("Trials")){ // compare trials
		    TreePath[] selectedTreePaths = DBTree.getSelectionPaths();
		    if (selectedTreePaths == null){
				JOptionPane.showMessageDialog(this, "Please first select trials to be compared.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    } else { // some tree nodes are selected
				if (selectedTreePaths.length == 1){
			    	JOptionPane.showMessageDialog(this, "Please select at least two trials for comparing.", "Warning!", JOptionPane.ERROR_MESSAGE);
			    
			    	return;
				} else {
			    	Vector trials = new Vector();
			    	DefaultMutableTreeNode selectedNode;
			    	Trial currentTrial;
			    	String metric = null;
			    	int numberOfMetrics, k;
			    	String[] metricNames;
					Metric metricObj = null;
			    
			    	for (int i=0; i<selectedTreePaths.length; i++) {
						selectedNode = (DefaultMutableTreeNode)selectedTreePaths[i].getLastPathComponent(); 
						if (selectedNode.isRoot()) {// PerfDMF node is selected.
				    		JOptionPane.showMessageDialog(this, "Wrong node is selected for trial comparison.", "Warning!", JOptionPane.ERROR_MESSAGE);
				    		return;
						}
			    		Wrapper nodeInfo = (Wrapper)selectedNode.getUserObject();
			    		if (nodeInfo.trial == null) {
							JOptionPane.showMessageDialog(this, "Wrong node is selected for trial comparison.", "Warning!", JOptionPane.ERROR_MESSAGE);
							return;
			    		}
						currentTrial = nodeInfo.trial;
						numberOfMetrics = currentTrial.getMetricCount();
						if (i==0) {					   
			   				metricNames = buildMetricNameArray(currentTrial);
			   				metric = (String)JOptionPane.showInputDialog(this, "Please select a COMMON counter of these trials to compare.", "Select counter", JOptionPane.PLAIN_MESSAGE, null, metricNames, metricNames[0]);
			   				if (metric == null)
			       				return; 
						} else {			    
			    			metricObj = findMetricInTrial(currentTrial, metric);
			    			if (metricObj == null) {
								JOptionPane.showMessageDialog(this, "Not all selected trials have "+ metric + " data.", "Warning!", JOptionPane.ERROR_MESSAGE);
								return;
			    			}						
						}	    				    
						trials.add(new ComparableTrial(currentTrial));				    
	    			}

		    		/**** start comparing the trials here.
		    		assume that the trials to be compared all have metric data of defaultMetric. 
		    		Otherwise, they are not comparable.
		    		****/		
			    	    
		    		PerfComparison pp = new PerfComparison(dbSession, trials, metricObj);
		    		pp.sortoutIntervalEventData();
		    		pp.displayComparisonResults();			    
				} 
	   		}
		} 
		
		else if (command.equals("IntervalEvents")) { // compare functions
		    TreePath[] selectedTreePaths = DBTree.getSelectionPaths();
		    if (selectedTreePaths == null){
				JOptionPane.showMessageDialog(this, "Please first select functions to be compared.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    } 
			// some tree nodes are selected
			if (selectedTreePaths.length == 1){
		    	JOptionPane.showMessageDialog(this, "Please select at least two functions for comparing.", "Warning!", JOptionPane.ERROR_MESSAGE);
		    	return;
			} 
	    	Vector functions = new Vector();
	    	DefaultMutableTreeNode selectedNode;
	    	String metric = null;
			Metric metricObj = null;

	    	for (int i=0; i<selectedTreePaths.length; i++) {
				selectedNode = (DefaultMutableTreeNode)selectedTreePaths[i].getLastPathComponent(); 
				if (selectedNode.isRoot()) {// PerfDMF node is selected.
		    		JOptionPane.showMessageDialog(this, "Wrong node is selected for function comparison.", "Warning!", JOptionPane.ERROR_MESSAGE);
		    		return;
				} 
	    		Wrapper nodeInfo = (Wrapper)selectedNode.getUserObject();
	    		if (nodeInfo.func == null) {
					JOptionPane.showMessageDialog(this, "Wrong node is selected for function comparison.", "Warning!", JOptionPane.ERROR_MESSAGE);				    
					return;
	    		} 
				IntervalEvent currentFunc = nodeInfo.func;
				Application nullApp = null;
				Experiment nullExp = null;					
				Trial fatherTrial;
				int numberOfMetrics, k;

				if (i==0){
	    			fatherTrial = dbSession.setTrial(currentFunc.getTrialID());
	    			String[] metricNames = buildMetricNameArray(fatherTrial);

	    			metric = (String)JOptionPane.showInputDialog(this, "Please select a COMMON counter for these functions", "Select counter", JOptionPane.PLAIN_MESSAGE, null, metricNames, metricNames[0]);
	    			if (metric == null)
						return;
				} else {
    				fatherTrial = dbSession.setTrial(currentFunc.getTrialID());
    				metricObj = findMetricInTrial(fatherTrial, metric);
    				if (metricObj == null) {
						JOptionPane.showMessageDialog(this, "Not all selected functions have "+ metric + " data.", "Warning!", JOptionPane.ERROR_MESSAGE);
						return;
    				}					    
				}
				functions.add(currentFunc);
	    	}
			    
	    	// start comparing the functions here.			    			    
	    	PerfComparison pp = new PerfComparison(dbSession, functions, metricObj);
	    	pp.sortoutIntervalEventData();
	    	pp.displayComparisonResults();			    
		}

/*
		else if (command.equals("Speedup")) {
		    TreePath[] selectedTreePaths = DBTree.getSelectionPaths();
		    if (selectedTreePaths == null){
				JOptionPane.showMessageDialog(this, "Please first select trials.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    } else if (selectedTreePaths.length == 1) {
			    JOptionPane.showMessageDialog(this, "Please select at least two trials for scalability analysis.", "Warning!", JOptionPane.ERROR_MESSAGE);
			    return;
		    }

			Vector trials = new Vector();			
			Trial tmpTrial, serialTrial=null;
			DefaultMutableTreeNode selectedNode;
			boolean hasSerialTrial = false;
			int minNodes = 500000;
			for (int i=0; i<selectedTreePaths.length; i++){
			    selectedNode = (DefaultMutableTreeNode)selectedTreePaths[i].getLastPathComponent(); 
			    if (selectedNode.isRoot()){// PerfDMF root node is selected.
					JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!", JOptionPane.ERROR_MESSAGE);
					return;
			    } 
				Wrapper nodeInfo = (Wrapper)selectedNode.getUserObject();
				if (nodeInfo.trial == null){
			    	JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!", JOptionPane.ERROR_MESSAGE);
			    	return;
				} 
		    	tmpTrial = nodeInfo.trial;
		    	if (tmpTrial.getNodeCount() * tmpTrial.getNumContextsPerNode() * tmpTrial.getNumThreadsPerContext() < minNodes){
					serialTrial = tmpTrial;
					hasSerialTrial = true;
					minNodes = tmpTrial.getNodeCount() * tmpTrial.getNumContextsPerNode() * tmpTrial.getNumThreadsPerContext();
					trials.add(new ComparableTrial(tmpTrial));
		    	} else {
					trials.add(new ComparableTrial(tmpTrial));
				}
			}

			// if (!hasSerialTrial){
			    // JOptionPane.showMessageDialog(this, "Please select a serial trial for speedup analysis.", "Warning!", JOptionPane.ERROR_MESSAGE);
			    // return;
			// }

			// assume that all the trials to be compared have metric data of defaultMetric. 
			//     Otherwise, they are not comparable.

			// sort trial according to their processor number.
			Collections.sort(trials);
			    
			double[] speedups = new double[selectedTreePaths.length];
			ListIterator tmpTrialData;
			IntervalEvent aFunc;
			int[] procList = new int[trials.size()];			
			double serialTime = 0.0;

			Application nullApp = null;
			dbSession.setApplication(nullApp);
			Experiment nullExp = null;
			dbSession.setExperiment(nullExp);
			dbSession.setTrial(serialTrial);
			dbSession.setMetric(findADefaultMetric(serialTrial, true));				    
			tmpTrialData = dbSession.getIntervalEvents();

			while (tmpTrialData.hasNext()) {
			    aFunc = (IntervalEvent) tmpTrialData.next();
				IntervalLocationProfile mainFuncMean = aFunc.getMeanSummary();
			    if (mainFuncMean.getInclusivePercentage() == 100.0) {
					serialTime = mainFuncMean.getInclusive();
					break;
			    }
			}

			for (int i=0; i<trials.size(); i++){
			    speedups[i] = 0.0;				
			    tmpTrial = ((ComparableTrial) trials.elementAt(i)).getTrial();
			    procList[i] = (tmpTrial.getNodeCount() * tmpTrial.getNumContextsPerNode() * tmpTrial.getNumThreadsPerContext());
			    dbSession.setApplication(nullApp);
			    dbSession.setExperiment(nullExp);
			    dbSession.setTrial(tmpTrial);
			    dbSession.setMetric(findADefaultMetric(tmpTrial, true));				    
			    tmpTrialData = dbSession.getIntervalEvents();
				    
			    while (tmpTrialData.hasNext()) {
					aFunc = (IntervalEvent) tmpTrialData.next();					
				    IntervalLocationProfile mainFuncMean = aFunc.getMeanSummary();
					if (mainFuncMean.getInclusivePercentage() == 100.0){
				    	speedups[i] = serialTime/mainFuncMean.getInclusive();
				    	//System.out.println(speedups[i]);
				    	break;
					}					
			    }					
			}
							    
			String mFile = WriteMatlabProgram.drawACurve(null, procList, speedups, "Speedup", "processor number", "Speedup");
			PlotDrawingPanel plotPanel = new PlotDrawingPanel(procList, speedups, "Speedup", "processor number", "Speedup", null);
			final PlotDrawingWindow plotWin = new PlotDrawingWindow("Speedup window");
			plotWin.getContentPane().add(plotPanel);
			plotWin.setVisible(true);	
			plotWin.setResizable(false);			    
		}
		*/

		else if (command.equals("Execution time") || command.equals("Speedup")) {
		    TreePath[] selectedTreePaths = DBTree.getSelectionPaths();
		    if (selectedTreePaths == null) {
				JOptionPane.showMessageDialog(this, "Please first select trials.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    } else if (selectedTreePaths.length == 1){
			    JOptionPane.showMessageDialog(this, "Please select at least two trials for scalability analysis.", "Warning!", JOptionPane.ERROR_MESSAGE);
			    return;
		    }

			Vector trials = new Vector();			
			Trial tmpTrial;
			DefaultMutableTreeNode selectedNode;
			for (int i=0; i<selectedTreePaths.length; i++) {
			    selectedNode = (DefaultMutableTreeNode)selectedTreePaths[i].getLastPathComponent(); 
			    if (selectedNode.isRoot()){// PerfDMF root node is selected.
					JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!", JOptionPane.ERROR_MESSAGE);
					return;
			    }
				Wrapper nodeInfo = (Wrapper)selectedNode.getUserObject();
				if (nodeInfo.trial == null){
				    JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!", JOptionPane.ERROR_MESSAGE);
				    return;
				}
			    tmpTrial = nodeInfo.trial;
			    trials.add(new ComparableTrial(tmpTrial));
			}		
			
			/****assume that all the trials to be compared have metric data of defaultMetric. 
			       Otherwise, they are not comparable.
			****/
			// sort trial according to their process number.
			Collections.sort(trials);
			
			double[] times = new double[selectedTreePaths.length];
			double[] times2 = new double[selectedTreePaths.length];
			double[] times3 = new double[selectedTreePaths.length];
			int[] procList = new int[trials.size()];
			ListIterator tmpTrialData;
			IntervalEvent aFunc;
			int numberOfMetrics;
			Application nullApp = null;
			Experiment nullExp = null;
			String mFile;
			
			for (int i=0; i<selectedTreePaths.length; i++){
				times[i] = 0.0;		
				tmpTrial = ((ComparableTrial) trials.elementAt(i)).getTrial();		
				procList[i] = (tmpTrial.getNodeCount() * tmpTrial.getNumContextsPerNode() * tmpTrial.getNumThreadsPerContext());
			}

			// choose a proper granularity of investigation.
			String[] options = {"trial","group","function"};
								   
			String grain = (String)JOptionPane.showInputDialog(this,
											   "Please select a granularity of interest.",
											   "Select granularity",
											   JOptionPane.PLAIN_MESSAGE,
											   null,
											   options,
											   options[0]);
			if (grain == null)
				   return; 		
			else if (grain.equals("trial")) { // show scalability metric at trial level.
				boolean firstTrial = true;
				double baseline = 0.0;
				for (int i=0; i<selectedTreePaths.length; i++) {
					tmpTrial = ((ComparableTrial) trials.elementAt(i)).getTrial();
					dbSession.setApplication(nullApp);
					dbSession.setExperiment(nullExp);
					dbSession.setTrial(tmpTrial);
					dbSession.setMetric(findADefaultMetric(tmpTrial, true));				    
					tmpTrialData = dbSession.getIntervalEvents();

					while (tmpTrialData.hasNext()){
						aFunc = (IntervalEvent) tmpTrialData.next();					
						IntervalLocationProfile mainFuncMean = aFunc.getMeanSummary();
						if (mainFuncMean.getInclusivePercentage() == 100.0){
							if (firstTrial) {
								baseline = mainFuncMean.getInclusive();
								firstTrial = false;
							}
							if (command.equals("Execution time")) {
								times[i] = mainFuncMean.getInclusive();
							} else {
								times[i] = baseline / mainFuncMean.getInclusive();
							}
							break;
						}
					}					
				}	


				//write matlab command.
				//mFile = WriteMatlabProgram.drawACurve(null, procList, times, "Execution time", "processor number", "Execution time (usec)");
				
				PlotDrawingPanel plotPanel = new PlotDrawingPanel(procList, times, command, "processor number", command, null);
				
				final PlotDrawingWindow plotWin = new PlotDrawingWindow(command + " window");
				plotWin.getContentPane().add(plotPanel);
				plotWin.setVisible(true);	
				plotWin.setResizable(false);				
							
			}

			else if (grain.equals("group")) { // show scalability metric at function group level.
				
				tmpTrial = ((ComparableTrial) trials.elementAt(0)).getTrial();
				Metric metricObj = findADefaultMetric(tmpTrial, true);
				//	let user select a group 
				PerfComparison pp = new PerfComparison(dbSession, trials, metricObj);					
				pp.classifyIntervalEvents(false);																
				Hashtable groupHT = pp.getMeanGroupValues(null);				
				String[] groupNames = new String[groupHT.size()];
				int counter = 0;
				for(Enumeration e1 = groupHT.keys(); e1.hasMoreElements();){
					 groupNames[counter++] = (String) e1.nextElement();
				}			
				
				String group = (String)JOptionPane.showInputDialog(this,
							"Please select a group",
							"Select group",
							JOptionPane.PLAIN_MESSAGE,
							null,
							groupNames,
							groupNames[0]);

				if (group==null)
					return;
				
				String[] metricList = {"exclusive time", "exclusive percentage", "inclusive time", "inclusive percentage"};
				String metric = (String)JOptionPane.showInputDialog(this,
								"Please select a metric",
								"Select metric",
								JOptionPane.PLAIN_MESSAGE,
								null,
								metricList,
								metricList[0]);
				
				if (metric == null)
					return;
					
				groupHT = pp.getMeanGroupValues(group);
				
				Enumeration e1 = groupHT.keys();
				String strKey;
				Vector valueVec;
				while (e1.hasMoreElements()) {
					// get a function name.
					strKey = (String) e1.nextElement();
					//System.out.println(strKey);
					
					// get function values.
					int k=0;
					ComparisonWindowIntervalEvent tmpFunc;
					valueVec = (Vector) groupHT.get(strKey); 
					for (int j=0; j < valueVec.size(); j++){
						if (valueVec.elementAt(j) != null){
							tmpFunc = (ComparisonWindowIntervalEvent) valueVec.elementAt(j);
							while (((ComparableTrial) trials.elementAt(k)).getTrial().getID() != tmpFunc.getTrialID())
								k++;
							
							if (metric.equals("exclusive time"))
								times[k] += tmpFunc.getExclusive();
							else if (metric.equals("exclusive percentage"))
								times[k] += tmpFunc.getExclusivePercentage();	
							else if (metric.equals("inclusive time"))
								times[k] += tmpFunc.getInclusive();	
							else 
								times[k] += tmpFunc.getInclusivePercentage();
						}
					}
				}
						
				PlotDrawingPanel plotPanel = new PlotDrawingPanel(procList, times, group+" "+metric, "processor number", group+" "+metric, null);
				
				final PlotDrawingWindow plotWin = new PlotDrawingWindow("Execution time window");
				plotWin.getContentPane().add(plotPanel);
				plotWin.setVisible(true);	
				plotWin.setResizable(false);		
						
			}

			else { // show scalability metric at function level.
				tmpTrial = ((ComparableTrial) trials.elementAt(0)).getTrial();
				dbSession.setApplication(nullApp);
				dbSession.setExperiment(nullExp);
				dbSession.setTrial(tmpTrial);
				dbSession.setMetric(findADefaultMetric(tmpTrial, true));				    
				DataSessionIterator functions = (DataSessionIterator) dbSession.getIntervalEvents();
				    				    
				int counter = 0;
				String[] functionNames = new String[functions.size()];    
				while (functions.hasNext()){
					aFunc = (IntervalEvent) functions.next();	
					functionNames[counter++] = aFunc.getName();
				}
				
				String selectedIntervalEvent = (String)JOptionPane.showInputDialog(this,
											"Please select a function",
											"Select function",
											JOptionPane.PLAIN_MESSAGE,
											null,
											functionNames,
											functionNames[0]);
				
				if (selectedIntervalEvent == null)
					return;											
				
				String[] metricList = {"exclusive time", "exclusive percentage", "inclusive time", "inclusive percentage"};
				String metric = (String)JOptionPane.showInputDialog(this,
								"Please select a metric",
								"Select metric",
								JOptionPane.PLAIN_MESSAGE,
								null,
								metricList,
								metricList[0]);
				if (metric == null)
					return;

				// new code
				Vector comparedTrials = new Vector();
				
				for (int i=0; i<selectedTreePaths.length; i++){
					tmpTrial = ((ComparableTrial) trials.elementAt(i)).getTrial();
					comparedTrials.add(tmpTrial);
				}			

				Scalability comparison = new Scalability((PerfDMFSession)dbSession);
				ScalabilityResults results = null;

				if (metric.equals("exclusive time")) {
					results = comparison.exclusive(comparedTrials, selectedIntervalEvent);
				} else if (metric.equals("exclusive percentage"))	 {
					results = comparison.exclusivePercentage(comparedTrials, selectedIntervalEvent);
				} else if (metric.equals("inclusive time")) {
					results = comparison.inclusive(comparedTrials, selectedIntervalEvent);
				} else {
					results = comparison.inclusivePercentage(comparedTrials, selectedIntervalEvent);
				}

				if (command.equals("Execution time")) {
					times = results.getAverageData(selectedIntervalEvent, false);
					times2 = results.getMaximumData(selectedIntervalEvent, false);
					times3 = results.getMinimumData(selectedIntervalEvent, false);
 				} else {// command.equals("Speedup") 
					times = results.getAverageData(selectedIntervalEvent, true);
					times2 = results.getMaximumData(selectedIntervalEvent, true);
					times3 = results.getMinimumData(selectedIntervalEvent, true);
				}

				// old code
/*
				for (int i=0; i<selectedTreePaths.length; i++){
					tmpTrial = ((ComparableTrial) trials.elementAt(i)).getTrial();
					dbSession.setApplication(nullApp);
					dbSession.setExperiment(nullExp);
					dbSession.setTrial(tmpTrial);
					dbSession.setMetric(findADefaultMetric(tmpTrial, true));				    
					tmpTrialData = dbSession.getIntervalEvents();
				    
					while (tmpTrialData.hasNext()){
						aFunc = (IntervalEvent) tmpTrialData.next();					
						if (aFunc.getName().equals(selectedIntervalEvent)){				
							IntervalLocationProfile funcMean = aFunc.getMeanSummary();
							if (metric.equals("exclusive time"))
								times[i] = funcMean.getExclusive();
							else if (metric.equals("exclusive percentage"))	
								times[i] = funcMean.getExclusivePercentage();
							else if (metric.equals("inclusive time"))
								times[i] = funcMean.getInclusive();
							else
								times[i] = funcMean.getInclusivePercentage();
								
							break;
						}					
					}		
				}		
*/
				
				// draw the plot
				// PlotDrawingPanel plotPanel = new PlotDrawingPanel(procList, times, selectedIntervalEvent+" "+metric, "processor number", selectedIntervalEvent+" "+metric, null);
				PlotDrawingPanel2 plotPanel = new PlotDrawingPanel2(procList, times, times2, times3, selectedIntervalEvent+" "+metric, "processor number", selectedIntervalEvent+" "+metric, null);
				String windowName = null;
				if (command.equals("Execution time")) {
					windowName = new String("Execution time window");
				} else {
					windowName = new String("Speedup window");
				}
				final PlotDrawingWindow plotWin = new PlotDrawingWindow(windowName);
				plotWin.getContentPane().add(plotPanel);
				plotWin.setVisible(true);	
				plotWin.setResizable(false);
			}
		}

		else if (command.equals("MPI Communication time")) {
		    TreePath[] selectedTreePaths = DBTree.getSelectionPaths();
		    if (selectedTreePaths == null){
				JOptionPane.showMessageDialog(this, "Please first select trials.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    } else if (selectedTreePaths.length == 1){
			    JOptionPane.showMessageDialog(this, "Please select at least two trials for scalability analysis.", "Warning!", JOptionPane.ERROR_MESSAGE);
			    return;
		    }

			Vector trials = new Vector();			
			Trial tmpTrial;
			DefaultMutableTreeNode selectedNode;
			for (int i=0; i<selectedTreePaths.length; i++){
			    selectedNode = (DefaultMutableTreeNode)selectedTreePaths[i].getLastPathComponent(); 
			    if (selectedNode.isRoot()){// PerfDMF root node is selected.
					JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!", JOptionPane.ERROR_MESSAGE);
					return;
			    }
				Wrapper nodeInfo = (Wrapper)selectedNode.getUserObject();
				if (nodeInfo.trial == null){
				    JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!", JOptionPane.ERROR_MESSAGE);
				    return;
				}
				tmpTrial = nodeInfo.trial;
				trials.add(new ComparableTrial(tmpTrial));
			}		

			/****assume that all the trials to be compared have metric data of defaultMetric. 
			       Otherwise, they are not comparable.
			****/
			// sort trial according to their process number.
			Collections.sort(trials);
			    
			double[] times = new double[selectedTreePaths.length];
			ListIterator tmpTrialData;
			IntervalEvent aFunc;
			int[] procList = new int[trials.size()];
			
			Application nullApp = null;
			Experiment nullExp = null;
			for (int i=0; i<selectedTreePaths.length; i++){
			    times[i] = 0.0;				
			    tmpTrial = ((ComparableTrial) trials.elementAt(i)).getTrial();
			    procList[i] = (tmpTrial.getNodeCount() * tmpTrial.getNumContextsPerNode() * tmpTrial.getNumThreadsPerContext());
			    
			    dbSession.setApplication(nullApp);
			    dbSession.setExperiment(nullExp);
			    dbSession.setTrial(tmpTrial);
			    dbSession.setMetric(findADefaultMetric(tmpTrial, true));				    
			    tmpTrialData = dbSession.getIntervalEvents();
				    
			    while (tmpTrialData.hasNext()){
					aFunc = (IntervalEvent) tmpTrialData.next();					
					if (aFunc.getName().startsWith("MPI_")){				
				    	IntervalLocationProfile mainFuncMean = aFunc.getMeanSummary();
				    	times[i] += mainFuncMean.getExclusive();				    
					}					
			    }					
			    
			    //System.out.println(times[i]);
			}
							    
			PlotDrawingPanel plotPanel = new PlotDrawingPanel(procList, times, "Mean MPI Communication Time", "processor number", "Mean MPI Communication Time(usec)", null);
				
			final PlotDrawingWindow plotWin = new PlotDrawingWindow("MPI Communication Time Window");
			plotWin.getContentPane().add(plotPanel);
			plotWin.setVisible(true);	
			plotWin.setResizable(false);    			    
		}

		else if (command.equals("show mean statistics") || command.equals("show total statistics")) {
		    TreePath[] selectedTreePaths = DBTree.getSelectionPaths();
		    if (selectedTreePaths == null){
				JOptionPane.showMessageDialog(this, "Please first select a trial or function.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    } else if (selectedTreePaths.length > 1){
				JOptionPane.showMessageDialog(this, "Please select only one trial or function .", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    }
		    // some tree nodes are selected
			DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode)selectedTreePaths[0].getLastPathComponent(); 
			if (selectedNode.isRoot()){// PerfDMF node is selected.
			    JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!",
							  JOptionPane.ERROR_MESSAGE);
			    return;
			}
			Wrapper nodeInfo = (Wrapper)selectedNode.getUserObject();
			if (nodeInfo.trial != null) {
				Trial selectedTrial = nodeInfo.trial;
				// reset perfdbsession
				dbSession.reset();
				dbSession.setTrial(selectedTrial);								
				IntervalEvent nullFunc = null;
				dbSession.setIntervalEvent(nullFunc);

				String[] metricNames = buildMetricNameArray(selectedTrial);

				String s = (String)JOptionPane.showInputDialog(this,
							       "Please select a counter",
							       "Select counter",
							       JOptionPane.PLAIN_MESSAGE,
							       null,
							       metricNames,
							       metricNames[0]);
				if (s == null)
				    return;

				dbSession.setMetric(findMetricInTrial(selectedTrial, s));
				ListIterator functions = dbSession.getIntervalEvents();
				String[] profColumnNames = {"IntervalEvent-name", "inclusive%", "inclusive", "exclusive%", "exclusive", "#Call", "#Subrs", "Inclusive/Call"};
					
				Vector rowsVec = new Vector();
				IntervalEvent aFunc;
				IntervalLocationProfile aFuncObject;

				while (functions.hasNext()){
				    aFunc = (IntervalEvent) functions.next();
				    if (command.equals("show mean statistics"))
						aFuncObject = aFunc.getMeanSummary();
				    else 
						aFuncObject = aFunc.getTotalSummary();
				    Vector aRow = new Vector();					    					    
				    aRow.add(aFunc.getName());
				    aRow.add(new Double(aFuncObject.getInclusivePercentage()));
				    aRow.add(new Double(aFuncObject.getInclusive()));
				    aRow.add(new Double(aFuncObject.getExclusivePercentage()));
				    aRow.add(new Double(aFuncObject.getExclusive()));
				    aRow.add(new Integer(aFuncObject.getNumCalls()));
				    aRow.add(new Integer(aFuncObject.getNumSubroutines()));
				    aRow.add(new Double(aFuncObject.getInclusivePerCall()));
				    rowsVec.add(aRow);
				}
										
				DBContent profData = new DBContent(rowsVec, profColumnNames);
				profTable.setModel(profData);
				profData.addMouseListenerToTableHeader(profTable);
				profTable.getColumnModel().getColumn(0).setPreferredWidth(300);
						
				profLabel.setVisible(true);
				if (command.equals("show mean statistics"))
				    profLabel.setText("Mean summary ( " + s +" ) for the trial");
				else 
				    profLabel.setText("Total summary ( " + s +" ) for the trial");
				
			} else if (nodeInfo.func != null){
				IntervalEvent selectedFunc = nodeInfo.func;				
				Application nullApp = null;
				Experiment nullExp = null;
				dbSession.setApplication(nullApp);
				dbSession.setExperiment(nullExp);
				Trial parentTrial = dbSession.setTrial(selectedFunc.getTrialID());
				String[] metricNames = buildMetricNameArray(parentTrial);
				
				String s = (String)JOptionPane.showInputDialog(this,
							       "Please select a counter",
							       "Select counter",
							       JOptionPane.PLAIN_MESSAGE,
							       null,
							       metricNames,
							       metricNames[0]);
				if (s == null)
				    return;

				dbSession.setMetric(findMetricInTrial(parentTrial, s));
				IntervalLocationProfile funcData; 

				if (command.equals("show mean statistics"))    
				    funcData = dbSession.setIntervalEvent(selectedFunc.getID()).getMeanSummary();
				else 
				    funcData = dbSession.setIntervalEvent(selectedFunc.getID()).getTotalSummary();
								
				String[] profColumns = {"inclusive%", "inclusive", "exclusive%", "exclusive", "#Call", "#Subroutines", "Inclusive/Call"};
								
				Object[][] profRows = {{new Double(funcData.getInclusivePercentage()), new Double(funcData.getInclusive()), new Double(funcData.getExclusivePercentage()), new Double(funcData.getExclusive()), new Integer(funcData.getNumCalls()), new Integer(funcData.getNumSubroutines()), new Double(funcData.getInclusivePerCall()) }};

				DBContent profData = new DBContent(profRows, profColumns);
				profTable.setModel(profData);
				profData.addMouseListenerToTableHeader(profTable);

				// resize columns
				for (int i=0; i<7; i++)
				    profTable.getColumnModel().getColumn(i).setPreferredWidth(100);
					
				if (command.equals("show mean statistics"))
				    profLabel.setText("Mean summary ( "+ s +" ) for "+selectedFunc.getName());
				else
				    profLabel.setText("Total summary ( "+ s +" ) for "+selectedFunc.getName());
				profLabel.setVisible(true);
				
			} else {
				JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
			}    
		}				

		else if (command.equals("show user-defined events")){
		    TreePath[] selectedTreePaths = DBTree.getSelectionPaths();
		    if (selectedTreePaths == null){
				JOptionPane.showMessageDialog(this, "Please first select a trial.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    }
		    else if (selectedTreePaths.length > 1){
				JOptionPane.showMessageDialog(this, "Please select only one trial .", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    }

		    // some tree nodes are selected
			DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode)selectedTreePaths[0].getLastPathComponent(); 
			if (selectedNode.isRoot()){// PerfDMF node is selected.
			    JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!",
							  JOptionPane.ERROR_MESSAGE);
			    return;
			}

			Wrapper nodeInfo = (Wrapper)selectedNode.getUserObject();
			if (nodeInfo.trial == null){
				JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
			}

			Trial selectedTrial = nodeInfo.trial;
			Application nullApp = null;
			Experiment nullExp = null;
			dbSession.setApplication(nullApp);
			dbSession.setExperiment(nullExp);
			dbSession.setTrial(selectedTrial);
	
			IntervalEvent nullFunc = null;
			dbSession.setIntervalEvent(nullFunc);
			ListIterator atomicEvents = dbSession.getAtomicEventData();
				
			// reset pprof table
			String[] profColumnNames = {"Event name", "Node", "Context", "Thread","Number of samples", "Maximum value", "Minimum value", "Mean value", "Standard deviation"};
					
			Vector rowsVec = new Vector();
			AtomicLocationProfile aUEData;
			AtomicEvent aUE;

			while (atomicEvents.hasNext()){
			    aUEData = (AtomicLocationProfile) atomicEvents.next();
			    Vector aRow = new Vector();				
			    aUE = dbSession.getAtomicEvent(aUEData.getAtomicEventID());
			    aRow.add(aUE.getName());
			    aRow.add(new Integer(aUEData.getNode()));
			    aRow.add(new Integer(aUEData.getContext()));
			    aRow.add(new Integer(aUEData.getThread()));
			    aRow.add(new Integer(aUEData.getSampleCount()));
			    aRow.add(new Double(aUEData.getMaximumValue()));
			    aRow.add(new Double(aUEData.getMinimumValue()));
			    aRow.add(new Double(aUEData.getMeanValue()));
			    aRow.add(new Double(aUEData.getStandardDeviation()));
			    rowsVec.add(aRow);
			}
				
			DBContent profData = new DBContent(rowsVec, profColumnNames);
			profTable.setModel(profData);
			profData.addMouseListenerToTableHeader(profTable);
			profTable.getColumnModel().getColumn(0).setPreferredWidth(200);

			profLabel.setVisible(true);
			profLabel.setText("User event performance data for the trial"); 
		}

		else if (command.equals("show counter")){
		    TreePath[] selectedTreePaths = DBTree.getSelectionPaths();
		    if (selectedTreePaths == null){
				JOptionPane.showMessageDialog(this, "Please first select a trial/function.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    } 
			if (selectedTreePaths.length > 1){
				JOptionPane.showMessageDialog(this, "Please select only one trial or function.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    }

		    // some tree nodes are selected
			DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode)selectedTreePaths[0].getLastPathComponent(); 
			if (selectedNode.isRoot()){// PerfDMF node is selected.
			    JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!", JOptionPane.ERROR_MESSAGE);
			    return;
			}

			Wrapper nodeInfo = (Wrapper)selectedNode.getUserObject();
		    if (nodeInfo.trial != null){
				Trial selectedTrial = nodeInfo.trial;

				String[] metricNames = buildMetricNameArray(selectedTrial);
				String s = (String)JOptionPane.showInputDialog(this,
							       "Please select a counter",
							       "Select counter",
							       JOptionPane.PLAIN_MESSAGE,
							       null,
							       metricNames,
							       metricNames[0]);
				if (s == null)
				    return;

				Application nullApp = null;
				Experiment nullExp  =null;
				dbSession.setApplication(nullApp);
				dbSession.setExperiment(nullExp);
				dbSession.setTrial(selectedTrial);
				IntervalEvent nullFunc = null;
				dbSession.setIntervalEvent(nullFunc);
				dbSession.setMetric(findMetricInTrial(selectedTrial, s));

				ListIterator funcData = dbSession.getIntervalEventData();
				
				String[] profColumnNames = {"IntervalEvent-name", "Node", "Context", "Thread","inclusive%", "inclusive", "exclusive%", "exclusive", "#Calls", "#Subroutines", "Inclusive/Call"};
					
				Vector rowsVec = new Vector();
				IntervalLocationProfile aFunc;
				IntervalEventProfile aFuncObject;

				while (funcData.hasNext()){
				    aFunc = (IntervalLocationProfile)funcData.next();
				    aFuncObject = new IntervalEventProfile(dbSession, aFunc);
				    Vector aRow = new Vector();					    					    
				    aRow.add(aFuncObject.getName());
				    aRow.add(new Integer(aFuncObject.getNode()));
				    aRow.add(new Integer(aFuncObject.getContext()));
				    aRow.add(new Integer(aFuncObject.getThread()));
				    aRow.add(new Double(aFuncObject.getInclusivePercentage()));
				    aRow.add(new Double(aFuncObject.getInclusive()));
				    aRow.add(new Double(aFuncObject.getExclusivePercentage()));
				    aRow.add(new Double(aFuncObject.getExclusive()));
				    aRow.add(new Integer(aFuncObject.getNumCalls()));
				    aRow.add(new Integer(aFuncObject.getNumSubroutines()));
				    aRow.add(new Double(aFuncObject.getInclusivePerCall()));
				    rowsVec.add(aRow);
				}								
				
				DBContent profData = new DBContent(rowsVec, profColumnNames);
				profTable.setModel(profData);
				profData.addMouseListenerToTableHeader(profTable);
				profTable.getColumnModel().getColumn(0).setPreferredWidth(300);
						
				profLabel.setVisible(true);
				profLabel.setText(s + " for the trial");			
			} 
			
			else if (nodeInfo.func != null){
				IntervalEvent selectedFunc = nodeInfo.func;
				
				Application nullApp = null;
				Experiment nullExp = null;
				dbSession.setApplication(nullApp);
				dbSession.setExperiment(nullExp);
				
				Trial parentTrial = dbSession.setTrial(selectedFunc.getTrialID());
				String[] metricNames = buildMetricNameArray(parentTrial);
				
				String s = (String)JOptionPane.showInputDialog(this,
							       "Please select a counter",
							       "Select counter",
							       JOptionPane.PLAIN_MESSAGE,
							       null,
							       metricNames,
							       metricNames[0]);
				if (s == null)
				    return;

				dbSession.setMetric(findMetricInTrial(parentTrial, s));
				dbSession.setIntervalEvent(selectedFunc);
					
				ListIterator intervalEventData = dbSession.getIntervalEventData();
				String[] profColumns = {"Node", "Context", "Thread", "inclusive%", "inclusive", "exclusive%", "exclusive", "#Call", "#Subroutines", "Inclusive/Call"};
								
				Vector profRows = new Vector();
				IntervalEventProfile aFunc;
				while (intervalEventData.hasNext()){
				    aFunc = new IntervalEventProfile(dbSession, (IntervalLocationProfile)intervalEventData.next());
				    Vector oneRow = new Vector();
				    //oneRow.add(aFunc.getName());
				    oneRow.add(new Integer(aFunc.getNode()));
				    oneRow.add(new Integer(aFunc.getContext()));
				    oneRow.add(new Integer(aFunc.getThread()));
				    oneRow.add(new Double(aFunc.getInclusivePercentage()));
				    oneRow.add(new Double(aFunc.getInclusive()));
				    oneRow.add(new Double(aFunc.getExclusivePercentage()));
				    oneRow.add(new Double(aFunc.getExclusive()));
				    oneRow.add(new Integer(aFunc.getNumCalls()));
				    oneRow.add(new Integer(aFunc.getNumSubroutines()));
				    oneRow.add(new Double(aFunc.getInclusivePerCall()));
				    profRows.add(oneRow);
				} 

				DBContent profData = new DBContent(profRows, profColumns);
				profTable.setModel(profData);
				profData.addMouseListenerToTableHeader(profTable);	
				profLabel.setText("Performance data ( "+ s +" ) for "+selectedFunc.getName());
				profLabel.setVisible(true);
				
			}
		    else {
				JOptionPane.showMessageDialog(this, "Wrong node is selected.", "Warning!", JOptionPane.ERROR_MESSAGE);
				return;
		    }    				
		}		
	    }
	}

    public static String usec2String(double usec){
	int hours, minutes;
	double seconds;

	seconds = usec/1000000;
	hours = (int) seconds / 3600;
	seconds = seconds - (hours * 3600);
	minutes = (int) seconds / 60;
	seconds = seconds - (minutes * 60);
	
	return hours+":"+minutes+":"+seconds;
    }

	// find a metric to show when a user clicks on a trial, should be execuion time or whatever close. If there is no execution time data
	// for this trial, then grab the first in the trial's metric list.
	private Metric findADefaultMetric(Trial aTrial, boolean defaultOnly){
		DataSessionIterator metrics = new DataSessionIterator(aTrial.getMetrics());
		Metric m = null;
		while (metrics.hasNext()) {
			m = (Metric)metrics.next();
			if (defaultOnly && m.getName().equals(defaultMetric)) {						    
				return m;
			} else if (m.getName().equals("GET_TIME_OF_DAY") || m.getName().equals("SGI_TIMERS")
				|| m.getName().equals("LINUX_TIMERS") || m.getName().equals("CPU_TIME")
				|| m.getName().equals("P_WALL_CLOCK_TIME") || m.getName().equals("P_VIRTUAL_TIME")
				|| m.getName().equals(defaultMetric)) {						    
				return m;
			}
		}
		// otherwise, just return the first one.
		metrics.reset();
		m = (Metric)metrics.next();
		return m;
	}

	private String[] buildMetricNameArray(Trial aTrial){
		int count = aTrial.getMetricCount();
		String[] nameArray = new String[count];
		DataSessionIterator metrics = new DataSessionIterator(aTrial.getMetrics());
		Metric m = null;
		int i = 0;
		while (metrics.hasNext()) {
			m = (Metric)metrics.next();
			nameArray[i++] = m.getName();
		}
		return nameArray;
	}
  
  	private Metric findMetricInTrial(Trial aTrial, String name) {
		DataSessionIterator metrics = new DataSessionIterator(aTrial.getMetrics());
		Metric m = null;
		while (metrics.hasNext()) {
			m = (Metric)metrics.next();
			if (m.getName().equals(name))
				return m;
		}
		m = null;
		return m;
	}

    // This class is for generating name for application/experiment/trial node through toString method.
    public class Wrapper{
	private String myString;
	public Application app = null;
	public Experiment exp = null;
	public Trial trial = null;
	public IntervalEvent func = null;

	public Wrapper(Application a){
	    this.app = a;
	    myString = a.getName()+a.getVersion();
	}

	public Wrapper(Experiment e){
	    this.exp = e;
		if (e.getName() == null || e.getName().equals(""))
	    	myString = "Experiment" + e.getID();
		else
	    	myString = e.getName();
	}

	public Wrapper(Trial t){
	    this.trial = t;
		if (t.getName() == null || t.getName().equals(""))
	    	myString = "Trial" + t.getID();
		else
	    	myString = t.getName();
	}

	public Wrapper(IntervalEvent f){
	    this.func = f;
	    myString = f.getName();
	}

	public String toString(){
	    return myString;
	}	
    }
	
	
	//	load-application dialog
	 public class AppLoadDialog extends JDialog implements ActionListener{
	 private Frame parentFrame;	 
	 final private JTextField fileNameField;
	
	 final JFileChooser fc = new JFileChooser();

	 private JButton okButton = new JButton("Ok");
	 private JButton cancelButton = new JButton("Cancel");
		 
	 public AppLoadDialog(Frame parentFrame){
	    
		 super(parentFrame, "Load Application");
		 this.parentFrame = parentFrame;		 	    
		 fileNameField = new JTextField(14);
	    
		 JLabel fileNameLabel = new JLabel("Application information file:");
	    
		 JButton fileBrowseButton = new JButton("Browse");
		 fileBrowseButton.setMnemonic(102);
		
		 Container contentPane = getContentPane();
		 GridBagLayout gridbag = new GridBagLayout();	    
		 contentPane.setLayout(gridbag);

		 JLabel mesg = new JLabel("Please provide following file name.");

		 contentPane.add(mesg, new GridBagConstraints(0, 0, 1, 1, 0.0, 0.0,
					 GridBagConstraints.CENTER,
					 GridBagConstraints.HORIZONTAL,
					 new Insets(20, 30, 10, 0), 0, 0));
	    
		 contentPane.add(fileNameLabel,    new GridBagConstraints(0, 1, 1, 1, 0.0, 0.0,
					 GridBagConstraints.WEST,
					 GridBagConstraints.HORIZONTAL,
					 new Insets(0, 20, 10, 0), 0, 0));
	    
		 contentPane.add(fileNameField,    new GridBagConstraints(1, 1, 2, 1, 0.0, 0.0,
					 GridBagConstraints.WEST,
					 GridBagConstraints.HORIZONTAL,
					 new Insets(0, 0, 10, 20), 0, 0));     

		 contentPane.add(fileBrowseButton, new GridBagConstraints(3, 1, 1, 1, 0.0, 0.0,
									 GridBagConstraints.WEST,
									 GridBagConstraints.HORIZONTAL,
									 new Insets(0, 0, 10, 20), 0, 0));
		 fileBrowseButton.addActionListener(this);		

		 contentPane.add(okButton,   new GridBagConstraints(1, 2, 1, 1, 0.0, 0.0,
					 GridBagConstraints.SOUTH,
					 GridBagConstraints.HORIZONTAL,
					 new Insets(30, 0, 10, 0), 20, 0));

		 contentPane.add(cancelButton,   new GridBagConstraints(2, 2, 1, 1, 0.0, 0.0,
					 GridBagConstraints.SOUTH,
					 GridBagConstraints.HORIZONTAL,
				 new Insets(30, 5, 10, 20), 15, 0));
	    
		 okButton.addActionListener(this);

		 cancelButton.addActionListener(this);

		 // Set the size and location
		 this.pack();
	    	    
		 //Center this dialog on the screen.
		 Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
		 int x = ( screen.width / 2) - (this.getSize().width /2 );
		 int y = ( screen.height / 2) - (this.getSize().height /2 );
		 setLocation(x,y);

		 // Show us.
		 this.show();	    	    	    
		 this.setResizable(false);
	 }

	 public void actionPerformed(ActionEvent evt) {
		 Object eventObj = evt.getSource();
      
		 if (eventObj instanceof JButton){ // if this event is fired by a button.
			  String command = evt.getActionCommand();
			 if (command.equals("Ok")){
				 // pass to main window the file names.
		    
				 ((MainWindow) parentFrame).loadApplication(fileNameField.getText());
				 this.dispose();
			 }
			 else if (command.equals("Cancel")){
				 this.dispose();
			 }
			 else if (command.equals("Browse")){
				 int returnVal = fc.showOpenDialog(this);
				 if (returnVal == JFileChooser.APPROVE_OPTION) {
					 File file = fc.getSelectedFile();
					 if (((JButton) eventObj).getMnemonic() == 102) {						 
						  fileNameField.setText(file.getAbsolutePath());
					 }			
						
					 //System.out.println(file.getAbsolutePath());
				 } 
				 else {
					 //	log.append("Open command cancelled by user." + newline);
				 }
			 }
		 }
	 }
	 }
	
	// load-experiment dialog
    public class ExpLoadDialog extends JDialog implements ActionListener{
	private Frame parentFrame;
	private DefaultMutableTreeNode parentAppNode;
	final private JTextField fileNameField;
	
	final JFileChooser fc = new JFileChooser();

	private JButton okButton = new JButton("Ok");
    private JButton cancelButton = new JButton("Cancel");
	
	// loading experiment dialog
	// selectedAppNode provides the application id, which is necessary when loadding an experiment.
	public ExpLoadDialog(DefaultMutableTreeNode selectedAppNode, Frame parentFrame){
	    
	    super(parentFrame, "Load Experiment");
	    this.parentFrame = parentFrame;
	    this.parentAppNode = selectedAppNode;
	    
	    fileNameField = new JTextField(14);
	    
	    JLabel fileNameLabel = new JLabel("Experiment information file:");
	    
		JButton fileBrowseButton = new JButton("Browse");
		fileBrowseButton.setMnemonic(102);
		
	    Container contentPane = getContentPane();
	    GridBagLayout gridbag = new GridBagLayout();	    
	    contentPane.setLayout(gridbag);

	    JLabel mesg = new JLabel("Please provide following file name.");

	    contentPane.add(mesg, new GridBagConstraints(0, 0, 1, 1, 0.0, 0.0,
                    GridBagConstraints.CENTER,
                    GridBagConstraints.HORIZONTAL,
                    new Insets(20, 30, 10, 0), 0, 0));
	    
	    contentPane.add(fileNameLabel,    new GridBagConstraints(0, 1, 1, 1, 0.0, 0.0,
                    GridBagConstraints.WEST,
                    GridBagConstraints.HORIZONTAL,
                    new Insets(0, 20, 10, 0), 0, 0));
	    
	    contentPane.add(fileNameField,    new GridBagConstraints(1, 1, 2, 1, 0.0, 0.0,
                    GridBagConstraints.WEST,
                    GridBagConstraints.HORIZONTAL,
                    new Insets(0, 0, 10, 20), 0, 0));     

		contentPane.add(fileBrowseButton, new GridBagConstraints(3, 1, 1, 1, 0.0, 0.0,
									GridBagConstraints.WEST,
									GridBagConstraints.HORIZONTAL,
									new Insets(0, 0, 10, 20), 0, 0));
		fileBrowseButton.addActionListener(this);		

	    contentPane.add(okButton,   new GridBagConstraints(1, 2, 1, 1, 0.0, 0.0,
                    GridBagConstraints.SOUTH,
                    GridBagConstraints.HORIZONTAL,
                    new Insets(30, 0, 10, 0), 20, 0));

	    contentPane.add(cancelButton,   new GridBagConstraints(2, 2, 1, 1, 0.0, 0.0,
                    GridBagConstraints.SOUTH,
                    GridBagConstraints.HORIZONTAL,
        	    new Insets(30, 5, 10, 20), 15, 0));
	    
	    okButton.addActionListener(this);

	    cancelButton.addActionListener(this);

	    // Set the size and location
	    this.pack();
	    	    
	    //Center this dialog on the screen.
	    Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
	    int x = ( screen.width / 2) - (this.getSize().width /2 );
	    int y = ( screen.height / 2) - (this.getSize().height /2 );
	    setLocation(x,y);

	    // Show us.
	    this.show();	    	    	    
	    this.setResizable(false);
	}

	public void actionPerformed(ActionEvent evt) {
	    Object eventObj = evt.getSource();
      
	    if (eventObj instanceof JButton){ // if this event is fired by a button.
			 String command = evt.getActionCommand();
			if (command.equals("Ok")){
			    // pass to main window the file names.
		    
		    	((MainWindow) parentFrame).loadExperiment(parentAppNode, fileNameField.getText());
		    	this.dispose();
			}
			else if (command.equals("Cancel")){
		    	this.dispose();
			}
			else if (command.equals("Browse")){
				int returnVal = fc.showOpenDialog(this);
				if (returnVal == JFileChooser.APPROVE_OPTION) {
					File file = fc.getSelectedFile();
					if (((JButton) eventObj).getMnemonic() == 102) {						 
						 fileNameField.setText(file.getAbsolutePath());
					}			
						
					//System.out.println(file.getAbsolutePath());
				} 
				else {
					//	log.append("Open command cancelled by user." + newline);
				}
			}
	    }
	}
    }
	

	//loading trial dialog.
    public class TrialLoadDialog extends JDialog implements ActionListener{
	private Frame parentFrame;
	private DefaultMutableTreeNode parentExpNode;
	final private JTextField pprofdatField;
	final private JTextField probSizeField;
	final JFileChooser fc = new JFileChooser();
	    	    
	private JButton okButton = new JButton("Ok");
	private JButton cancelButton = new JButton("Cancel");
	    
	public TrialLoadDialog(DefaultMutableTreeNode selectedNode, Frame frame){
	    super(frame, "Load Trial");
	    this.parentFrame = frame;
	    this.parentExpNode = selectedNode;
	    pprofdatField = new JTextField(14); // location of profile file
	    probSizeField = new JTextField(14); // problem input file
	    
	    JButton pprofButton = new JButton("Browse");
	    pprofButton.setMnemonic(102);
	    JButton probSizeButton = new JButton("Browse");
	    probSizeButton.setMnemonic(103);
	    
	    
	    JLabel pprofdatFileLabel = new JLabel("Performance data file generated from pprof -d :");
	    JLabel probSizeLabel = new JLabel("Problem definition file for the trial :");
	       	    
	    Container contentPane = getContentPane();
	    GridBagLayout gridbag = new GridBagLayout();	    
	    contentPane.setLayout(gridbag);

	    JLabel mesg = new JLabel("Please provide following information.");

	    contentPane.add(mesg,    new GridBagConstraints(0, 0, 1, 1, 0.0, 0.0,
                    GridBagConstraints.CENTER,
                    GridBagConstraints.HORIZONTAL,
                    new Insets(20, 20, 10, 0), 0, 0));
	    
	    contentPane.add(pprofdatFileLabel,    new GridBagConstraints(0, 1, 1, 1, 0.0, 0.0,
                    GridBagConstraints.WEST,
                    GridBagConstraints.HORIZONTAL,
                    new Insets(10, 20, 10, 0), 0, 0));

	    contentPane.add(probSizeLabel,     new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0,
                    GridBagConstraints.WEST,
                    GridBagConstraints.HORIZONTAL,
                    new Insets(0, 20, 10, 0), 0, 0));

	    contentPane.add(pprofdatField,    new GridBagConstraints(1, 1, 2, 1, 0.0, 0.0,
                    GridBagConstraints.WEST,
                    GridBagConstraints.HORIZONTAL,
                    new Insets(10, 5, 10, 20), 0, 0));

	    contentPane.add(probSizeField,     new GridBagConstraints(1, 2, 2, 1, 0.0, 0.0,
                    GridBagConstraints.WEST,
                    GridBagConstraints.HORIZONTAL,         
                    new Insets(0, 0, 10, 20), 0, 0));
                    
		contentPane.add(pprofButton,   new GridBagConstraints(3, 1, 2, 1, 0.0, 0.0,
							GridBagConstraints.WEST,
							GridBagConstraints.HORIZONTAL,
							new Insets(10, 5, 10, 20), 0, 0));

		contentPane.add(probSizeButton,     new GridBagConstraints(3, 2, 2, 1, 0.0, 0.0,
							GridBagConstraints.WEST,
							GridBagConstraints.HORIZONTAL,         
							new Insets(0, 0, 10, 20), 0, 0));            
	    
	    contentPane.add(okButton,   new GridBagConstraints(1, 3, 1, 1, 0.0, 0.0,
                    GridBagConstraints.SOUTH,
                    GridBagConstraints.HORIZONTAL,
                    new Insets(20, 0, 10, 0), 20, 0));

	    contentPane.add(cancelButton,   new GridBagConstraints(2, 3, 1, 1, 0.0, 0.0,
                    GridBagConstraints.SOUTH,
                    GridBagConstraints.HORIZONTAL,
                    new Insets(20, 5, 10, 20), 15, 0));
	    
		pprofButton.addActionListener(this);
		probSizeButton.addActionListener(this);
	    
	    okButton.addActionListener(this);

	    cancelButton.addActionListener(this);

	    // Set the size and location
	    this.pack();
	    
	    //Center this dialog on the screen.
	    Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
	    int x = ( screen.width / 2) - (this.getSize().width /2 );
	    int y = ( screen.height / 2) - (this.getSize().height /2 );
	    this.setLocation(x,y);

	    // Show us.
	    this.setVisible(true);
	    this.setResizable(false);
	}

	public void actionPerformed(ActionEvent evt) {
	    Object eventObj = evt.getSource();
      
	    if (eventObj instanceof JButton){ // if this event is fired by a button.
		String command = evt.getActionCommand();
		if (command.equals("Ok")){
		    // pass to main window the file names.
		    
		    ((MainWindow) parentFrame).loadTrial(parentExpNode, pprofdatField.getText(), probSizeField.getText()); // load the trial with input parameters
		    this.dispose();
		}
		else if (command.equals("Cancel")){
		    this.dispose();
		}
		else if (command.equals("Browse")){
			int returnVal = fc.showOpenDialog(this);
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				if (((JButton) eventObj).getMnemonic() == 102) {						 
					pprofdatField.setText(file.getAbsolutePath());
				}
				else {
					probSizeField.setText(file.getAbsolutePath());
				}	
			}				
		}
	    }
	}	
    }

	public class MultiCounterLoadDialog extends JDialog implements ActionListener{
		private Frame parentFrame;
		private DefaultMutableTreeNode trialNode;
		final private JTextField fileNameField;
	
		final JFileChooser fc = new JFileChooser();

		private JButton okButton = new JButton("Ok");
		private JButton cancelButton = new JButton("Cancel");

	 //	load multiple counter performance data dialog
	 // selectedTrialNode provides the trial id.	 
	 public MultiCounterLoadDialog(DefaultMutableTreeNode selectedTrialNode, Frame parentFrame){
	    
		 super(parentFrame, "Load hardware performance counter data");
		 this.parentFrame = parentFrame;
		 this.trialNode = selectedTrialNode;
	    
		 fileNameField = new JTextField(14);
	    
		 JLabel fileNameLabel = new JLabel("Performance data file name generated from pprof -d :");
	    
		 JButton fileBrowseButton = new JButton("Browse");
		 fileBrowseButton.setMnemonic(102);
		
		 Container contentPane = getContentPane();
		 GridBagLayout gridbag = new GridBagLayout();	    
		 contentPane.setLayout(gridbag);

		 JLabel mesg = new JLabel("Please provide following file name.");

		 contentPane.add(mesg, new GridBagConstraints(0, 0, 1, 1, 0.0, 0.0,
					 GridBagConstraints.CENTER,
					 GridBagConstraints.HORIZONTAL,
					 new Insets(20, 30, 10, 0), 0, 0));
	    
		 contentPane.add(fileNameLabel,    new GridBagConstraints(0, 1, 1, 1, 0.0, 0.0,
					 GridBagConstraints.WEST,
					 GridBagConstraints.HORIZONTAL,
					 new Insets(0, 20, 10, 0), 0, 0));
	    
		 contentPane.add(fileNameField,    new GridBagConstraints(1, 1, 2, 1, 0.0, 0.0,
					 GridBagConstraints.WEST,
					 GridBagConstraints.HORIZONTAL,
					 new Insets(0, 0, 10, 20), 0, 0));     

		 contentPane.add(fileBrowseButton, new GridBagConstraints(3, 1, 1, 1, 0.0, 0.0,
									 GridBagConstraints.WEST,
									 GridBagConstraints.HORIZONTAL,
									 new Insets(0, 0, 10, 20), 0, 0));
		 fileBrowseButton.addActionListener(this);		

		 contentPane.add(okButton,   new GridBagConstraints(1, 2, 1, 1, 0.0, 0.0,
					 GridBagConstraints.SOUTH,
					 GridBagConstraints.HORIZONTAL,
					 new Insets(30, 0, 10, 0), 20, 0));

		 contentPane.add(cancelButton,   new GridBagConstraints(2, 2, 1, 1, 0.0, 0.0,
					 GridBagConstraints.SOUTH,
					 GridBagConstraints.HORIZONTAL,
				 new Insets(30, 5, 10, 20), 15, 0));
	    
		 okButton.addActionListener(this);

		 cancelButton.addActionListener(this);

		 // Set the size and location
		 this.pack();
	    	    
		 //Center this dialog on the screen.
		 Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
		 int x = ( screen.width / 2) - (this.getSize().width /2 );
		 int y = ( screen.height / 2) - (this.getSize().height /2 );
		 setLocation(x,y);

		 // Show us.
		 this.show();	    	    	    
		 this.setResizable(false);
	 }

	 public void actionPerformed(ActionEvent evt) {
		 Object eventObj = evt.getSource();
      
		 if (eventObj instanceof JButton){ // if this event is fired by a button.
			  String command = evt.getActionCommand();
			 if (command.equals("Ok")){
				 // pass to main window the file names.
		    
				 ((MainWindow) parentFrame).loadHWCounterData(trialNode, fileNameField.getText());
				 this.dispose();
			 }
			 else if (command.equals("Cancel")){
				 this.dispose();
			 }
			 else if (command.equals("Browse")){
				 int returnVal = fc.showOpenDialog(this);
				 if (returnVal == JFileChooser.APPROVE_OPTION) {
					 File file = fc.getSelectedFile();
					 if (((JButton) eventObj).getMnemonic() == 102) {						 
						  fileNameField.setText(file.getAbsolutePath());
					 }			
						
					 //System.out.println(file.getAbsolutePath());
				 } 
				 else {
					 //	log.append("Open command cancelled by user." + newline);
				 }
			 }
		 }
	 }
	 }

	public static void main(String[] args) {
		final DataSession session = new PerfDMFSession();
		session.initialize(args[0]);

		//Make sure we have nice window decorations.
		JFrame.setDefaultLookAndFeelDecorated(true);

		final JFrame frame = new MainWindow(session, args[0]);

		frame.addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
			    			    
			    session.terminate();
			    frame.dispose();
			    System.exit(0);
			}
		});  

		frame.pack();
		frame.setVisible(true);
	}	
 }
