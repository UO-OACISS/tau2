package edu.uoregon.tau.paraprof.treetable;

import java.awt.EventQueue;
import java.awt.Point;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.dnd.*;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.SQLException;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.ListIterator;

import javax.swing.*;
import javax.swing.tree.*;

import org.jfree.ui.tabbedui.DetailEditor;

import edu.uoregon.tau.paraprof.LoadTrialProgressWindow;
import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.paraprof.ParaProfApplication;
import edu.uoregon.tau.paraprof.ParaProfExperiment;
import edu.uoregon.tau.paraprof.ParaProfMetric;
import edu.uoregon.tau.paraprof.ParaProfTreeCellRenderer;
import edu.uoregon.tau.paraprof.ParaProfTrial;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.DBDataSource;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.DataSourceException;
import edu.uoregon.tau.perfdmf.Database;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.database.DB;

public class TreeDropTarget implements DropTargetListener {

    DropTarget target;

    JTree targetTree;

    public TreeDropTarget(JTree tree) {
	targetTree = tree;
	target = new DropTarget(targetTree, this);
    }

    /*
     * Drop Event Handlers
     */
    private TreeNode getNodeForEvent(DropTargetDragEvent dtde) {
	Point p = dtde.getLocation();
	DropTargetContext dtc = dtde.getDropTargetContext();
	JTree tree = (JTree) dtc.getComponent();
	TreePath path = tree.getClosestPathForLocation(p.x, p.y);
	return (TreeNode) path.getLastPathComponent();
    }

    public void dragEnter(DropTargetDragEvent dtde) {
	TreeNode node = getNodeForEvent(dtde);
	if (node.isLeaf()) {
	    dtde.rejectDrag();
	} else {
	    // start by supporting move operations
	    // dtde.acceptDrag(DnDConstants.ACTION_MOVE);
	    dtde.acceptDrag(dtde.getDropAction());
	}
    }

    public void dragOver(DropTargetDragEvent dtde) {
	TreeNode node = getNodeForEvent(dtde);
	// if (node.isLeaf()) {
	// dtde.rejectDrag();
	// } else {
	// start by supporting move operations
	// dtde.acceptDrag(DnDConstants.ACTION_MOVE);
	dtde.acceptDrag(dtde.getDropAction());
	// }
    }

    public void dragExit(DropTargetEvent dte) {
    }

    public void dropActionChanged(DropTargetDragEvent dtde) {
    }

    public void drop(DropTargetDropEvent dtde) {
	Point pt = dtde.getLocation();
	DropTargetContext dtc = dtde.getDropTargetContext();
	JTree tree = (JTree) dtc.getComponent();
	TreePath parentpath = tree.getClosestPathForLocation(pt.x, pt.y);
	DefaultMutableTreeNode newParent = (DefaultMutableTreeNode) parentpath
	.getLastPathComponent();

	try {
	    Transferable tr = dtde.getTransferable();
	    DataFlavor[] flavors = tr.getTransferDataFlavors();
	    for (int i = 0; i < flavors.length; i++) {
		if (tr.isDataFlavorSupported(flavors[i])) {
		    dtde.acceptDrop(dtde.getDropAction());
		    DefaultMutableTreeNode node = (DefaultMutableTreeNode) tr
		    .getTransferData(flavors[i]);
		    Object object = node.getUserObject();
		    if (node.getParent() == newParent) {
			// dtde.rejectDrop();
			return;
		    }

		    if (object instanceof ParaProfApplication) {
			if (!(newParent.getUserObject() instanceof Database)) {
			    return;
			}

			ParaProfApplication app = (ParaProfApplication) object;
			DatabaseAPI newDB = ParaProf.paraProfManagerWindow
			.getDatabaseAPI((Database) newParent.getUserObject());
			
			expand(node);
			
			String appName = getAppName(app);
			String dbName =((Database) newParent.getUserObject()).getName();
			
			if (JOptionPane.showConfirmDialog(ParaProf.paraProfManagerWindow,
				"Are you sure you want to move "+appName+" to "+dbName+" ?", "Move Application",
				JOptionPane.YES_NO_OPTION) == JOptionPane.NO_OPTION) {
			    return;
			}


			uploadApplication(newDB, app, node.children());

			deleteObject(node.getUserObject());
			TreePath path = new TreePath(newParent.getPath());
			if (targetTree.isExpanded(path))
			    targetTree.collapsePath(path);
			targetTree.expandPath(path);

		    }

		    else if (object instanceof ParaProfExperiment) {
			if (!(newParent.getUserObject() instanceof ParaProfApplication)) {
			    return;
			}
			ParaProfExperiment exp = (ParaProfExperiment) object;
			ParaProfApplication app = (ParaProfApplication) newParent.getUserObject();
			if(!app.dBApplication() && exp.dBExperiment())
			    return;
			String expName =getAppName(exp.getApplication())+":" +exp.getName();
			String appName = getAppName(app);
			
			if (JOptionPane.showConfirmDialog(ParaProf.paraProfManagerWindow,
				"Are you sure you want to move "+expName+" to "+appName+"?", "Move Experiment",
				JOptionPane.YES_NO_OPTION) == JOptionPane.NO_OPTION) {
			    return;
			}


			expand(node);
			if(!app.dBApplication()){
			    ParaProfApplication oldapp = exp.getApplication();
			    DefaultMutableTreeNode oldDMTN = exp.getDMTN();
			    
			    uploadExperment(app, exp, node.children());
			    ParaProf.paraProfManagerWindow.getTreeModel().removeNodeFromParent(oldDMTN);
			    oldapp.removeExperiment(exp);

			}else{
			    DatabaseAPI newDB = ParaProf.paraProfManagerWindow
			    .getDatabaseAPI((Database) ((DefaultMutableTreeNode) newParent
				    .getParent()).getUserObject());


			    uploadExperiment(app, exp, newDB, node.children());
			    deleteObject(node.getUserObject());
			}
			TreePath path = new TreePath(newParent.getPath());
			if (targetTree.isExpanded(path))
			    targetTree.collapsePath(path);
			targetTree.expandPath(path);


		    } else if (object instanceof ParaProfTrial) {
			if (!(newParent.getUserObject() instanceof ParaProfExperiment)) {
			    return;
			}
			ParaProfTrial trial = (ParaProfTrial) node.getUserObject();
			ParaProfExperiment exp = (ParaProfExperiment) newParent.getUserObject();
			if(!exp.dBExperiment() && trial.dBTrial())
			    return;
			String trialName = getAppName(trial.getExperiment().getApplication())+":"+
			trial.getExperiment().getName()+":"+trial.getName();
			String expName =getAppName(exp.getApplication())+":" +exp.getName();

			if (JOptionPane.showConfirmDialog(ParaProf.paraProfManagerWindow,
				"Are you sure you want to move "+trialName+" to "+expName+"?", "Move Trial",
				JOptionPane.YES_NO_OPTION) == JOptionPane.NO_OPTION) {
			    return;
			}

			expand(node);
			if(!exp.dBExperiment()){
			    ParaProfExperiment oldexp = trial.getExperiment();
			    ParaProf.paraProfManagerWindow.getTreeModel().removeNodeFromParent(trial.getDMTN());
			    uploadTrial(trial, exp);
			    oldexp.removeTrial(trial);
			}else{
			    DatabaseAPI newDB = ParaProf.paraProfManagerWindow
			    .getDatabaseAPI((Database) ((DefaultMutableTreeNode) newParent
				    .getParent().getParent()).getUserObject());
			    uploadTrial(trial, exp, newDB);
			    deleteObject(node.getUserObject());
			}
			TreePath path = new TreePath(newParent.getPath());
			if (targetTree.isExpanded(path))
			    targetTree.collapsePath(path);
			targetTree.expandPath(path);

		    } else {
			return;
		    }

		    dtde.dropComplete(true);
		    return;
		}
	    }
	    dtde.rejectDrop();
	} catch (Exception e) {
	    e.printStackTrace();

	}
    }

    private String getAppName(ParaProfApplication app) {
	if(app.getDatabase() != null)
	   return  app.getDatabase().getName()+":"+app.getName();
	else
	   return "Standard Applications:"+app.getName();
    }

    private void uploadExperment(ParaProfApplication app, ParaProfExperiment exp,
	    Enumeration<DefaultMutableTreeNode> children) throws FileNotFoundException, DataSourceException, InterruptedException, IOException, SQLException {
	ParaProfExperiment newExp = app.addExperiment();
	newExp.setName(exp.getName());
	if (children != null) {
	    while (children.hasMoreElements()) {
		DefaultMutableTreeNode child = children.nextElement();
		expand(child);
		uploadTrial((ParaProfTrial) child.getUserObject(), newExp);
	    }
	}


    }


    private void uploadTrial(ParaProfTrial trial, ParaProfExperiment exp) {
	exp.addTrial(trial);
	trial.setApplicationID(exp.getApplicationID());
	trial.setExperimentID(exp.getID());
	//trial.setDBTrial(false);

    }

    private void expand(DefaultMutableTreeNode node) throws InterruptedException,
    FileNotFoundException, DataSourceException, IOException, SQLException {

	if (node.getUserObject() instanceof ParaProfTrial) {
	    trialWillExpand(node);
	    waitForLoad((ParaProfTrial) node.getUserObject());
	} else {
	    targetTree.expandPath(new TreePath(node.getPath()));
	}
	Enumeration<DefaultMutableTreeNode> children = node.children();

	if (children != null) {
	    while (children.hasMoreElements()) {
		DefaultMutableTreeNode child = children.nextElement();
		expand(child);

	    }
	}

    }

    private void uploadApplication(DatabaseAPI newDB, ParaProfApplication app,
	    Enumeration<DefaultMutableTreeNode> childern) throws DatabaseException, SQLException,
	    InterruptedException, FileNotFoundException, DataSourceException, IOException {
	Application newApp = new Application(app);
	newApp.setID(-1); // must set the ID to -1 to indicate that this is a
	// new application (bug found by Sameer on 2005-04-19)
	ParaProfApplication application = new ParaProfApplication(newApp);
	application.setID(-1);
	application.setDBApplication(true);
	application.setID(newDB.saveApplication(application));

	for (; childern.hasMoreElements();) {
	    DefaultMutableTreeNode node = childern.nextElement();
	    ParaProfExperiment ppExp = (ParaProfExperiment) node.getUserObject();
	    uploadExperiment(application, ppExp, newDB, node.children());
	}
    }

    private void uploadTrial(ParaProfTrial ppTrial, ParaProfExperiment dbExp, DatabaseAPI dbAPI)
    throws FileNotFoundException, DataSourceException, IOException, SQLException,
    InterruptedException {
	waitForLoad(ppTrial);

	if (ppTrial.getTrial().getDataSource().getMetadataString().equals(""))
	    ppTrial.getTrial().getDataSource().buildXMLMetaData();
	while (ppTrial.loading()) {
	    // If the trial is in the middle of loading, we can't move it
	    Thread.sleep(10);
	}
	ppTrial.setExperiment(dbExp);

	ParaProfTrial dbTrial = new ParaProfTrial(ppTrial.getTrial());
	dbTrial.setID(-1);
	dbTrial.setExperimentID(dbExp.getID());
	dbTrial.setApplicationID(dbExp.getApplicationID());

	dbTrial.getTrial().setDataSource(ppTrial.getDataSource());
	dbTrial.setExperiment(dbExp);

	dbTrial.setUpload(true);
	dbTrial.setDatabaseAPI(dbAPI);
	dbTrial.getTrial().setID(-1);
	if (dbAPI != null) {
	    // this call will block until the entire thing is uploaded (could be
	    // a while)
	    dbTrial.setID(dbAPI.uploadTrial(dbTrial.getTrial()));
	    // dbAPI.terminate();
	}

	// Now safe to set this to be a dbTrial.
	dbTrial.setDBTrial(true);
	// ParaProf.paraProfManagerWindow.populateTrialMetrics(ppTrial);

    }

    private void uploadExperiment(ParaProfApplication app, ParaProfExperiment exp,
	    DatabaseAPI databaseAPI, Enumeration<DefaultMutableTreeNode> childern)
    throws DatabaseException, SQLException, InterruptedException, FileNotFoundException,
    DataSourceException, IOException {
	Experiment newExp = new Experiment(exp);
	ParaProfExperiment experiment = new ParaProfExperiment(newExp);
	newExp.setID(-1);
	experiment.setID(-1);
	experiment.setDBExperiment(true);
	experiment.setApplicationID(app.getID());
	experiment.setApplication(app);
	experiment.setID(databaseAPI.saveExperiment(experiment));

	for (; childern.hasMoreElements();) {
	    DefaultMutableTreeNode node = childern.nextElement();
	    ParaProfTrial ppTrial = (ParaProfTrial) node.getUserObject();
	    DatabaseAPI newDB = ParaProf.paraProfManagerWindow.getDatabaseAPI(experiment
		    .getDatabase());
	    uploadTrial(ppTrial, experiment, databaseAPI);
	}

    }

    private void waitForLoad(ParaProfTrial ppTrial) throws InterruptedException {
	while (ppTrial.loading()) {
	    Thread.sleep(10);
	}

    }

    private void trialWillExpand(DefaultMutableTreeNode selectedNode) throws FileNotFoundException,
    DataSourceException, IOException, SQLException {
	Object userObject = selectedNode.getUserObject();

	ParaProfTrial trial = (ParaProfTrial) userObject;
	if (trial.dBTrial()) {

	    // test to see if trial has already been loaded
	    // if so, we re-associate the ParaProfTrial with the DMTN since
	    // the old one is gone
	    boolean loaded = false;
	    for (Enumeration<ParaProfTrial> e = ParaProf.paraProfManagerWindow.getLoadedDBTrials()
		    .elements(); e.hasMoreElements();) {
		ParaProfTrial loadedTrial = e.nextElement();
		if ((trial.getID() == loadedTrial.getID())
			&& (trial.getExperimentID() == loadedTrial.getExperimentID())
			&& (trial.getApplicationID() == loadedTrial.getApplicationID())) {
		    selectedNode.setUserObject(loadedTrial);
		    loadedTrial.setDMTN(selectedNode);
		    trial = loadedTrial;
		    loaded = true;
		}
	    }
	    final ParaProfTrial ppTrial = trial;

	    if (!loaded) {

		if (ppTrial.loading()) {
		    return;
		}

		// load the trial in from the db
		ppTrial.setLoading(true);

		DatabaseAPI databaseAPI = ParaProf.paraProfManagerWindow.getDatabaseAPI(ppTrial
			.getDatabase());
		if (databaseAPI != null) {
		    databaseAPI.setApplication(ppTrial.getApplicationID());
		    databaseAPI.setExperiment(ppTrial.getExperimentID());
		    databaseAPI.setTrial(ppTrial.getID(), true);
		    DBDataSource dbDataSource = new DBDataSource(databaseAPI);
		    dbDataSource.setGenerateIntermediateCallPathData(ParaProf.preferences
			    .getGenerateIntermediateCallPathData());
		    ppTrial.getTrial().setDataSource(dbDataSource);
		    final DataSource dataSource = dbDataSource;
		    final ParaProfTrial theTrial = ppTrial;

		    dataSource.load();
		    theTrial.finishLoad();
		    ParaProf.paraProfManagerWindow.getLoadedTrials().add(ppTrial);

		    // Add to the list of loaded trials.
		    ParaProf.paraProfManagerWindow.getLoadedDBTrials().add(ppTrial);
		}
	    }
	}
    }

    private void deleteObject(Object ppTrial) {

	try {
	    ParaProf.paraProfManagerWindow.handleDelete(ppTrial);
	} catch (DatabaseException e) {
	    // TODO Auto-generated catch block
	    e.printStackTrace();
	} catch (SQLException e) {
	    // TODO Auto-generated catch block
	    e.printStackTrace();
	}
    }

}
