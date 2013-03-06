package edu.uoregon.tau.perfexplorer.client;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.sql.SQLException;

import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPopupMenu;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreeNode;
import javax.swing.tree.TreePath;

import edu.uoregon.tau.common.TreeUI;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;

public class PerfExplorerTreeMouseListener implements MouseListener,
		ActionListener {

	private PerfExplorerJTree tree;
	private Object clickedOnObject = null;
	private DefaultMutableTreeNode selectedNode = null;
	private TreePath fullPathNode = null;
	JPopupMenu perfExPopup = null;
	JPopupMenu multiPopup = null;

	// private class MouseActionListener implements ActionListener{
	public void actionPerformed(ActionEvent evt) {
		try {
			Object EventSrc = evt.getSource();
			if (EventSrc instanceof JMenuItem) {
				String arg = evt.getActionCommand();

				if (arg.equals("Delete")) {
					handleDelete(clickedOnObject, true);
				} else if (arg.equals("Rename")) {
					if (clickedOnObject instanceof Application) {
						tree.startEditingAtPath(fullPathNode);
					} else if (clickedOnObject instanceof Experiment) {
						tree.startEditingAtPath(fullPathNode);
					} else if (clickedOnObject instanceof Trial) {
						tree.startEditingAtPath(fullPathNode);
					} else if (clickedOnObject instanceof View) {
						tree.startEditingAtPath(new TreePath(
								((View) clickedOnObject).getDMTN().getPath()));
					} else if (clickedOnObject instanceof Metric) {
						tree.startEditingAtPath(fullPathNode);
					}

				}
				clickedOnObject = null;
				fullPathNode = null;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	// }

	public PerfExplorerTreeMouseListener(PerfExplorerJTree tree) {
		super();
		this.tree = tree;
		perfExPopup = TreeUI.getPerfExPopUp(this);
		multiPopup = TreeUI.getMultiPopUp(this);
	}

	public void mouseClicked(MouseEvent evt) {
		// if (userObject instanceof ParaProfMetric) {
		// ParaProfMetric ppMetric = (ParaProfMetric) userObject;

		// if (e.getClickCount() == 2) {
		if (!TreeUI.rightClick(evt)) {
			Object selected = PerfExplorerModel.getModel()
					.getCurrentSelection();
			if (selected != null) {
				if (selected instanceof Metric) {
					PerfExplorerJTabbedPane pane = PerfExplorerJTabbedPane
							.getPane();
					int index = pane.getSelectedIndex();
					if (index == 4) {
						DeriveMetricsPane.getPane().metricClick(
								(Metric) selected);
					}

				}
			}
			return;
		}

		int row = tree.getRowForLocation(evt.getX(), evt.getY());
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
		} else if (paths.length == 1) { // only one item is selected
			TreePath path = paths[0];
			fullPathNode = path;
			selectedNode = (DefaultMutableTreeNode) path.getLastPathComponent();
			Object userObject = selectedNode.getUserObject();

			// System.out.println(userObject.getClass());

			if (userObject instanceof Application) {
				clickedOnObject = userObject;
				this.perfExPopup.show(tree, evt.getX(), evt.getY());
			} else if (userObject instanceof Experiment) {
				clickedOnObject = userObject;
				this.perfExPopup.show(tree, evt.getX(), evt.getY());
			} else if (userObject instanceof Trial) {
				clickedOnObject = userObject;
				this.perfExPopup.show(tree, evt.getX(), evt.getY());
			} else if (userObject instanceof View) {
				if (((View) userObject).toString().equals("All Trials")) {
					return;
				}
				clickedOnObject = userObject;
				this.perfExPopup.show(tree, evt.getX(), evt.getY());
			}
			// if (userObject instanceof Application) {
			// clickedOnObject = userObject;

			// {

			// if (((ParaProfApplication) userObject)
			// .dBApplication()) {
			// dbAppPopup.show(tree, evt.getX(),
			// evt.getY());
			// } else {
			// stdAppPopup.show(tree, evt.getX(),
			// evt.getY());
			// }
			// this.perfExPopup.show(tree, evt.getX(),
			// evt.getY());
			// }
			// else if (userObject instanceof ParaProfExperiment) {
			// clickedOnObject = userObject;
			// if (((ParaProfExperiment) userObject)
			// .dBExperiment()) {
			// dbExpPopup.show(tree, evt.getX(),
			// evt.getY());
			// } else {
			// stdExpPopup.show(tree, evt.getX(),
			// evt.getY());
			// }
			//
			// } else if (userObject instanceof ParaProfTrial) {
			// clickedOnObject = userObject;
			// if (((ParaProfTrial) userObject).dBTrial()) {
			// dbTrialPopup.show(tree, evt.getX(),
			// evt.getY());
			// } else {
			// stdTrialPopup.show(tree, evt.getX(),
			// evt.getY());
			// }
			// } else if (userObject instanceof ParaProfMetric) {
			// clickedOnObject = userObject;
			// metricPopup.show(tree, evt.getX(), evt.getY());
			// } else if (userObject instanceof Database) {
			// // standard or database
			// clickedOnObject = selectedNode;
			// Database db = (Database)userObject;
			// DatabaseAPI dbapi = getDatabaseAPI(db);
			// if(dbapi.db().getSchemaVersion()>0){
			// db.setTAUdb(true);
			// }else{db.setTAUdb(false);}
			// if(db.isTAUdb()){
			// TAUdbPopUp.show(tree, evt.getX(), evt.getY());
			// }else{
			// databasePopUp.show(tree, evt.getX(), evt.getY());
			// }
			//
			// } else if (userObject instanceof String) {
			// // standard or database
			// clickedOnObject = selectedNode;
			// if (((String) userObject).indexOf("Standard") != -1) {
			// databasePopUp.show(tree, evt.getX(), evt.getY());
			// }
			//
			// } else if (userObject instanceof View){
			// clickedOnObject = selectedNode;
			// ViewPopUp.show(tree, evt.getX(), evt.getY());
			// }
			// }

		}

	}

	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mousePressed(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	private void handleDelete(Object object, boolean ShowConfirmation)
			throws SQLException, DatabaseException {

		if (ShowConfirmation) {
			int confirm = JOptionPane
					.showConfirmDialog(
							tree,
							"Are you sure you want to permanently delete this item from the database and all views?",
							"Confirm Delete", JOptionPane.YES_NO_OPTION);

			if (confirm != 0) {
				return;
			}
		}

		if (object instanceof TreePath[]) {
			TreePath[] paths = (TreePath[]) object;
			for (int i = 0; i < paths.length; i++) {
				DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) paths[i]
						.getLastPathComponent();
				Object userObject = selectedNode.getUserObject();
				handleDelete(userObject, false);
			}

		} else if (object instanceof Application) {
			Application application = (Application) object;

			DatabaseAPI databaseAPI = PerfExplorerServer.getServer()
					.getSession(application.getDatabase());

			if (databaseAPI != null) {
				databaseAPI.deleteApplication(application.getID());
				deleteTreeItem(application);
				// databaseAPI.terminate();
				// Remove any loaded trials associated with this
				// application.
				// for (Enumeration<ParaProfTrial> e = loadedDBTrials
				// .elements(); e.hasMoreElements();) {
				// ParaProfTrial loadedTrial = e.nextElement();
				// if (loadedTrial.getApplicationID() == application
				// .getID() && loadedTrial.loading() == false) {
				// loadedDBTrials.remove(loadedTrial);
				// }
				// }
				// if (application.getDMTN() != null)
				// getTreeModel().removeNodeFromParent(
				// application.getDMTN());
				// }
			}

		} else if (object instanceof Experiment) {
			Experiment experiment = (Experiment) object;
			// if (experiment.dBExperiment()) {

			// DatabaseAPI databaseAPI =
			// this.getDatabaseAPI(experiment.getDatabase());
			DatabaseAPI databaseAPI = PerfExplorerServer.getServer()
					.getSession(experiment.getDatabase());
			if (databaseAPI != null) {
				databaseAPI.deleteExperiment(experiment.getID());
				deleteTreeItem(experiment);
				// databaseAPI.terminate();
				// Remove any loaded trials associated with this
				// application.
				// for (Enumeration<ParaProfTrial> e = loadedDBTrials
				// .elements(); e.hasMoreElements();) {
				// ParaProfTrial loadedTrial = e.nextElement();
				// if (loadedTrial.getApplicationID() == experiment
				// .getApplicationID()
				// && loadedTrial.getExperimentID() == experiment
				// .getID()
				// && loadedTrial.loading() == false) {
				// loadedDBTrials.remove(loadedTrial);
				// }
				// }
				// if (experiment.getDMTN() != null) {
				// getTreeModel().removeNodeFromParent(
				// experiment.getDMTN());
				// }
				// }
			}
			// else {
			// experiment.getApplication().removeExperiment(experiment);
			// getTreeModel().removeNodeFromParent(experiment.getDMTN());
			// }

		} else if (object instanceof Trial) {
			Trial trial = (Trial) object;
			// if (ppTrial.dBTrial()) {
			DatabaseAPI databaseAPI = PerfExplorerServer.getServer()
					.getSession(trial.getDatabase());
			// DatabaseAPI databaseAPI =
			// this.getDatabaseAPI(ppTrial.getDatabase());
			if (databaseAPI != null) {
				databaseAPI.deleteTrial(trial.getID());
				deleteTreeItem(trial);
				// databaseAPI.terminate();
				// Remove any loaded trials associated with this
				// application.
				// for (Enumeration<ParaProfTrial> e = loadedDBTrials
				// .elements(); e.hasMoreElements();) {
				// ParaProfTrial loadedTrial = e.nextElement();
				// if (loadedTrial.getApplicationID() == ppTrial
				// .getApplicationID()
				// && loadedTrial.getExperimentID() == ppTrial
				// .getID()
				// && loadedTrial.getID() == ppTrial.getID()
				// && loadedTrial.loading() == false) {
				// loadedDBTrials.remove(loadedTrial);
				// }
				// }
				// getTreeModel().removeNodeFromParent(ppTrial.getDMTN());
			}
			// } else {
			// ppTrial.getExperiment().removeTrial(ppTrial);
			// getTreeModel().removeNodeFromParent(ppTrial.getDMTN());
		}

		// else if (object instanceof ParaProfMetric) {
		// ParaProfMetric ppMetric = (ParaProfMetric) object;
		// deleteMetric(ppMetric);
		// } else if (object instanceof DefaultMutableTreeNode){
		//
		// View view = (View) ((DefaultMutableTreeNode)object).getUserObject();
		// deleteView(view);
		// }
		else if (object instanceof View) {

			View view = (View) object;
			deleteView(view);
		}

	}

	private void deleteTreeItem(Object target) {
		if (selectedNode != null && selectedNode.getUserObject().equals(target)) {
			DefaultTreeModel dtm = (DefaultTreeModel) tree.getModel();

			TreeNode parent = selectedNode.getParent();
			selectedNode.removeFromParent();
			dtm.reload(parent);
		}
	}

	private void deleteView(View view) {
		DatabaseAPI dbAPI = PerfExplorerServer.getServer().getSession(
				view.getDatabase());

		try {
			View.deleteView(view.getID(), dbAPI.getDb());
			deleteTreeItem(view);
		} catch (SQLException e) {
			e.printStackTrace();
		}
	}
}
