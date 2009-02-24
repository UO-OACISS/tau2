package edu.uoregon.tau.perfexplorer.client;

import javax.swing.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import javax.swing.tree.DefaultTreeModel;

import edu.uoregon.tau.perfexplorer.common.RMIView;

public class PerfExplorerViews {

	public static void createNewView (JFrame mainFrame, int parent) {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// is this an Application, Experiment, or Trial selection?
		Object[] options = new Object[3];
		options[0] = new String("Application");
		options[1] = new String("Experiment");
		options[2] = new String("Trial");
		String table = (String)JOptionPane.showInputDialog (mainFrame,
			"Select a level in the hierarchy:",
			"Select Level",
			JOptionPane.PLAIN_MESSAGE,
			null,
			options,
			"Application");

		if (table == null)
			return;

		// get the columns for the level they selected
		String[] tmp = server.getMetaData(table);
		String[] columns = new String[tmp.length+1];
		columns[0] = new String("name");
		for (int i = 0 ; i < tmp.length ; i++)
			columns[i+1] = tmp[i];
		String column = (String)JOptionPane.showInputDialog (mainFrame,
			"Select a column to filter:",
			"Select Column",
			JOptionPane.PLAIN_MESSAGE,
			null,
			columns,
			columns[0]);

		if (column == null)
			return;

		// get the operator they want
		Object[] operators = {"=", "!=", "<", ">", "<=", ">=", "like", "not like"};
		String oper = (String)JOptionPane.showInputDialog (mainFrame,
			"Select an operator:",
			"Select Operator",
			JOptionPane.PLAIN_MESSAGE,
			null,
			operators,
			operators[0]);

		if (oper == null)
			return;

		// get the value they want to search on
		List values = server.getPossibleValues(table, column);
		if (values.size() < 2) {
			JOptionPane.showMessageDialog(mainFrame, "The column " + 
				column.toUpperCase() + " in table " + table.toUpperCase() + 
				" has only one distinct value.\nPlease select another table and/or column.",
				"Request Status", JOptionPane.ERROR_MESSAGE);
			return;
		} 

		final String all = new String ("CREATE VIEW FOR EACH");
		// only allow them to create a view for all values if
		// they chose the equals operator
		if (oper.equals("="))
			values.add(all);
		String value = null;
		if (oper.equals("like") || oper.equals("not like")) {
			value = (String)JOptionPane.showInputDialog (mainFrame,
				"Please enter a value to filter (use % for wildcard):",
				"Enter Value",
				JOptionPane.PLAIN_MESSAGE);
		} else {
			value = (String)JOptionPane.showInputDialog (mainFrame,
			"Select a value to filter (required):",
			"Select Value",
			JOptionPane.PLAIN_MESSAGE,
			null,
			values.toArray(),
			values.get(0));
		}
				
		if (value == null)
			return;

		if (value.equals(all)) {
			// create views for each value
			for (int i = 0 ; i < values.size()-1 ; i++) {
				value = (String) values.get(i);
				String name = table + ":" + column + " = " + value;
				if (value != null)
					server.createNewView(name, parent, table, column, oper, value);
			}
		} else {
			String name = (String)JOptionPane.showInputDialog (mainFrame,
				"Please enter a name for this view (required):",
				"Enter View Name",
				JOptionPane.PLAIN_MESSAGE, null, null, 
				table + ":" + column + " " + oper + " " + value);

			if (name == null)
				return;

			// insert the view!
			int viewID = server.createNewView(name, parent, table, column, oper, value);
		}
		JOptionPane.showMessageDialog (mainFrame, "View(s) Created.", 
			"View Created", JOptionPane.PLAIN_MESSAGE);
	}

	public static void createNewSubView (JFrame mainFrame) {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		int parent = -1;  // get all the views!
		List views = server.getViews(parent);
		
		Iterator e = views.iterator();
		List options = new ArrayList();
		Hashtable myViews = new Hashtable();
		while (e.hasNext()) {
			RMIView view = (RMIView) e.next();
			options.add(view.getField("NAME"));
			myViews.put(view.getField("NAME"), view.getField("ID"));
		}
		String viewName = (String)JOptionPane.showInputDialog (mainFrame, 
			"Select a view on which to base this sub-view:", 
			"Select View", 
			JOptionPane.PLAIN_MESSAGE, 
			null, 
			options.toArray(), null );

		if (viewName != null) {
			String viewID = (String) myViews.get(viewName);
			createNewView(mainFrame, Integer.parseInt(viewID));
		}
	}

	public static void deleteCurrentView(JFrame mainFrame) {
		PerfExplorerModel model = PerfExplorerModel.getModel();
		Object selection = model.getCurrentSelection();
		if (selection instanceof RMIView) {
			// get the server
			PerfExplorerConnection server = PerfExplorerConnection.getConnection();
			RMIView view = (RMIView)selection;
			// delete the view
			server.deleteView(view.getField("id"));

			// get the tree
			JTree tree = PerfExplorerJTree.getTree();
			DefaultTreeModel treeModel = (DefaultTreeModel)tree.getModel();

			// delete the node from the tree
			treeModel.removeNodeFromParent(view.getDMTN());
		} else {
			JOptionPane.showMessageDialog(mainFrame, 
				"Current selection is not a View.\nPlease select the view to delete before attempting to delete it.",
				"Selection Error", JOptionPane.ERROR_MESSAGE);
		}
	}
}
