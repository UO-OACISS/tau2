/*
 * @(#)TreeTableModelAdapter.java	1.2 98/10/27
 *
 * Copyright 1997, 1998 by Sun Microsystems, Inc.,
 * 901 San Antonio Road, Palo Alto, California, 94303, U.S.A.
 * All rights reserved.
 *
 * This software is the confidential and proprietary information
 * of Sun Microsystems, Inc. ("Confidential Information").  You
 * shall not disclose such Confidential Information and shall use
 * it only in accordance with the terms of the license agreement
 * you entered into with Sun.
 */
package edu.uoregon.tau.common.treetable;

import java.util.Enumeration;
import java.util.Vector;

import javax.swing.JTree;
import javax.swing.SwingUtilities;
import javax.swing.event.TreeExpansionEvent;
import javax.swing.event.TreeExpansionListener;
import javax.swing.event.TreeModelEvent;
import javax.swing.event.TreeModelListener;
import javax.swing.table.AbstractTableModel;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;

/**
 * This is a wrapper class takes a TreeTableModel and implements 
 * the table model interface. The implementation is trivial, with 
 * all of the event dispatching support provided by the superclass: 
 * the AbstractTableModel. 
 *
 * @version 1.2 10/27/98
 *
 * @author Philip Milne
 * @author Scott Violet
 */
public class TreeTableModelAdapter extends AbstractTableModel implements SortTableModel {
    /**
	 * 
	 */
	private static final long serialVersionUID = -2121699681828347799L;
	private JTree tree;
    private AbstractTreeTableModel treeTableModel;
    private JTreeTable jTreeTable;

    public TreeTableModelAdapter(AbstractTreeTableModel treeTableModel, JTreeTable jTreeTable) {
        this.tree = jTreeTable.getTree();
        this.jTreeTable = jTreeTable;
        this.treeTableModel = treeTableModel;

        tree.addTreeExpansionListener(new TreeExpansionListener() {
            // Don't use fireTableRowsInserted() here; the selection model
            // would get updated twice. 
            public void treeExpanded(TreeExpansionEvent event) {
                fireTableDataChanged();
            }

            public void treeCollapsed(TreeExpansionEvent event) {
                fireTableDataChanged();
            }
        });

        // Install a TreeModelListener that can update the table when
        // tree changes. We use delayedFireTableDataChanged as we can
        // not be guaranteed the tree will have finished processing
        // the event before us.
        treeTableModel.addTreeModelListener(new TreeModelListener() {
            public void treeNodesChanged(TreeModelEvent e) {
                delayedFireTableDataChanged();
            }

            public void treeNodesInserted(TreeModelEvent e) {
                delayedFireTableDataChanged();
            }

            public void treeNodesRemoved(TreeModelEvent e) {
                delayedFireTableDataChanged();
            }

            public void treeStructureChanged(TreeModelEvent e) {
                delayedFireTableDataChanged();
            }
        });
    }

    // Wrappers, implementing TableModel interface. 

    public int getColumnCount() {
        return treeTableModel.getColumnCount();
    }

    public String getColumnName(int column) {
        return treeTableModel.getColumnName(column);
    }

    public Class<Object> getColumnClass(int column) {
        return treeTableModel.getColumnClass(column);
    }

    public int getRowCount() {
        return tree.getRowCount();
    }

    protected Object nodeForRow(int row) {
        TreePath treePath = tree.getPathForRow(row);
        return treePath.getLastPathComponent();
    }

    public Object getValueAt(int row, int column) {
        return treeTableModel.getValueAt(nodeForRow(row), column);
    }

    public boolean isCellEditable(int row, int column) {
        return treeTableModel.isCellEditable(nodeForRow(row), column);
    }

    public void setValueAt(Object value, int row, int column) {
        treeTableModel.setValueAt(value, nodeForRow(row), column);
    }

    /***
     * Returns a vector of open paths in the tree, can be used to
     * re-open the paths in a tree after a call to 'treeStructureChanged'
     * (which causes all open paths to collapse)
     */
    public Vector<TreePath> getExpandedPaths() {
        Enumeration<TreePath> expanded = tree.getExpandedDescendants(getRootPath());

        Vector<TreePath> paths = new Vector<TreePath>();
        if (expanded != null) {
            while (expanded.hasMoreElements()) {
                paths.add(expanded.nextElement());
            }
        }

        return paths;
    }

    /***
     * Restores the given open paths on the treeModel.
     * @param paths a Vector of TreePaths which are going to be opened.
     */
    public void restoreExpandedPaths(Vector<TreePath> paths) {
        Enumeration<TreePath> e = paths.elements();
        while (e.hasMoreElements()) {
            TreePath path = e.nextElement();
            tree.expandPath(path);
        }
    }

    /***
     * Returns the (tree)path to the root of the model.
     * @return
     */
    public TreePath getRootPath() {
        DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) treeTableModel.getRoot();
        return new TreePath(rootNode.getPath());
    }

    /**
     * Invokes fireTableDataChanged after all the pending events have been
     * processed. SwingUtilities.invokeLater is used to handle this.
     */
    protected void delayedFireTableDataChanged() {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                fireTableDataChanged();
            }
        });
    }

    public void updateTreeTable() {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                Vector<TreePath> pathState = getExpandedPaths();

                treeTableModel.fireTreeStructureChanged(this, ((DefaultMutableTreeNode) treeTableModel.getRoot()).getPath(), null, null);

                restoreExpandedPaths(pathState);

                jTreeTable.getTableHeader().repaint();

                delayedFireTableDataChanged();
            }
        });
    }

    public boolean isSortable(int col) {
        return treeTableModel.isSortable(col);
    }

    public void sortColumn(int col, boolean ascending) {
        treeTableModel.sortColumn(col, ascending);
        updateTreeTable();
    }

}
