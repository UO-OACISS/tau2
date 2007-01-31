package edu.uoregon.tau.common.treetable;

/*
 * @(#)JTreeTable.java	1.2 98/10/27
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

import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.text.NumberFormat;
import java.util.Date;
import java.util.Enumeration;
import java.util.EventObject;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.table.*;
import javax.swing.tree.*;

/**
 * This example shows how to create a simple JTreeTable component, 
 * by using a JTree as a renderer (and editor) for the cells in a 
 * particular column in the JTable.  
 *
 * This file (originally) comes from :
 * 
 * http://java.sun.com/products/jfc/tsc/articles/treetable1/
 *  
 * But has been heavily modified/fixed.
 *
 * @version 1.2 10/27/98
 *
 * @author Philip Milne
 * @author Scott Violet
 * @author Alan Morris
 */
public class JTreeTable extends JTable implements MouseListener {
    /** A subclass of JTree. */
    protected TreeTableCellRenderer tree;

    private int sortedColumnIndex = 0;
    private boolean sortedColumnAscending;

    public JTreeTable(final AbstractTreeTableModel treeTableModel, boolean showHandles) {
        super();

        // Create the tree. It will be used as a renderer and editor. 
        tree = new TreeTableCellRenderer(treeTableModel);
        tree.setShowsRootHandles(showHandles);
        tree.setRootVisible(false);

        this.getTableHeader().setDefaultRenderer(new SortHeaderRenderer());
        this.getTableHeader().addMouseListener(this);

        // Install a tableModel representing the visible rows in the tree. 
        super.setModel(new TreeTableModelAdapter(treeTableModel, this));

        // Force the JTable and JTree to share their row selection models. 
        ListToTreeSelectionModelWrapper selectionWrapper = new ListToTreeSelectionModelWrapper();
        tree.setSelectionModel(selectionWrapper);
        setSelectionModel(selectionWrapper.getListSelectionModel());

        // Install the tree editor renderer and editor. 
        setDefaultRenderer(TreeTableModel.class, tree);
        setDefaultEditor(TreeTableModel.class, new TreeTableCellEditor());

        // No grid.
        setShowGrid(false);

        // No intercell spacing
        setIntercellSpacing(new Dimension(0, 0));

        // And update the height of the trees row to match that of
        // the table.
        if (tree.getRowHeight() < 1) {
            // Metal looks better like this.
            setRowHeight(18);
        }

        setRowHeight(18);

    }

    public void forceRedraw() {
        SortTableModel model = (SortTableModel) getModel();
        model.updateTreeTable();
    }

    /**
     * Overridden to message super and forward the method to the tree.
     * Since the tree is not actually in the component hieachy it will
     * never receive this unless we forward it in this manner.
     */
    public void updateUI() {
        super.updateUI();
        if (tree != null) {
            tree.updateUI();
        }
        // Use the tree's default foreground and background colors in the
        // table. 
        LookAndFeel.installColorsAndFont(this, "Tree.background", "Tree.foreground", "Tree.font");
    }

    /* Workaround for BasicTableUI anomaly. Make sure the UI never tries to 
     * paint the editor. The UI currently uses different techniques to 
     * paint the renderers and editors and overriding setBounds() below 
     * is not the right thing to do for an editor. Returning -1 for the 
     * editing row in this case, ensures the editor is never painted. 
     */
    public int getEditingRow() {
        return (getColumnClass(editingColumn) == TreeTableModel.class) ? -1 : editingRow;
    }

    /**
     * Overridden to pass the new rowHeight to the tree.
     */
    public void setRowHeight(int rowHeight) {
        super.setRowHeight(rowHeight);
        if (tree != null && tree.getRowHeight() != rowHeight) {
            tree.setRowHeight(getRowHeight());
        }
    }

    /**
     * Returns the tree that is being shared between the model.
     */
    public JTree getTree() {
        return tree;
    }

    public void mouseClicked(MouseEvent e) {
        TableColumnModel colModel = this.getColumnModel();

        // get the visible column index at the current mouse coordinates
        int index = colModel.getColumnIndexAtX(e.getX());

        // check to make sure the user isn't just resizing a column
        int indexleft = colModel.getColumnIndexAtX(e.getX() - 3);
        int indexright = colModel.getColumnIndexAtX(e.getX() + 3);
        if (index != indexleft || index != indexright) {
            return;
        }

        // get the index of that column in the model
        int modelIndex = colModel.getColumn(index).getModelIndex();

        // sort the model by the selected column if that column is sortable
        SortTableModel model = (SortTableModel) getModel();
        if (model.isSortable(modelIndex)) {
            // toggle ascension, if already sorted
            if (sortedColumnIndex == modelIndex) {
                sortedColumnAscending = !sortedColumnAscending;
            }
            sortedColumnIndex = modelIndex;

            model.sortColumn(modelIndex, sortedColumnAscending);
        }

    }

    public void mouseEntered(MouseEvent e) {
    }

    public void mouseExited(MouseEvent e) {
    }

    public void mousePressed(MouseEvent e) {
    }

    public void mouseReleased(MouseEvent e) {
    }

    public Component prepareRenderer(TableCellRenderer renderer, int rowIndex, int vColIndex) {
        Component c = super.prepareRenderer(renderer, rowIndex, vColIndex);

        if (rowIndex % 2 == 0) {
            c.setBackground(new Color(235, 235, 235));
        } else {
            // If not shaded, match the table's background
            c.setBackground(getBackground());
        }
        return c;
    }

    static class NumberCellRenderer extends DefaultTableCellRenderer {
        NumberFormat formatter;

        public NumberCellRenderer() {
            super();
        }

        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row,
                int column) {

            Component c = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);

            return c;
        }

        public void setValue(Object value) {
            if (value instanceof Double) {
                setHorizontalAlignment(JLabel.RIGHT);

                if (formatter == null) {
                    formatter = NumberFormat.getInstance();
                }
                setText((value == null) ? "" : formatter.format(value));
            } else {
                setHorizontalAlignment(JLabel.LEFT);
                setText(value.toString());
            }
        }

    }

    /**
     * A TreeCellRenderer that displays a JTree.
     */
    public class TreeTableCellRenderer extends JTree implements TableCellRenderer {
        /** Last table/tree row asked to renderer. */
        protected int visibleRow;

        public TreeTableCellRenderer(TreeModel model) {
            super(model);
        }

        /**
         * updateUI is overridden to set the colors of the Tree's renderer
         * to match that of the table.
         */
        public void updateUI() {
            super.updateUI();
            // Make the tree's cell renderer use the table's cell selection colors. 
            TreeCellRenderer tcr = getCellRenderer();
            if (tcr instanceof DefaultTreeCellRenderer) {
                DefaultTreeCellRenderer dtcr = ((DefaultTreeCellRenderer) tcr);
                // For 1.1 uncomment this, 1.2 has a bug that will cause an
                // exception to be thrown if the border selection color is
                // null.
                // dtcr.setBorderSelectionColor(null);
                dtcr.setTextSelectionColor(UIManager.getColor("Table.selectionForeground"));
                dtcr.setBackgroundSelectionColor(UIManager.getColor("Table.selectionBackground"));
            }
        }

        /**
         * Sets the row height of the tree, and forwards the row height to
         * the table.
         */
        public void setRowHeight(int rowHeight) {
            if (rowHeight > 0) {
                super.setRowHeight(rowHeight);
                if (JTreeTable.this != null && JTreeTable.this.getRowHeight() != rowHeight) {
                    JTreeTable.this.setRowHeight(getRowHeight());
                }
            }
        }

        /**
         * This is overridden to set the height to match that of the JTable.
         */
        public void setBounds(int x, int y, int w, int h) {
            super.setBounds(x, 0, w, JTreeTable.this.getHeight());
        }

        /**
         * Sublcassed to translate the graphics such that the last visible
         * row will be drawn at 0,0.
         */
        public void paint(Graphics g) {
            g.translate(0, -visibleRow * getRowHeight());
            super.paint(g);
        }

        /**
         * TreeCellRenderer method. Overridden to update the visible row.
         */
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row,
                int column) {
            visibleRow = row;
            return this;
        }
    }

    /**
     * TreeTableCellEditor implementation. Component returned is the
     * JTree.
     */
    public class TreeTableCellEditor extends AbstractCellEditor implements TableCellEditor {
        public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int r, int c) {
            return tree;
        }

        /**
         * Overridden to return false, and if the event is a mouse event
         * it is forwarded to the tree.<p>
         * The behavior for this is debatable, and should really be offered
         * as a property. By returning false, all keyboard actions are
         * implemented in terms of the table. By returning true, the
         * tree would get a chance to do something with the keyboard
         * events. For the most part this is ok. But for certain keys,
         * such as left/right, the tree will expand/collapse where as
         * the table focus should really move to a different column. Page
         * up/down should also be implemented in terms of the table.
         * By returning false this also has the added benefit that clicking
         * outside of the bounds of the tree node, but still in the tree
         * column will select the row, whereas if this returned true
         * that wouldn't be the case.
         * <p>By returning false we are also enforcing the policy that
         * the tree will never be editable (at least by a key sequence).
         */
        public boolean isCellEditable(EventObject e) {
            if (e instanceof MouseEvent) {
                for (int counter = getColumnCount() - 1; counter >= 0; counter--) {
                    if (getColumnClass(counter) == TreeTableModel.class) {
                        MouseEvent me = (MouseEvent) e;
                        MouseEvent newME = new MouseEvent(tree, MouseEvent.MOUSE_PRESSED, me.getWhen(), me.getModifiers(),
                                me.getX() - getCellRect(0, counter, true).x, me.getY(), me.getClickCount(), me.isPopupTrigger());

                        tree.dispatchEvent(newME);

                        newME = new MouseEvent(tree, MouseEvent.MOUSE_RELEASED, me.getWhen(), me.getModifiers(), me.getX()
                                - getCellRect(0, counter, true).x, me.getY(), me.getClickCount(), me.isPopupTrigger());
                        tree.dispatchEvent(newME);
                        break;
                    }
                }
            }
            return false;
        }
    }

    /**
     * ListToTreeSelectionModelWrapper extends DefaultTreeSelectionModel
     * to listen for changes in the ListSelectionModel it maintains. Once
     * a change in the ListSelectionModel happens, the paths are updated
     * in the DefaultTreeSelectionModel.
     */
    class ListToTreeSelectionModelWrapper extends DefaultTreeSelectionModel {
        /** Set to true when we are updating the ListSelectionModel. */
        protected boolean updatingListSelectionModel;

        public ListToTreeSelectionModelWrapper() {
            super();
            getListSelectionModel().addListSelectionListener(createListSelectionListener());
        }

        /**
         * Returns the list selection model. ListToTreeSelectionModelWrapper
         * listens for changes to this model and updates the selected paths
         * accordingly.
         */
        ListSelectionModel getListSelectionModel() {
            return listSelectionModel;
        }

        /**
         * This is overridden to set <code>updatingListSelectionModel</code>
         * and message super. This is the only place DefaultTreeSelectionModel
         * alters the ListSelectionModel.
         */
        public void resetRowSelection() {
            if (!updatingListSelectionModel) {
                updatingListSelectionModel = true;
                try {
                    super.resetRowSelection();
                } finally {
                    updatingListSelectionModel = false;
                }
            }
            // Notice how we don't message super if
            // updatingListSelectionModel is true. If
            // updatingListSelectionModel is true, it implies the
            // ListSelectionModel has already been updated and the
            // paths are the only thing that needs to be updated.
        }

        /**
         * Creates and returns an instance of ListSelectionHandler.
         */
        protected ListSelectionListener createListSelectionListener() {
            return new ListSelectionHandler();
        }

        /**
         * If <code>updatingListSelectionModel</code> is false, this will
         * reset the selected paths from the selected rows in the list
         * selection model.
         */
        protected void updateSelectedPathsFromSelectedRows() {
            if (!updatingListSelectionModel) {
                updatingListSelectionModel = true;
                try {
                    // This is way expensive, ListSelectionModel needs an
                    // enumerator for iterating.
                    int min = listSelectionModel.getMinSelectionIndex();
                    int max = listSelectionModel.getMaxSelectionIndex();

                    clearSelection();
                    if (min != -1 && max != -1) {
                        for (int counter = min; counter <= max; counter++) {
                            if (listSelectionModel.isSelectedIndex(counter)) {
                                TreePath selPath = tree.getPathForRow(counter);

                                if (selPath != null) {
                                    addSelectionPath(selPath);
                                }
                            }
                        }
                    }
                } finally {
                    updatingListSelectionModel = false;
                }
            }
        }

        /**
         * Class responsible for calling updateSelectedPathsFromSelectedRows
         * when the selection of the list changse.
         */
        class ListSelectionHandler implements ListSelectionListener {
            public void valueChanged(ListSelectionEvent e) {
                updateSelectedPathsFromSelectedRows();
            }
        }
    }

    public boolean getSortedColumnAscending() {
        return sortedColumnAscending;
    }

    public int getSortedColumnIndex() {
        return sortedColumnIndex;
    }

    // If expand is true, expands all nodes in the tree.
    // Otherwise, collapses all nodes in the tree.
    public void expandAll(JTree tree, boolean expand) {
        TreeNode root = (TreeNode) tree.getModel().getRoot();

        // Traverse tree from root
        expandAll(tree, new TreePath(root), expand);
    }

    private void expandAll(JTree tree, TreePath parent, boolean expand) {
        // Traverse children
        TreeNode node = (TreeNode) parent.getLastPathComponent();
        if (node.getChildCount() >= 0) {
            for (Enumeration e = node.children(); e.hasMoreElements();) {
                TreeNode n = (TreeNode) e.nextElement();
                TreePath path = parent.pathByAddingChild(n);
                expandAll(tree, path, expand);
            }
        }

        // Expansion or collapse must be done bottom-up
        if (expand) {
            tree.expandPath(parent);
        } else {
            tree.collapsePath(parent);
        }
    }
    
    
    
    public static abstract class MergeSort extends Object {
        protected Object           toSort[];
        protected Object           swapSpace[];

        public void sort(Object array[]) {
        if(array != null && array.length > 1)
        {
            int             maxLength;
      
            maxLength = array.length;
            swapSpace = new Object[maxLength];
            toSort = array;
            this.mergeSort(0, maxLength - 1);
            swapSpace = null;
            toSort = null;
        }
        }

        public abstract int compareElementsAt(int beginLoc, int endLoc);

        protected void mergeSort(int begin, int end) {
        if(begin != end)
        {
            int           mid;

            mid = (begin + end) / 2;
            this.mergeSort(begin, mid);
            this.mergeSort(mid + 1, end);
            this.merge(begin, mid, end);
        }
        }

        protected void merge(int begin, int middle, int end) {
        int           firstHalf, secondHalf, count;

        firstHalf = count = begin;
        secondHalf = middle + 1;
        while((firstHalf <= middle) && (secondHalf <= end))
        {
            if(this.compareElementsAt(secondHalf, firstHalf) < 0)
            swapSpace[count++] = toSort[secondHalf++];
            else
            swapSpace[count++] = toSort[firstHalf++];
        }
        if(firstHalf <= middle)
        {
            while(firstHalf <= middle)
            swapSpace[count++] = toSort[firstHalf++];
        }
        else
        {
            while(secondHalf <= end)
            swapSpace[count++] = toSort[secondHalf++];
        }
        for(count = begin;count <= end;count++)
            toSort[count] = swapSpace[count];
        }
    }
    

    public static class FileSystemModel extends AbstractTreeTableModel implements TreeTableModel {

        // Names of the columns.
        String[] cNames = { "Name", "Size", "Type", "Modified" };

        // Types of the columns.
        Class[] cTypes = { TreeTableModel.class, Integer.class, String.class, Date.class };

        // The the returned file length for directories. 
        final Integer ZERO = new Integer(0);

        public FileSystemModel() {
            super(new FileNode(new File(File.separator)));
        }

        //
        // Some convenience methods. 
        //

        protected File getFile(Object node) {
            FileNode fileNode = ((FileNode) node);
            return fileNode.getFile();
        }

        protected Object[] getChildren(Object node) {
            FileNode fileNode = ((FileNode) node);
            return fileNode.getChildren();
        }

        //
        // The TreeModel interface
        //

        public int getChildCount(Object node) {
            Object[] children = getChildren(node);
            return (children == null) ? 0 : children.length;
        }

        public Object getChild(Object node, int i) {
            return getChildren(node)[i];
        }

        // The superclass's implementation would work, but this is more efficient. 
        public boolean isLeaf(Object node) {
            return getFile(node).isFile();
        }

        //
        //  The TreeTableNode interface. 
        //

        public int getColumnCount() {
            return cNames.length;
        }

        public String getColumnName(int column) {
            return cNames[column];
        }

        public Class getColumnClass(int column) {
            return cTypes[column];
        }

        public Object getValueAt(Object node, int column) {
            File file = getFile(node);
            try {
                switch (column) {
                case 0:
                    return file.getName();
                case 1:
                    return file.isFile() ? new Integer((int) file.length()) : ZERO;
                case 2:
                    return file.isFile() ? "File" : "Directory";
                case 3:
                    return new Date(file.lastModified());
                }
            } catch (SecurityException se) {
            }

            return null;
        }

        public int getColorMetric() {
            // TODO Auto-generated method stub
            return 0;
        }
    }

    /* A FileNode is a derivative of the File class - though we delegate to 
     * the File object rather than subclassing it. It is used to maintain a 
     * cache of a directory's children and therefore avoid repeated access 
     * to the underlying file system during rendering. 
     */
    static class FileNode extends DefaultMutableTreeNode {
        File file;
        Object[] children;

        public FileNode(File file) {
            this.file = file;
        }

        // Used to sort the file names.
        private MergeSort fileMS = new MergeSort() {
            public int compareElementsAt(int a, int b) {
                return ((String) toSort[a]).compareTo((String) toSort[b]);
            }
        };

        /**
         * Returns the the string to be used to display this leaf in the JTree.
         */
        public String toString() {
            return file.getName();
        }

        public File getFile() {
            return file;
        }

        /**
         * Loads the children, caching the results in the children ivar.
         */
        protected Object[] getChildren() {
            if (children != null) {
                return children;
            }
            try {
                String[] files = file.list();
                if (files != null) {
                    fileMS.sort(files);
                    children = new FileNode[files.length];
                    String path = file.getPath();
                    for (int i = 0; i < files.length; i++) {
                        File childFile = new File(path, files[i]);
                        children[i] = new FileNode(childFile);
                    }
                }
            } catch (SecurityException se) {
            }
            return children;
        }
    }

    public static void main(String[] args) {

        JFrame frame = new JFrame("TreeTable");
        JTreeTable treeTable = new JTreeTable(new FileSystemModel(),true);

        frame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent we) {
                System.exit(0);
            }
        });

        frame.getContentPane().add(new JScrollPane(treeTable));
        frame.pack();
        frame.show();
    }

}
