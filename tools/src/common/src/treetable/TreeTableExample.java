package edu.uoregon.tau.common.treetable;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.util.Date;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.tree.DefaultMutableTreeNode;

public class TreeTableExample {

    public static abstract class MergeSort extends Object {
        protected Object toSort[];
        protected Object swapSpace[];

        public void sort(Object array[]) {
            if (array != null && array.length > 1) {
                int maxLength;

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
            if (begin != end) {
                int mid;

                mid = (begin + end) / 2;
                this.mergeSort(begin, mid);
                this.mergeSort(mid + 1, end);
                this.merge(begin, mid, end);
            }
        }

        protected void merge(int begin, int middle, int end) {
            int firstHalf, secondHalf, count;

            firstHalf = count = begin;
            secondHalf = middle + 1;
            while ((firstHalf <= middle) && (secondHalf <= end)) {
                if (this.compareElementsAt(secondHalf, firstHalf) < 0)
                    swapSpace[count++] = toSort[secondHalf++];
                else
                    swapSpace[count++] = toSort[firstHalf++];
            }
            if (firstHalf <= middle) {
                while (firstHalf <= middle)
                    swapSpace[count++] = toSort[firstHalf++];
            } else {
                while (secondHalf <= end)
                    swapSpace[count++] = toSort[secondHalf++];
            }
            for (count = begin; count <= end; count++)
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
            } catch (SecurityException se) {}

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
            } catch (SecurityException se) {}
            return children;
        }
    }

    public static void main(String[] args) {

        JFrame frame = new JFrame("TreeTable");
        JTreeTable treeTable = new JTreeTable(new FileSystemModel(), true, true);

        frame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent we) {
                System.exit(0);
            }
        });

        frame.getContentPane().add(new JScrollPane(treeTable));
        frame.pack();
        frame.setVisible(true);
    }

}
