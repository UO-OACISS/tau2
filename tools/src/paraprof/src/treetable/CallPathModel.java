package edu.uoregon.tau.paraprof.treetable;

import java.util.*;

import edu.uoregon.tau.paraprof.DataSorter;
import edu.uoregon.tau.paraprof.PPFunctionProfile;
import edu.uoregon.tau.paraprof.ParaProfTrial;
import edu.uoregon.tau.perfdmf.*;


/**
 * Data model for treetable using callpaths
 *    
 * TODO : ...
 *
 * <P>CVS $Id: CallPathModel.java,v 1.1 2005/09/26 21:12:49 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public class CallPathModel extends AbstractTreeTableModel {

    private List roots;
    private edu.uoregon.tau.perfdmf.Thread thread;
    private DataSource dataSource;
    private ParaProfTrial ppTrial;
    private double[] maxValues;

    private int sortColumn;
    private boolean sortAscending;

    private TreeTableWindow window;

    public CallPathModel(TreeTableWindow window, ParaProfTrial ppTrial, edu.uoregon.tau.perfdmf.Thread thread) {
        super(null);
        this.window = window;
        root = new TreeTableNode(null, this, "root");
        
        dataSource = ppTrial.getDataSource();
        this.thread = thread;
        this.ppTrial = ppTrial;

        setupData();

    }

    private void setupData() {

        roots = new ArrayList();
        DataSorter dataSorter = new DataSorter(ppTrial);

        // don't ask the thread for its functions directly, since we want group masking to work
        List functionProfileList = dataSorter.getFunctionProfiles(thread);

        Map rootNames = new HashMap();

        if (window.getTreeMode()) {
            for (Iterator it = functionProfileList.iterator(); it.hasNext();) {
                // Find all the rootNames (as strings)
                PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) it.next(); 
                FunctionProfile fp = ppFunctionProfile.getFunctionProfile();

                if (fp != null && fp.isCallPathFunction()) {
                    String rootName = UtilFncs.getLeftSide(fp.getName());
                    if (rootNames.get(rootNames) == null) {
                        rootNames.put(rootName, "1");
                    }
                }
            }
            for (Iterator it = rootNames.keySet().iterator(); it.hasNext();) {
                // no go through the strings and get the actual functions
                String rootName = (String) it.next();
                Function function = dataSource.getFunction(rootName);

                TreeTableNode node;

                if (function == null) {
                    node = new TreeTableNode(null, this, rootName);
                } else {
                    FunctionProfile fp = thread.getFunctionProfile(function);
                    node = new TreeTableNode(fp, this, null);
                }

                roots.add(node);
            }

        } else {
            for (Iterator it = functionProfileList.iterator(); it.hasNext();) {
                PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) it.next(); 
                FunctionProfile fp = ppFunctionProfile.getFunctionProfile();

                if (fp != null && ppTrial.displayFunction(fp.getFunction())) {
                    String fname = fp.getName();

                    TreeTableNode node = new TreeTableNode(fp, this, null);
                    roots.add(node);
                }

            }
        }

        Collections.sort(roots);
        computeMaximum();
    }

    public void computeMaximum() {
        int numMetrics = window.getPPTrial().getNumberOfMetrics();
        maxValues = new double[numMetrics];
        for (Iterator it = roots.iterator(); it.hasNext();) {
            TreeTableNode treeTableNode = (TreeTableNode) it.next();

            if (treeTableNode.getFunctionProfile() != null) {

                for (int i = 0; i < numMetrics; i++) {
                    // there are two ways to do this (cube brings up a window to ask you which way you 
                    // want to compute the max value for the color)

                    //maxValue[i] = Math.max(maxValue[i], fp.getInclusive(i));
                    maxValues[i] += treeTableNode.getFunctionProfile().getInclusive(i);

                }
            }
        }

    }

    public int getColumnCount() {
        return 1 + window.getColumns().size();
    }

    public String getColumnName(int column) {
        if (column == 0) {
            return "Name";
        }
        return window.getColumns().get(column - 1).toString();
    }

    public Object getValueAt(Object node, int column) {

        if (node == root) {
            return null;
        }

        TreeTableNode treeTableNode = (TreeTableNode) node;

        return treeTableNode;
    }

    /**
     * Returns the class for the particular column.
     */
    public Class getColumnClass(int column) {
        if (column == 0)
            return TreeTableModel.class;
        return window.getColumns().get(column - 1).getClass();
    }

    public int getChildCount(Object parent) {
        if (parent == root) {
            return roots.size();
        }

        if (window.getTreeMode()) {
            TreeTableNode node = (TreeTableNode) parent;
            return node.getNumChildren();
        } else {
            return 0;
        }
    }

    public Object getChild(Object parent, int index) {
        if (parent == root) {
            return roots.get(index);
        }

        TreeTableNode node = (TreeTableNode) parent;

        return node.getChildren().get(index);
    }

    public edu.uoregon.tau.perfdmf.Thread getThread() {
        return thread;
    }

    public double[] getMaxValues() {
        return maxValues;
    }

    public int getSortColumn() {
        return sortColumn;
    }

    public boolean getSortAscending() {
        return sortAscending;
    }

    public void sortColumn(int index, boolean ascending) {
        super.sortColumn(index, ascending);
        sortColumn = index;
        sortAscending = ascending;

        Collections.sort(roots);
        for (Iterator it = roots.iterator(); it.hasNext();) {
            TreeTableNode treeTableNode = (TreeTableNode) it.next();
            treeTableNode.sortChildren();
        }

    }

    public TreeTableWindow getWindow() {
        return window;
    }

    public ParaProfTrial getPPTrial() {
        return ppTrial;
    }

}
