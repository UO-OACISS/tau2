package edu.uoregon.tau.paraprof.treetable;

import java.util.*;

import edu.uoregon.tau.dms.dss.DataSource;
import edu.uoregon.tau.dms.dss.Function;
import edu.uoregon.tau.dms.dss.FunctionProfile;
import edu.uoregon.tau.paraprof.DataSorter;
import edu.uoregon.tau.paraprof.PPFunctionProfile;
import edu.uoregon.tau.paraprof.ParaProfTrial;


/**
 * Data model for treetable using callpaths
 *    
 * TODO : ...
 *
 * <P>CVS $Id: CallPathModel.java,v 1.2 2005/06/17 22:13:49 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class CallPathModel extends AbstractTreeTableModel {

    private List roots;
    private edu.uoregon.tau.dms.dss.Thread thread;
    private DataSource dataSource;
    private ParaProfTrial ppTrial;
    private double[] maxValues;

    private int sortColumn;
    private boolean sortAscending;

    private TreeTableWindow window;

    public CallPathModel(TreeTableWindow window, ParaProfTrial ppTrial, edu.uoregon.tau.dms.dss.Thread thread) {
        super(null);
        root = new TreeTableNode(null, this, "root");
        this.window = window;

        dataSource = ppTrial.getDataSource();
        this.thread = thread;
        this.ppTrial = ppTrial;

        setupData();

    }

    private void setupData() {

        roots = new ArrayList();
        DataSorter dataSorter = new DataSorter(ppTrial);
        
        List functionProfileList = dataSorter.getFunctionProfiles(thread);
        
        //List functionProfileList = thread.getFunctionProfiles();

        Map rootNames = new HashMap();

        if (window.getTreeMode()) {
            for (Iterator it = functionProfileList.iterator(); it.hasNext();) {
                //FunctionProfile fp = (FunctionProfile) it.next();
                PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) it.next(); 
                FunctionProfile fp = ppFunctionProfile.getFunctionProfile();

                if (fp != null && fp.isCallPathFunction()) {
                    String fname = fp.getName();
                    int loc = fname.indexOf("=>");
                    String rootName = fname.substring(0, loc).trim();
                    if (rootNames.get(rootNames) == null) {
                        rootNames.put(rootName, "1");
                    }
                }
            }
            for (Iterator it = rootNames.keySet().iterator(); it.hasNext();) {
                String rootName = (String) it.next();
                Function function = dataSource.getFunction(rootName);

                TreeTableNode node;

                FunctionProfile fp = null;
                if (fp == null) {
                    node = new TreeTableNode(null, this, rootName);

                } else {
                    fp = thread.getFunctionProfile(function);
                    node = new TreeTableNode(fp, this, null);
                }

                roots.add(node);
            }

        } else {
            for (Iterator it = functionProfileList.iterator(); it.hasNext();) {
                //FunctionProfile fp = (FunctionProfile) it.next();
                PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) it.next(); 
                FunctionProfile fp = ppFunctionProfile.getFunctionProfile();

                if (fp != null) {
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

                    //maxValue = Math.max(maxValue, fp.getInclusive(0));
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
        // TODO Auto-generated method stub
        if (parent == root) {
            return roots.get(index);
        }

        TreeTableNode node = (TreeTableNode) parent;

        return node.getChildren().get(index);
    }

    public edu.uoregon.tau.dms.dss.Thread getThread() {
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
