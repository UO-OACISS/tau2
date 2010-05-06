package edu.uoregon.tau.paraprof.treetable;

import java.util.*;

import edu.uoregon.tau.common.treetable.AbstractTreeTableModel;
import edu.uoregon.tau.common.treetable.TreeTableModel;
import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.perfdmf.Thread;

public class ContextEventModel extends AbstractTreeTableModel {

    private static String[] cNames = { "Name", "Total", "NumSamples", "MaxValue", "MinValue", "MeanValue", "Std. Dev." };
    private static Class[] cTypes = { TreeTableModel.class, Double.class, Double.class, Double.class, Double.class, Double.class, Double.class };

    private List roots;

    private ParaProfTrial ppTrial;
    private Thread thread;

    private ContextEventWindow window;

    private int sortColumn;
    private boolean sortAscending;
    DataSorter dataSorter;

    public ContextEventModel(ContextEventWindow window, ParaProfTrial ppTrial, Thread thread, boolean reversedCallPaths) {
        super(null);
        this.ppTrial = ppTrial;
        this.thread = thread;
        this.window = window;

        root = new ContextEventTreeNode("root", this);

        setupData();
    }

    public Thread getThread() {
        return thread;
    }

    private void setupData() {

        roots = new ArrayList();
        dataSorter = new DataSorter(ppTrial);

        // don't ask the thread for its functions directly, since we want group masking to work
        List uepList = dataSorter.getUserEventProfiles(thread);

        Map rootNames = new HashMap();

        if (window.getTreeMode()) {
            for (Iterator it = uepList.iterator(); it.hasNext();) {
                // Find all the rootNames (as strings)
                PPUserEventProfile ppUserEventProfile = (PPUserEventProfile) it.next();
                UserEventProfile uep = ppUserEventProfile.getUserEventProfile();

                if (uep.getUserEvent().isContextEvent()) {
                    String rootName;

                    rootName = UtilFncs.getContextEventRoot(uep.getName());

                    //System.out.println("root = " + rootName);
                    if (rootNames.get(rootNames) == null) {
                        rootNames.put(rootName, "1");
                    }
                }
            }
            for (Iterator it = rootNames.keySet().iterator(); it.hasNext();) {
                // no go through the strings and get the actual functions
                String rootName = (String) it.next();

                ContextEventTreeNode node = new ContextEventTreeNode(rootName, this);
                roots.add(node);
            }

        } else {
            //            for (Iterator it = functionProfileList.iterator(); it.hasNext();) {
            //                PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) it.next();
            //                FunctionProfile fp = ppFunctionProfile.getFunctionProfile();
            //
            //                if (fp != null && ppTrial.displayFunction(fp.getFunction())) {
            //                    String fname = fp.getName();
            //
            //                    TreeTableNode node = new TreeTableNode(fp, this, null);
            //                    roots.add(node);
            //                }
            //
            //            }
        }

        Collections.sort(roots);
        //computeMaximum();
    }

    public String getColumnName(int column) {
        return cNames[column];
    }

    public Class getColumnClass(int column) {
        return cTypes[column];
    }

    public int getColorMetric() {
        // TODO Auto-generated method stub
        return 0;
    }

    public int getColumnCount() {
        return cNames.length;
    }

    public Object getValueAt(Object node, int column) {
        ContextEventTreeNode cnode = (ContextEventTreeNode) node;
        UserEventProfile uep = cnode.getUserEventProfile();
        if (uep == null) {
            return null;
        } else {
            switch (column) {
            case 1:
                if (uep.getName().contains("/s)")) { // rates are ignored for total
                    return null;
                } else {
                    return new Double(uep.getNumSamples(dataSorter.getSelectedSnapshot())*uep.getMeanValue(dataSorter.getSelectedSnapshot()));
                }
            case 2:
                return new Double(uep.getNumSamples(dataSorter.getSelectedSnapshot()));
            case 3:
                return new Double(uep.getMaxValue(dataSorter.getSelectedSnapshot()));
            case 4:
                return new Double(uep.getMinValue(dataSorter.getSelectedSnapshot()));
            case 5:
                return new Double(uep.getMeanValue(dataSorter.getSelectedSnapshot()));
            case 6:
                return new Double(uep.getStdDev(dataSorter.getSelectedSnapshot()));
            default:
                return null;
            }

        }
    }

    public Object getChild(Object parent, int index) {
        if (parent == root) {
            return roots.get(index);
        }

        ContextEventTreeNode node = (ContextEventTreeNode) parent;

        return node.getChildren().get(index);
    }

    public int getChildCount(Object parent) {
        if (parent == root) {
            return roots.size();
        }

        if (window.getTreeMode()) {
            ContextEventTreeNode node = (ContextEventTreeNode) parent;
            return node.getNumChildren();
        } else {
            return 0;
        }
    }

    public int getSortColumn() {
        return sortColumn;
    }

    public void sortColumn(int index, boolean ascending) {
        super.sortColumn(index, ascending);
        sortColumn = index;
        sortAscending = ascending;

        Collections.sort(roots);
        for (Iterator it = roots.iterator(); it.hasNext();) {
            ContextEventTreeNode node = (ContextEventTreeNode) it.next();
            node.sortChildren();
        }
    }

    public boolean getSortAscending() {
        return sortAscending;
    }
}
