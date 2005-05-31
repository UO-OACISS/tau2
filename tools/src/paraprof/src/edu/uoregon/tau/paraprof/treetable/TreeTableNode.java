package edu.uoregon.tau.paraprof.treetable;

import java.awt.Color;
import java.awt.Component;
import java.awt.Graphics;
import java.util.*;

import javax.swing.Icon;
import javax.swing.tree.DefaultMutableTreeNode;

import edu.uoregon.tau.dms.dss.FunctionProfile;
import edu.uoregon.tau.dms.dss.UtilFncs;
import edu.uoregon.tau.paraprof.ColorBar;
import edu.uoregon.tau.paraprof.ParaProf;

public class TreeTableNode extends DefaultMutableTreeNode implements Comparable {
    private List children;
    private FunctionProfile functionProfile;
    private String displayName;
    private boolean expanded;
    private Icon icon;
    private CallPathModel model;
    private String alternateName;

    /**
     * @param functionProfile   FunctionProfile for this node, null for nodes with no associated FunctionProfile
     * @param model             The associated CallPathModel
     * @param alternateName     String to display for nodes with no associated FunctionProfile
     */
    TreeTableNode(FunctionProfile functionProfile, CallPathModel model, String alternateName) {
        this.functionProfile = functionProfile;
        this.model = model;
        this.alternateName = alternateName;
        if (functionProfile != null) {
            displayName = functionProfile.getName();

            if (model.getWindow().getTreeMode()) {
                int loc = displayName.lastIndexOf("=>");
                if (loc != -1) {
                    displayName = displayName.substring(loc + 2).trim();
                }
            }
        } else {
            displayName = alternateName;
        }
    }

    public FunctionProfile getFunctionProfile() {
        return functionProfile;
    }

    public List getChildren() {
        checkInitChildren();
        return children;
    }

    public int getNumChildren() {
        checkInitChildren();
        return children.size();
    }

    private void checkInitChildren() {
        if (children == null) {
            children = new ArrayList();

            List functionProfileList = model.getThread().getFunctionProfiles();

            for (Iterator it = functionProfileList.iterator(); it.hasNext();) {
                FunctionProfile fp = (FunctionProfile) it.next();
                if (fp == this.functionProfile)
                    continue;

                if (fp != null && fp.isCallPathFunction()) {
                    String fname = fp.getName();

                    String thisName = alternateName;
                    if (functionProfile != null) {
                        thisName = functionProfile.getName();
                    }

                    int loc = fname.indexOf(thisName);

                    if (loc == 0) {
                        String remainder = fname.substring(thisName.length()).trim();

                        int loc2 = remainder.lastIndexOf("=>");
                        if (loc2 == 0) {
                            TreeTableNode node = new TreeTableNode(fp, model, null);
                            children.add(node);
                        }
                    }
                }
            }
        }

        Collections.sort(children);
    }



    public String toString() {
        return displayName;
    }

    public boolean getExpanded() {
        return expanded;
    }

    public void setExpanded(boolean expanded) {
        this.expanded = expanded;
        icon = null;
    }

    
    public Color getColor(int metricID) {
        if (!model.getWindow().getTreeMode()) {
            return null;
        }

        if (functionProfile != null && model.getMaxValues()[metricID] != 0) {
            double value;
            if (expanded) {
                value = functionProfile.getExclusive(metricID);
            } else {
                value = functionProfile.getInclusive(metricID);
            }

            Color color = ColorBar.getColor((float) (value / model.getMaxValues()[metricID]));
            return color;
        }
        return null;
    }

    public void sortChildren() {
        if (children != null) {
            Collections.sort(children);
            for (Iterator it = children.iterator(); it.hasNext();) {
                TreeTableNode treeTableNode = (TreeTableNode) it.next();
                treeTableNode.sortChildren();
            }
        }
    }

    public int compareTo(Object o) {

        int result;
        if (model.getSortColumn() == 0) {
            result = this.toString().compareTo(((TreeTableNode) o).toString());
        } else {
            TreeTableColumn column = (TreeTableColumn) model.getWindow().getColumns().get(model.getSortColumn() - 1);


            Comparable a = (Comparable) column.getValueFor(this, true);
            Comparable b = (Comparable) column.getValueFor((TreeTableNode) o, true);

            result = a.compareTo(b);
        }
        if (model.getSortAscending()) {
            return -result;
        }
        return result;
    }


    public CallPathModel getModel() {
        return model;
    }

}
