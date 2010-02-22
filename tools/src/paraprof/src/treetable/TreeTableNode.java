package edu.uoregon.tau.paraprof.treetable;

import java.awt.Color;
import java.util.*;

import javax.swing.tree.DefaultMutableTreeNode;

import edu.uoregon.tau.common.AlphanumComparator;
import edu.uoregon.tau.paraprof.ColorBar;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.perfdmf.FunctionProfile;

/**
 * Represents a node in the TreeTable
 *    
 * TODO : ...
 *
 * <P>CVS $Id: TreeTableNode.java,v 1.12 2010/02/22 20:01:17 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.12 $
 */
public class TreeTableNode extends DefaultMutableTreeNode implements Comparable {
    private List children;
    private FunctionProfile functionProfile;
    private String displayName;
    private boolean expanded;
    private CallPathModel model;
    private String alternateName;
    private static AlphanumComparator cmp = new AlphanumComparator();

    /**
     * @param functionProfile   FunctionProfile for this node, null for nodes with no associated FunctionProfile
     * @param model             The associated CallPathModel
     * @param alternateName     String to display for nodes with no associated FunctionProfile
     */
    public TreeTableNode(FunctionProfile functionProfile, CallPathModel model, String alternateName) {
        this.functionProfile = functionProfile;
        this.model = model;
        this.alternateName = alternateName;

        if (functionProfile != null) {
            if (model.getWindow().getTreeMode()) {
                if (model.getReversedCallPaths()) {
                    displayName = ParaProfUtils.getReversedLeafDisplayName(functionProfile.getFunction());
                } else {
                    displayName = ParaProfUtils.getLeafDisplayName(functionProfile.getFunction());
                }
            } else {
                displayName = ParaProfUtils.getDisplayName(functionProfile.getFunction());
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

            boolean foundAsInternal = false;
            // If we are B, this will be true if there is "A => B => C", but there is
            // no "A => B".  This should never happen with TAU, but will happen all the time
            // with multi-level mpiP data.

            Map potentialChildren = new HashMap();

            boolean foundActual = false;

            for (Iterator it = functionProfileList.iterator(); it.hasNext();) {
                FunctionProfile fp = (FunctionProfile) it.next();
                if (fp == this.functionProfile)
                    continue;

                if (fp != null && fp.isCallPathFunction()) {
                    String fname;
                    String pathDelimeter;
                    if (model.getReversedCallPaths()) {
                        fname = fp.getFunction().getReversedName();
                        pathDelimeter = "<=";
                    } else {
                        fname = fp.getName();
                        pathDelimeter = "=>";
                    }

                    // For mpiP (and possibly others), there will be nodes that do not have
                    // a FunctionProfile associated with them, use the alternateName instead
                    String thisName = alternateName;
                    if (functionProfile != null) {
                        if (model.getReversedCallPaths()) {
                            thisName = functionProfile.getFunction().getReversedName();
                        } else {
                            thisName = functionProfile.getName();
                        }
                    }

                    // thisname = "main"
                    int loc = fname.indexOf(thisName);

                    // fname = "main => a => b => MPI_Send"
                    // want "main => a"

                    if (loc == 0) {
                        String remainder = fname.substring(thisName.length()).trim();

                        int loc2 = remainder.lastIndexOf(pathDelimeter);
                        if (loc2 == 0) {
                            foundActual = true;
                            TreeTableNode node = new TreeTableNode(fp, model, null);
                            children.add(node);
                        }
                    } else if (loc != -1) {
                        int loc2 = fname.indexOf(pathDelimeter, loc + thisName.length());

                        int loc3 = fname.indexOf(pathDelimeter, loc2 + 1);

                        if (loc2 != -1) {
                            foundAsInternal = true;

                            if (loc3 == -1) {
                                potentialChildren.put(fp, new Object());
                            } else {

                                if (model.getReversedCallPaths()) {
                                    // we might have the following:
                                    // MPI_Put_attr <= MPI_Init <= MAIN
                                    // There will be no "MPI_Put_attr <= MPI_Init"
                                    // so we use this one instead
                                    //potentialChildren.put(fp, new Object());
                                    potentialChildren.put(fname.substring(0, loc3), new Object());
                                } else {
                                    potentialChildren.put(fname.substring(0, loc3), new Object());
                                }
                            }
                        }

                    }

                }
            }

            if (!foundActual && foundAsInternal) {
                for (Iterator it = potentialChildren.keySet().iterator(); it.hasNext();) {
                    Object obj = it.next();
                    TreeTableNode node;

                    if (obj instanceof String) {
                        node = new TreeTableNode(null, model, (String) obj);
                    } else {
                        node = new TreeTableNode((FunctionProfile) obj, model, null);
                    }
                    children.add(node);
                }
            }
        }

        try {
            Collections.sort(children);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String toString() {
        return displayName;
    }

    public boolean getExpanded() {
        return expanded;
    }

    public void setExpanded(boolean expanded) {
        this.expanded = expanded;
    }

    public double getColorValue(int metricID, boolean expanded) {
        if (!model.getWindow().getTreeMode()) {
            return -1;
        }

        if (functionProfile == null) {
            double value = 0;
            if (expanded) {
                value = 0;
            } else {
                for (Iterator it = getChildren().iterator(); it.hasNext();) {
                    TreeTableNode child = (TreeTableNode) it.next();
                    value += child.getColorValue(metricID, false);
                }
            }
            return value;
        }

        if (functionProfile != null && model.getMaxValues()[metricID] != 0) {
            double value;
            if (expanded) {
                value = functionProfile.getExclusive(metricID);
            } else {
                value = functionProfile.getInclusive(metricID);
            }

            return value;
        }
        return 0;
    }

    public Color getColor(int metricID) {
        double value = getColorValue(metricID, expanded);
        if (value == -1) {
            return null;
        } else {
            Color color = ColorBar.getColor((float) (value / model.getMaxValues()[metricID]));
            return color;
        }
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

        int result = -99;
        if (model.getSortColumn() == 0) {
            // Compare with the alphanumeric comparator
            result = cmp.compare(this.toString(), o.toString());
        } else {
            TreeTableColumn column = (TreeTableColumn) model.getWindow().getColumns().get(model.getSortColumn() - 1);

            Comparable a = (Comparable) column.getValueFor(this, true);
            Comparable b = (Comparable) column.getValueFor((TreeTableNode) o, true);
            try {
                result = a.compareTo(b);

            } catch (Exception e) {
                e.printStackTrace();
            }
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
