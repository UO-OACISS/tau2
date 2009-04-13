package edu.uoregon.tau.paraprof.treetable;

import java.util.*;

import javax.swing.tree.DefaultMutableTreeNode;

import edu.uoregon.tau.perfdmf.UserEventProfile;
import edu.uoregon.tau.perfdmf.UtilFncs;

public class ContextEventTreeNode extends DefaultMutableTreeNode implements Comparable {

    private List children;
    private String displayName;
    private ContextEventModel model;
    private UserEventProfile userEventProfile;
    private String name;

    public ContextEventTreeNode(String alternateName, ContextEventModel model) {
        this(null, model, alternateName);
    }

    public ContextEventTreeNode(UserEventProfile uep, ContextEventModel model, String alternateName) {
        userEventProfile = uep;
        this.model = model;
        if (uep == null) {
            name = alternateName;
            displayName = UtilFncs.getRightMost(alternateName);
        } else {
            name = uep.getUserEvent().getName();
            displayName = name.substring(0, name.indexOf(":")).trim();
        }
    }

    public UserEventProfile getUserEventProfile() {
        return userEventProfile;
    }

    public List getChildren() {
        checkInitChildren();
        return children;
    }

    private String removeRuns(String str) {
        int loc = str.indexOf("  ");
        while (loc > 0) {
            str = str.substring(0, loc) + str.substring(loc + 1);
            loc = str.indexOf("  ");
        }
        return str;
    }

    private void checkInitChildren() {
        if (children == null) {
            children = new ArrayList();

            Map internalMap = new HashMap();

            for (Iterator it = model.getThread().getUserEventProfiles(); it.hasNext();) {
                UserEventProfile uep = (UserEventProfile) it.next();
                if (uep == null) {
                    continue;
                }
                if (!uep.getUserEvent().isContextEvent()) {
                    continue;
                }

                String path = uep.getName().substring(uep.getName().indexOf(":") + 1).trim();
                path = removeRuns(path);
                if (path.startsWith(name)) {

                    String remain = path.substring(name.length()).trim();
                    if (remain.startsWith("=>")) {
                        remain = remain.substring(2).trim();
                        String child = name + " => " + UtilFncs.getLeftSide(remain);

                        internalMap.put(child, "1");
                    } else if (remain.length() == 0) {
                        ContextEventTreeNode node = new ContextEventTreeNode(uep, model, null);
                        children.add(node);
                    }
                }
            }

            for (Iterator it = internalMap.keySet().iterator(); it.hasNext();) {
                String child = (String) it.next();
                ContextEventTreeNode node = new ContextEventTreeNode(child, model);
                children.add(node);
            }

        }
    }

    public int getNumChildren() {
        checkInitChildren();
        return children.size();
    }

    public void sortChildren() {
        if (children != null) {
            Collections.sort(children);
            for (Iterator it = children.iterator(); it.hasNext();) {
                ContextEventTreeNode node = (ContextEventTreeNode) it.next();
                node.sortChildren();
            }
        }
    }

    public int compareTo(Object o) {
        int result = 0;

        int column = model.getSortColumn(); 
        
        if (column == 0) {
            result = this.toString().compareTo(((ContextEventTreeNode) o).toString());
        } else {
            Double val1 = (Double)model.getValueAt(o, column);
            Double val2 = (Double)model.getValueAt(this, column);

            if (val1 == null && val2 != null) {
                return 1;
            } else if (val1 != null && val2 == null) {
                return -1;
            } else if (val1 == null && val2 == null) {
                result = 0;
            } else {
                result = (int)(val2.doubleValue() - val1.doubleValue());
            }
        }
        if (model.getSortAscending()) {
            return -result;
        }
        return result;
    }

    public String toString() {
        return displayName;
    }
}
