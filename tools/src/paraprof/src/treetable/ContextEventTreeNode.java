package edu.uoregon.tau.paraprof.treetable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.tree.DefaultMutableTreeNode;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.perfdmf.UserEventProfile;
import edu.uoregon.tau.perfdmf.UtilFncs;

public class ContextEventTreeNode extends DefaultMutableTreeNode implements Comparable<ContextEventTreeNode> {

    /**
	 * 
	 */
	private static final long serialVersionUID = 8704255864552442447L;
	private List<ContextEventTreeNode> children;
    private String displayName;
    private ContextEventModel model;
    private UserEventProfile userEventProfile;
    private String name;

    public ContextEventTreeNode(String alternateName, ContextEventModel model) {
        this(null, model, alternateName);
    }

    public ContextEventTreeNode(UserEventProfile uep, ContextEventModel model, String alternateName) {
    	this.setUserObject(uep);
        userEventProfile = uep;
        this.model = model;
        if (uep == null) {
            name = alternateName.trim();
            displayName = UtilFncs.getRightMost(alternateName);
        } else {
			name = ParaProfUtils.getUserEventDisplayName(uep.getUserEvent())
					.trim();
			if (name.indexOf(SPACE_COLON_SPACE) != -1) {
                // remove the path
				displayName = name.substring(0, name.lastIndexOf(SPACE_COLON_SPACE)).trim();
            } else {
                displayName = name;
            }
        }
    }

    public UserEventProfile getUserEventProfile() {
        return userEventProfile;
    }

    public List<ContextEventTreeNode> getChildren() {
        checkInitChildren();
        return children;
    }

    private static final String SPACE_COLON_SPACE=" : ";
    private static final String ARROW="=>";
    private static final String SPACE_ARROW_SPACE=" => ";
    //private static final String ONE="1";
    private void checkInitChildren() {
        if (children == null) {
            children = new ArrayList<ContextEventTreeNode>();

            //Map<String, String> internalMap = new HashMap<String, String>();
            Set<String> internalSet = new HashSet<String>();
            // search all the user events for this node and find our children 
            for (Iterator<UserEventProfile> it = model.getThread().getUserEventProfiles(); it.hasNext();) {
                UserEventProfile uep = it.next();
                if (uep == null) {
                    continue;
                }
                if (!uep.getUserEvent().isContextEvent()) {
                    continue;
                }

				String uename = ParaProfUtils.getUserEventDisplayName(uep
						.getUserEvent());
				String path = uename.substring(uename.lastIndexOf(SPACE_COLON_SPACE) + 2)
						.trim();
				
                if (path.startsWith(name)) {
                    String remain = path.substring(name.length()).trim();
                    if (remain.startsWith(ARROW)) {
                        remain = remain.substring(2).trim();
                        String child = name + SPACE_ARROW_SPACE + UtilFncs.getLeftSide(remain);

                        internalSet.add(child);//.put(child, ONE);
                    } else if (remain.length() == 0) {
                        ContextEventTreeNode node = new ContextEventTreeNode(uep, model, null);
                        children.add(node);
                    }
                }
            }

            for (Iterator<String> it = internalSet.iterator(); it.hasNext();) {
                String child = it.next();
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
            for (Iterator<ContextEventTreeNode> it = children.iterator(); it.hasNext();) {
                ContextEventTreeNode node = it.next();
                node.sortChildren();
            }
        }
    }

    public String toString() {
        return displayName;
    }
    
    public String getName(){
    	return name;
    }

    public int compareTo(ContextEventTreeNode o) {
        int result = 0;

        int column = model.getSortColumn();

        if (column == 0) {
            result = this.toString().compareTo(((ContextEventTreeNode) o).toString());
        } else {
            Double val1 = (Double) model.getValueAt(o, column);
            Double val2 = (Double) model.getValueAt(this, column);

            if (val1 == null && val2 != null) {
                return 1;
            } else if (val1 != null && val2 == null) {
                return -1;
            } else if (val1 == null && val2 == null) {
                result = 0;
            } else {
                result = (int) (val2.doubleValue() - val1.doubleValue());
            }
        }
        if (model.getSortAscending()) {
            return -result;
        }
        return result;
    }
}
