package edu.uoregon.tau.paraprof.treetable;

import edu.uoregon.tau.common.treetable.AbstractTreeTableModel;
import edu.uoregon.tau.paraprof.ParaProfTrial;

public class ContextEventModel extends AbstractTreeTableModel {

    public ContextEventModel(ContextEventWindow window, ParaProfTrial ppTrial, Thread thread,
            boolean reversedCallPaths) {
        super(null);
    }
    
    
    public int getColorMetric() {
        // TODO Auto-generated method stub
        return 0;
    }

    public int getColumnCount() {
        // TODO Auto-generated method stub
        return 0;
    }

    public String getColumnName(int column) {
        // TODO Auto-generated method stub
        return null;
    }

    public Object getValueAt(Object node, int column) {
        // TODO Auto-generated method stub
        return null;
    }

    public Object getChild(Object parent, int index) {
        // TODO Auto-generated method stub
        return null;
    }

    public int getChildCount(Object parent) {
        // TODO Auto-generated method stub
        return 0;
    }

    
    
}
