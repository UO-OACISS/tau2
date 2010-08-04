package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;
import java.awt.event.MouseEvent;
import java.util.List;

import javax.swing.JComponent;

import edu.uoregon.tau.paraprof.DataSorter;
import edu.uoregon.tau.paraprof.ParaProfTrial;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.paraprof.SnapshotThreadWindow;
import edu.uoregon.tau.paraprof.util.ObjectFilter;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Snapshot;
import edu.uoregon.tau.perfdmf.Thread;


public class ThreadSnapshotBarChartModel extends AbstractBarChartModel {
    //private SnapshotThreadWindow window;
    private DataSorter dataSorter;
    private ParaProfTrial ppTrial;

    private List<Object> snapshots;
    
    private Thread thread;
    private ObjectFilter filter;
    
    private List<FunctionProfile> list;
    
    public ThreadSnapshotBarChartModel(SnapshotThreadWindow window, DataSorter dataSorter, ParaProfTrial ppTrial, Thread thread) {
        //this.window = window;
        this.dataSorter = dataSorter;
        this.ppTrial = ppTrial;
        this.thread = thread;
        filter = new ObjectFilter(thread.getSnapshots());
        //filter.hide(thread.getSnapshots().get(0));
        //filter.hide(thread.getSnapshots().get(thread.getSnapshots().size()-1));
        snapshots = filter.getFilteredObjects();
        
        list = dataSorter.getBasicFunctionProfiles(thread);
    }
    
    
    public int getSubSize() {
        //return thread.getFunctionProfiles().size();
        return list.size();
    }
    public void fireRowLabelClick(int row, MouseEvent e, JComponent owner) {
        Snapshot snapshot = ((Snapshot)snapshots.get(row));

        ParaProfUtils.handleSnapshotClick(ppTrial, thread, snapshot, owner, e);
        System.out.println("right click on " + snapshot);
    }

    public void fireValueClick(int row, int subIndex, MouseEvent e, JComponent owner) {
        // TODO Auto-generated method stub

    }

    public int getNumRows() {
        return snapshots.size();
    }

    public String getOtherToolTopText(int row) {
        // TODO Auto-generated method stub
        return "asdf";
    }

    public String getRowLabel(int row) {
        return ((Snapshot)snapshots.get(row)).toString();
        //return "asdf";
    }

    public String getRowLabelToolTipText(int row) {
        return "Snapshot: " + ((Snapshot)snapshots.get(row)).toString();
    }

    public double getValue(int row, int subIndex) {
        //FunctionProfile fp = (FunctionProfile) thread.getFunctionProfiles().get(subIndex);
        FunctionProfile fp = list.get(subIndex);
        if (fp == null) {
            return 0;
        }

        Snapshot snapshot = ((Snapshot)snapshots.get(row));

        int snapshotID = snapshot.getID();
        
        int differential = 1;
        
        if (differential == 1 && snapshotID != 0) {
            return fp.getExclusive(snapshotID, ppTrial.getDefaultMetric().getID()) - fp.getExclusive(snapshotID-1, ppTrial.getDefaultMetric().getID());
        } else {
            return fp.getExclusive(snapshotID, ppTrial.getDefaultMetric().getID());
        }
    }

    public Color getValueColor(int row, int subIndex) {
        //FunctionProfile fp = (FunctionProfile) thread.getFunctionProfiles().get(subIndex);
        FunctionProfile fp = list.get(subIndex);
        if (fp == null) {
            return Color.black;
        }
        
        return fp.getFunction().getColor();
    }

    public Color getValueHighlightColor(int row, int subIndex) {
        // TODO Auto-generated method stub
        return null;
    }

    public String getValueLabel(int row, int subIndex) {
        // TODO Auto-generated method stub
        return "asdf";
    }

    public String getValueToolTipText(int row, int subIndex) {
        //FunctionProfile fp = (FunctionProfile) thread.getFunctionProfiles().get(subIndex);
        FunctionProfile fp = list.get(subIndex);
        if (fp == null) {
            return "";
        }
        return fp.getName();
    }

    public void reloadData() {
        // TODO Auto-generated method stub
        
       
        

    }

	public DataSorter getDataSorter() {
		return dataSorter;
	}

}
