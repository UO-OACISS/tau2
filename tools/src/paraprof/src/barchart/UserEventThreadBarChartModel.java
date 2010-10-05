package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;
import java.awt.event.MouseEvent;
import java.util.List;

import javax.swing.JComponent;

import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UserEvent;
import edu.uoregon.tau.perfdmf.UtilFncs;

public class UserEventThreadBarChartModel extends AbstractBarChartModel {

    private UserEventWindow window;
    private DataSorter dataSorter;
    private Thread thread;

    private List<PPUserEventProfile> list;

    public UserEventThreadBarChartModel(UserEventWindow window, DataSorter dataSorter, Thread thread) {
        this.window = window;
        this.dataSorter = dataSorter;
        this.thread = thread;
        this.reloadData();
    }

    public int getNumRows() {
        return list.size();
    }

    public String getRowLabel(int row) {
        PPUserEventProfile ppUserEventProfile = (PPUserEventProfile) list.get(row);
        return ppUserEventProfile.getUserEventName();
    }

    public String getValueLabel(int row, int subIndex) {
        PPUserEventProfile ppUserEventProfile = (PPUserEventProfile) list.get(row);
        double value = window.getValueType().getValue(ppUserEventProfile.getUserEventProfile(),dataSorter.getSelectedSnapshot());
        return UtilFncs.getOutputString(0, value, ParaProf.defaultNumberPrecision, false);
    }

    public double getValue(int row, int subIndex) {
        PPUserEventProfile ppUserEventProfile = (PPUserEventProfile) list.get(row);
        return window.getValueType().getValue(ppUserEventProfile.getUserEventProfile(),dataSorter.getSelectedSnapshot());
    }

    public Color getValueColor(int row, int subIndex) {
        PPUserEventProfile ppUserEventProfile = (PPUserEventProfile) list.get(row);
        return ppUserEventProfile.getColor();
    }

    public Color getValueHighlightColor(int row, int subIndex) {
        PPUserEventProfile ppUserEventProfile = (PPUserEventProfile) list.get(row);
        if (ppUserEventProfile.getUserEvent() == (window.getPpTrial().getHighlightedUserEvent())) {
            return window.getPpTrial().getColorChooser().getUserEventHighlightColor();
        }
        return null;
    }

    public void fireValueClick(int row, int subIndex, MouseEvent e, JComponent owner) {
        // TODO Auto-generated method stub

    }

    public void fireRowLabelClick(int row, MouseEvent e, JComponent owner) {
        PPUserEventProfile ppUserEventProfile = (PPUserEventProfile) list.get(row);

        UserEvent userEvent = ppUserEventProfile.getUserEvent();
        if (ParaProfUtils.rightClick(e)) {
            ParaProfUtils.handleUserEventClick(window.getPpTrial(), userEvent, owner, e);

        } else {
            FunctionBarChartWindow threadDataWindow = new FunctionBarChartWindow(window.getPpTrial(),
                    ppUserEventProfile.getThread(), null, owner);
            threadDataWindow.setVisible(true);
        }

    }

    public String getValueToolTipText(int row, int subIndex) {
        // TODO Auto-generated method stub
        return null;
    }

    public String getRowLabelToolTipText(int row) {
        // TODO Auto-generated method stub
        return null;
    }

    public String getOtherToolTopText(int row) {
        // TODO Auto-generated method stub
        return null;
    }

    public void reloadData() {
        list = dataSorter.getUserEventProfiles(thread);
        fireModelChanged();
    }


	public DataSorter getDataSorter() {
		return dataSorter;
	}
}
