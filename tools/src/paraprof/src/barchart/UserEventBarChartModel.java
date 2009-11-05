package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;
import java.awt.event.MouseEvent;
import java.util.List;

import javax.swing.JComponent;

import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.perfdmf.UserEvent;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * A BarChartModel for displaying one User Event over all threads.
 * 
 * <P>CVS $Id: UserEventBarChartModel.java,v 1.7 2009/11/05 09:43:33 khuck Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.7 $
 */
public class UserEventBarChartModel extends AbstractBarChartModel {

    private UserEventWindow window;
    private DataSorter dataSorter;
    private UserEvent userEvent;

    private List list;

    public UserEventBarChartModel(UserEventWindow window, DataSorter dataSorter, UserEvent userEvent) {
        this.window = window;
        this.dataSorter = dataSorter;
        this.userEvent = userEvent;
        this.reloadData();
    }

    public int getNumRows() {
        return list.size();
    }

    public String getRowLabel(int row) {
        PPUserEventProfile ppUserEventProfile = (PPUserEventProfile) list.get(row);
        return ParaProfUtils.getThreadLabel(ppUserEventProfile.getThread());
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
        if (userEvent == (window.getPpTrial().getHighlightedUserEvent())) {
            return window.getPpTrial().getColorChooser().getUserEventHighlightColor();
        }
        return null;
    }

    public void fireValueClick(int row, int subIndex, MouseEvent e, JComponent owner) {
        // TODO we should do something here
    }

    public void fireRowLabelClick(int row, MouseEvent e, JComponent owner) {
        PPUserEventProfile ppUserEventProfile = (PPUserEventProfile) list.get(row);

        ppUserEventProfile.getThread();
        if (ParaProfUtils.rightClick(e)) {
            ParaProfUtils.handleThreadClick(window.getPpTrial(), null, ppUserEventProfile.getThread(), owner, e);

        } else {
            FunctionBarChartWindow threadDataWindow = new FunctionBarChartWindow(window.getPpTrial(),
                    ppUserEventProfile.getThread(), null, owner);
            threadDataWindow.setVisible(true);
        }

    }

    public String getValueToolTipText(int row, int subIndex) {
        return null;
    }

    public String getRowLabelToolTipText(int row) {
        return null;
    }

    public String getOtherToolTopText(int row) {
        return null;
    }

    public void reloadData() {
        list = dataSorter.getUserEventData(userEvent);
        fireModelChanged();
    }


	public DataSorter getDataSorter() {
		return dataSorter;
	}
}
