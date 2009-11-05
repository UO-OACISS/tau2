package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;
import java.awt.event.MouseEvent;
import java.util.List;

import javax.swing.JComponent;
import javax.swing.JPopupMenu;

import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * A BarChartModel for displaying all functions for one thread.
 * 
 * <P>CVS $Id: ThreadBarChartModel.java,v 1.6 2009/11/05 09:43:32 khuck Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.6 $
 */
public class ThreadBarChartModel extends AbstractBarChartModel {

    private List list;

    private FunctionBarChartWindow window;
    private DataSorter dataSorter;
    private ParaProfTrial ppTrial;
    private PPThread ppThread;
    
    public ThreadBarChartModel(FunctionBarChartWindow window, DataSorter dataSorter, PPThread thread) {
        this.window = window;
        this.dataSorter = dataSorter;
        this.ppThread = thread;
        this.ppTrial = window.getPpTrial();
        this.reloadData();
    }

    public int getNumRows() {
        return list.size();
    }

    public String getRowLabel(int row) {
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) list.get(row);

        if (window.getPhase() != null) {
            // we can't use PPFunctionProfile's getFunctionName since the callpath might be reversed
            // return UtilFncs.getRightSide(ppFunctionProfile.getFunctionName());
            return UtilFncs.getRightSide(ppFunctionProfile.getFunction().getName());
        } else {
            return ppFunctionProfile.getDisplayName();
        }
    }


    public double getValue(int row, int subIndex) {
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) list.get(row);

        return ppFunctionProfile.getValue();
    }

    public String getValueLabel(int row, int subIndex) {
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) list.get(row);
        double value = ppFunctionProfile.getValue();
        if (window.getDataSorter().getValueType() == ValueType.EXCLUSIVE_PERCENT
                || window.getDataSorter().getValueType() == ValueType.INCLUSIVE_PERCENT) {

            //s = (UtilFncs.adjustDoublePresision(value, 4)) + "%";
            return UtilFncs.getOutputString(0, value, 6, ppFunctionProfile.getDataSorter().getSelectedMetric().isTimeDenominator()) + "%";

        } else {
            return UtilFncs.getOutputString(window.units(), value, 6, ppFunctionProfile.getDataSorter().getSelectedMetric().isTimeDenominator());
        }
    }

    public Color getValueColor(int row, int subIndex) {
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) list.get(row);

        return ppFunctionProfile.getColor();
    }

    public Color getValueHighlightColor(int row, int subIndex) {
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) list.get(row);

        Function function = ppFunctionProfile.getFunction();
        if (function == (ppTrial.getHighlightedFunction())) {
            return ppTrial.getColorChooser().getHighlightColor();
        } else if (function.isGroupMember(ppTrial.getHighlightedGroup())) {
            return ppTrial.getColorChooser().getGroupHighlightColor();
        }
        return null;
    }

    public void reloadData() {
        list = ppThread.getSortedFunctionProfiles(dataSorter, false);
        fireModelChanged();
    }

  
    public void fireRowLabelClick(int row, MouseEvent e, JComponent owner) {
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) list.get(row);

        if (ParaProfUtils.rightClick(e)) {
            JPopupMenu popup = ParaProfUtils.createFunctionClickPopUp(ppTrial, ppFunctionProfile.getFunction(),
                    ppFunctionProfile.getThread(), owner);
            popup.show(owner, e.getX(), e.getY());

        } else {
            ppTrial.toggleHighlightedFunction(ppFunctionProfile.getFunction());
        }
    }

    public void fireValueClick(int row, int subIndex, MouseEvent e, JComponent owner) {
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) list.get(row);
        Function function = ppFunctionProfile.getFunction();
        if (ParaProfUtils.rightClick(e)) {

            JPopupMenu popup = ParaProfUtils.createFunctionClickPopUp(window.getPpTrial(), function,
                    ppFunctionProfile.getThread(), owner);
            popup.show(owner, e.getX(), e.getY());
        } else {
            window.getPpTrial().toggleHighlightedFunction(function);
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


	public DataSorter getDataSorter() {
		return dataSorter;
	}
}
