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
 * A BarChartModel for doing...
 *  1) One function across threads, or 
 *  2) One function across all phases (for one thread).
 * 
 * <P>CVS $Id: FunctionBarChartModel.java,v 1.6 2009/01/23 02:11:11 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.6 $
 */

public class FunctionBarChartModel extends AbstractBarChartModel {

    private List list;

    private FunctionBarChartWindow window;
    private DataSorter dataSorter;
    private Function function;
    private ParaProfTrial ppTrial;

    public FunctionBarChartModel(FunctionBarChartWindow window, DataSorter dataSorter, Function function) {
        this.window = window;
        this.dataSorter = dataSorter;
        this.function = function;
        this.reloadData();
        this.ppTrial = window.getPpTrial();
    }

    public int getNumRows() {
        return list.size();
    }

    public String getRowLabel(int row) {
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) list.get(row);
        String barString;

        if (window.isPhaseDisplay()) {
            barString = UtilFncs.getLeftSide(ppFunctionProfile.getDisplayName());
        } else {
            if (ppFunctionProfile.getNodeID() == -1) {
                barString = "mean";
            } else if (ppFunctionProfile.getNodeID() == -2) {
                barString = "total";
            } else if (ppFunctionProfile.getNodeID() == -3) {
                barString = "std. dev.";
            } else {
                barString = ParaProfUtils.getThreadLabel(ppFunctionProfile.getThread());
            }
        }
        return barString;
    }

    public String getRowValueLabel(int row) {
        // TODO Auto-generated method stub
        return null;
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
            return UtilFncs.getOutputString(0, value, 6) + "%";

        } else {
            return UtilFncs.getOutputString(window.units(), value, ParaProf.defaultNumberPrecision);
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
        return Color.black;
    }

    public void reloadData() {
        if (window.isPhaseDisplay()) {
            list = dataSorter.getFunctionAcrossPhases(function, window.getThread());
        } else {
            list = dataSorter.getFunctionData(function, true, true);
        }
        this.fireModelChanged();
    }

    public String getValueToolTipText(int row, int subIndex) {
        // TODO Auto-generated method stub
        return null;
    }

    public String getRowLabelToolTipText(int row) {
        // TODO Auto-generated method stub
        return null;
    }

    public void fireRowLabelClick(int row, MouseEvent e, JComponent owner) {
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) list.get(row);

        if (window.isPhaseDisplay()) {
            if (ParaProfUtils.rightClick(e)) {
                JPopupMenu popup = ParaProfUtils.createFunctionClickPopUp(ppTrial, ppFunctionProfile.getFunction(),
                        ppFunctionProfile.getThread(), owner);
                popup.show(owner, e.getX(), e.getY());

            } else {
                ppTrial.toggleHighlightedFunction(ppFunctionProfile.getFunction());
            }

        } else {

            if (ParaProfUtils.rightClick(e)) {
                ParaProfUtils.handleThreadClick(window.getPpTrial(), function.getParentPhase(), ppFunctionProfile.getThread(),
                        owner, e);

            } else {
                FunctionBarChartWindow threadDataWindow = new FunctionBarChartWindow(window.getPpTrial(),
                        ppFunctionProfile.getThread(), function.getParentPhase(), owner);
                threadDataWindow.setVisible(true);
            }
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

    public String getOtherToolTopText(int row) {
        // TODO Auto-generated method stub
        return null;
    }

}
