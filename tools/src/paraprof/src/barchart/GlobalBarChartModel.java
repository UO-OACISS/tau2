package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;
import java.awt.event.MouseEvent;
import java.util.List;

import javax.swing.JComponent;
import javax.swing.JPopupMenu;

import edu.uoregon.tau.common.Common;
import edu.uoregon.tau.paraprof.DataSorter;
import edu.uoregon.tau.paraprof.FunctionBarChartWindow;
import edu.uoregon.tau.paraprof.FunctionOrdering;
import edu.uoregon.tau.paraprof.GlobalDataWindow;
import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.paraprof.ParaProfTrial;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * A BarChartModel for doing the GlobalDataWindow
 * 
 * <P>CVS $Id: GlobalBarChartModel.java,v 1.17 2009/11/05 09:43:32 khuck Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.17 $
 */

public class GlobalBarChartModel extends AbstractBarChartModel {
    private GlobalDataWindow window;
    private DataSorter dataSorter;
    private ParaProfTrial ppTrial;

    //private List threads = new ArrayList();

    private FunctionOrdering functionOrder;
    private List<Thread> theThreads;

    public GlobalBarChartModel(GlobalDataWindow window, DataSorter dataSorter, ParaProfTrial ppTrial) {
        this.window = window;
        this.dataSorter = dataSorter;
        this.ppTrial = ppTrial;
    }

    public int getNumRows() {
        if (theThreads == null) {
            return 0;
        } else {
            //        return threads.size();
            return theThreads.size();
        }
    }

    public int getSubSize() {
        //        if (threads != null && threads.size() >= 1) {
        //            return ((List) threads.get(0)).size();
        //        } else {
        //            return 0;
        //        }
        if (functionOrder != null) {
            return functionOrder.getOrder().length;
        } else {
            return 0;
        }
    }

    private static String getName(Thread thread) {
        return ParaProfUtils.getThreadLabel(thread);
    }

    //    public String getRowLabel(int row) {
    //        List fpList = (List) threads.get(row);
    //        for (int i=0; i<fpList.size(); i++) {
    //            if (fpList.get(i) != null) {
    //                Thread thread = ((FunctionProfile)fpList.get(i)).getThread();
    //                return getName(thread);
    //            }
    //        }
    //        return "";
    //    }

    public String getRowLabel(int row) {
        Thread thread = theThreads.get(row);
        return getName(thread);
    }

    public String getValueLabel(int row, int subIndex) {
        return "";
    }

    //    public double getValue(int row, int subIndex) {
    //        List fpList = (List) threads.get(row);
    //        if (subIndex >= fpList.size()) {
    //            return -1;
    //        }
    //        FunctionProfile fp = (FunctionProfile) fpList.get(subIndex);
    //        if (fp == null) {
    //            return -1;
    //        } else {
    //            return dataSorter.getValueType().getValue(fp, dataSorter.getSelectedMetricID());
    //        }
    //    }

    public double getValue(int row, int subIndex) {
        Thread thread = theThreads.get(row);
        Function function = functionOrder.getOrder()[subIndex];
        FunctionProfile fp = thread.getFunctionProfile(function);
        if (fp == null) {
            return -1;
        } else {
            return dataSorter.getValueType().getValue(fp, dataSorter.getSelectedMetric(), dataSorter.getSelectedSnapshot());
        }
    }

    //    public Color getValueColor(int row, int subIndex) {
    //        List fpList = (List) threads.get(row);
    //        if (subIndex >= fpList.size()) {
    //            return null;
    //        }
    //
    //        FunctionProfile fp = (FunctionProfile) fpList.get(subIndex);
    //        if (fp == null) {
    //            return null;
    //        } else {
    //            return fp.getFunction().getColor();
    //        }
    //    }
    public Color getValueColor(int row, int subIndex) {
        Thread thread = theThreads.get(row);
        Function function = functionOrder.getOrder()[subIndex];
        FunctionProfile fp = thread.getFunctionProfile(function);
        if (fp == null) {
            return Color.black;
        } else {
            return fp.getFunction().getColor();
        }
    }

    //    public Color getValueHighlightColor(int row, int subIndex) {
    //        List fpList = (List) threads.get(row);
    //        if (subIndex >= fpList.size()) {
    //            return null;
    //        }
    //
    //        FunctionProfile fp = (FunctionProfile) fpList.get(subIndex);
    //        Function function = fp.getFunction();
    //        if (function == (ppTrial.getHighlightedFunction())) {
    //            return ppTrial.getColorChooser().getHighlightColor();
    //        } else if (function.isGroupMember(ppTrial.getHighlightedGroup())) {
    //            return ppTrial.getColorChooser().getGroupHighlightColor();
    //        }
    //        return null;
    //    }
    public Color getValueHighlightColor(int row, int subIndex) {
        //Thread thread = theThreads.get(row);
        Function function = functionOrder.getOrder()[subIndex];
        //FunctionProfile fp = thread.getFunctionProfile(function);
        if (function == (ppTrial.getHighlightedFunction())) {
            return ppTrial.getColorChooser().getHighlightColor();
        } else if (function.isGroupMember(ppTrial.getHighlightedGroup())) {
            return ppTrial.getColorChooser().getGroupHighlightColor();
        }
        return null;
    }

    public void fireValueClick(int row, int subIndex, MouseEvent e, JComponent owner) {
        Thread thread = theThreads.get(row);
        Function function = functionOrder.getOrder()[subIndex];
        FunctionProfile fp = thread.getFunctionProfile(function);
        //       List fpList = (List) threads.get(row);
        //        final FunctionProfile fp = (FunctionProfile) fpList.get(subIndex);
        if (ParaProfUtils.rightClick(e)) { // Bring up context menu
            JPopupMenu popup = ParaProfUtils.createFunctionClickPopUp(ppTrial, fp.getFunction(), fp.getThread(), owner);

            popup.show(owner, e.getX(), e.getY());
        } else {
            ppTrial.setHighlightedFunction(fp.getFunction());
            FunctionBarChartWindow functionDataWindow = new FunctionBarChartWindow(ppTrial, fp.getFunction(), owner);
            functionDataWindow.setVisible(true);
        }
    }

    public void fireRowLabelClick(int row, MouseEvent e, JComponent owner) {
        //        List fpList = (List) threads.get(row);
        //        Thread thread = ((FunctionProfile)fpList.get(0)).getThread();
        Thread thread = theThreads.get(row);

        if (ParaProfUtils.rightClick(e)) { // Bring up context menu
            ParaProfUtils.handleThreadClick(ppTrial, window.getPhase(), thread, owner, e);
        } else {
            FunctionBarChartWindow threadDataWindow = new FunctionBarChartWindow(ppTrial, thread, window.getPhase(), owner);
            threadDataWindow.setVisible(true);
        }

    }

    // handles reversed callpaths
    private static String getDisplayName(FunctionProfile functionProfile) {
        if (ParaProf.preferences.getReversedCallPaths()) {
            return functionProfile.getFunction().getReversedName();
        } else {
            return functionProfile.getFunction().getName();
        }
    }

    public String getValueToolTipText(int row, int subIndex) {
        //        List fpList = (List) threads.get(row);
        //        FunctionProfile fp = (FunctionProfile) fpList.get(subIndex);
        Thread thread = theThreads.get(row);
        Function function = functionOrder.getOrder()[subIndex];
        FunctionProfile fp = thread.getFunctionProfile(function);

        String name;
        if (ppTrial.getDataSource().getPhasesPresent()) {

            //return "Other Patches";

            // we can't use PPFunctionProfile's getFunctionName since the callpath might be reversed
            //return UtilFncs.getRightSide(ppFunctionProfile.getFunctionName());
            name = UtilFncs.getRightSide(fp.getFunction().getName());
        } else {
            //Return the name of the function
            name = getDisplayName(fp);
        }

        String metricName = dataSorter.getSelectedMetric().getName();

        Metric metric = dataSorter.getSelectedMetric();

        String unitsString = UtilFncs.getUnitsString(window.units(), dataSorter.isTimeMetric(), false);

        String exclusiveValue = UtilFncs.getOutputString(window.units(), fp.getExclusive(metric.getID()),
                ParaProf.defaultNumberPrecision, metric.isTimeDenominator());
        String inclusiveValue = UtilFncs.getOutputString(window.units(), fp.getInclusive(metric.getID()),
                ParaProf.defaultNumberPrecision, metric.isTimeDenominator());

        String exclusive = "<br>Exclusive " + metricName + ": " + exclusiveValue + " " + unitsString;
        String inclusive = "<br>Inclusive " + metricName + ": " + inclusiveValue + " " + unitsString;
        String calls = "<br>Calls: " + fp.getNumCalls();
        String subr = "<br>SubCalls: " + fp.getNumSubr();

        return "<html>" + Common.HTMLEntityEncode(name) + exclusive + inclusive + calls + subr + "</html>";

    }

    public String getRowLabelToolTipText(int row) {
        Thread thread = theThreads.get(row);

        //        List fpList = (List) threads.get(row);
        //        edu.uoregon.tau.perfdmf.Thread thread = ((FunctionProfile)fpList.get(0)).getThread();

        if (ParaProf.getHelpWindow().isShowing()) {
            ParaProf.getHelpWindow().clearText();
            if (thread.getNodeID() == -1) {
                ParaProf.getHelpWindow().writeText("This line represents the mean statistics (over all threads).\n");

            } else if (thread.getNodeID() == -2) {

            } else if (thread.getNodeID() == -3) {
                ParaProf.getHelpWindow().writeText(
                        "This line represents the standard deviation of each function (over threads).\n");

            } else {
                ParaProf.getHelpWindow().writeText("n,c,t stands for: Node, Context and Thread.\n");
            }
            ParaProf.getHelpWindow().writeText("Right click to display options for viewing the data.");
            ParaProf.getHelpWindow().writeText("Left click to go directly to the Thread Data Window");
        }

        return "Right click for options";
    }

    public void reloadData() {
        functionOrder = dataSorter.getOrdering();
        theThreads = dataSorter.getThreads();
        //        
        //        threads = dataSorter.getAllFunctionProfilesMinimal();
        //        
        //
        //        if (threads.size() > 1 && !window.getStackBars()) { // insert dummies
        //            List mean = (List) threads.get(0);
        //            for (int i = 0; i < threads.size(); i++) {
        //                List thread = (List) threads.get(i);
        //
        //                int meanIndex = 0;
        //
        //                int index = 0;
        //
        //                while (meanIndex < mean.size()) {
        //                    Function meanComparison = ((FunctionProfile) mean.get(meanIndex)).getFunction();
        //                    //while (((PPFunctionProfile) thisThread.get(index)).getFunction() != meanComparison) {
        //
        //                    if (index >= thread.size()) {
        //                        thread.add(index, null);
        //                    } else {
        //                        if (((FunctionProfile) thread.get(index)).getFunction() != meanComparison) {
        //                            thread.add(index, null);
        //                        }
        //                    }
        //                    index++;
        //                    meanIndex++;
        //                }
        //            }
        //        }
        fireModelChanged();
    }

    public String getOtherToolTopText(int row) {
        if (ParaProf.getHelpWindow().isShowing()) {
            ParaProf.getHelpWindow().clearText();
            ParaProf.getHelpWindow().writeText("Your mouse is over the misc. function section!\n");
            ParaProf.getHelpWindow().writeText(
                    "These are functions which have a non zero value,"
                            + " but whose screen representation is less than a pixel.\n");
            ParaProf.getHelpWindow().writeText(
                    "To view these function, right or left click to the left of"
                            + " this bar to bring up windows which will show more detailed information.");
        }

        return "Misc function section ... see help window for details";
    }

    public List<Thread> getThreads() {
        return theThreads;
    }


	public DataSorter getDataSorter() {
		return dataSorter;
	}
}
