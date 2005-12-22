package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JComponent;
import javax.swing.JPopupMenu;

import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * A BarChartModel for doing the GlobalDataWindow
 * 
 * <P>CVS $Id: GlobalBarChartModel.java,v 1.3 2005/12/22 19:29:27 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.3 $
 */

public class GlobalBarChartModel extends AbstractBarChartModel {
    private GlobalDataWindow window;
    private DataSorter dataSorter;
    private ParaProfTrial ppTrial;

    private List threads = new ArrayList();

    public GlobalBarChartModel(GlobalDataWindow window, DataSorter dataSorter, ParaProfTrial ppTrial) {
        this.window = window;
        this.dataSorter = dataSorter;
        this.ppTrial = ppTrial;
    }

    public int getNumRows() {
        return threads.size();
    }

    public int getSubSize() {
        if (threads != null && threads.size() >= 1) {
            return ((PPThread) threads.get(0)).getFunctionList().size();
        } else {
            return 0;
        }

    }

    public String getRowLabel(int row) {
        PPThread ppThread = (PPThread) threads.get(row);

        return ppThread.getName();
    }

    public String getValueLabel(int row, int subIndex) {
        // TODO Auto-generated method stub
        return "";
    }

    public double getValue(int row, int subIndex) {
        PPThread ppThread = (PPThread) threads.get(row);
        if (subIndex >= ppThread.getFunctionList().size()) {
            return -1;
        }
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) ppThread.getFunctionList().get(subIndex);
        if (ppFunctionProfile == null) {
            return -1;
        } else {
            return ppFunctionProfile.getValue();
        }
    }

    public Color getValueColor(int row, int subIndex) {
        PPThread ppThread = (PPThread) threads.get(row);
        if (subIndex >= ppThread.getFunctionList().size()) {
            return null;
        }

        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) ppThread.getFunctionList().get(subIndex);
        if (ppFunctionProfile == null) {
            return null;
        } else {
            return ppFunctionProfile.getColor();
        }
    }

    public Color getValueHighlightColor(int row, int subIndex) {
        PPThread ppThread = (PPThread) threads.get(row);
        if (subIndex >= ppThread.getFunctionList().size()) {
            return null;
        }

        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) ppThread.getFunctionList().get(subIndex);
        Function function = ppFunctionProfile.getFunction();
        if (function == (ppTrial.getHighlightedFunction())) {
            return ppTrial.getColorChooser().getHighlightColor();
        } else if (function.isGroupMember(ppTrial.getHighlightedGroup())) {
            return ppTrial.getColorChooser().getGroupHighlightColor();
        }
        return null;
    }

    public void fireValueClick(int row, int subIndex, MouseEvent e, JComponent owner) {
        PPThread ppThread = (PPThread) threads.get(row);
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) ppThread.getFunctionList().get(subIndex);
        if (ParaProfUtils.rightClick(e)) { // Bring up context menu
            JPopupMenu popup = ParaProfUtils.createFunctionClickPopUp(ppTrial, ppFunctionProfile.getFunction(),
                    ppThread.getThread(), owner);
            popup.show(owner, e.getX(), e.getY());
        } else {
            ppTrial.setHighlightedFunction(ppFunctionProfile.getFunction());
            FunctionBarChartWindow functionDataWindow = new FunctionBarChartWindow(ppTrial, ppFunctionProfile.getFunction(),
                    owner);
            functionDataWindow.show();
        }
    }

    public void fireRowLabelClick(int row, MouseEvent e, JComponent owner) {
        PPThread ppThread = (PPThread) threads.get(row);
        if (ParaProfUtils.rightClick(e)) { // Bring up context menu
            ParaProfUtils.handleThreadClick(ppTrial, window.getPhase(), ppThread.getThread(), owner, e);
        } else {
            FunctionBarChartWindow threadDataWindow = new FunctionBarChartWindow(ppTrial, ppThread.getThread(),
                    window.getPhase(), owner);
            threadDataWindow.show();
        }

    }

    public String getValueToolTipText(int row, int subIndex) {
        PPThread ppThread = (PPThread) threads.get(row);
        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) ppThread.getFunctionList().get(subIndex);

        if (ppTrial.getDataSource().getPhasesPresent()) {

            //return "Other Patches";
            
            // we can't use PPFunctionProfile's getFunctionName since the callpath might be reversed
            //return UtilFncs.getRightSide(ppFunctionProfile.getFunctionName());
            return UtilFncs.getRightSide(ppFunctionProfile.getFunction().getName());
        } else {
            //Return the name of the function
            return ppFunctionProfile.getFunctionName();
        }
    }

    public String getRowLabelToolTipText(int row) {

        PPThread ppThread = (PPThread) threads.get(row);

        if (ParaProf.helpWindow.isShowing()) {
            ParaProf.helpWindow.clearText();
            if (ppThread.getNodeID() == -1) {
                ParaProf.helpWindow.writeText("This line represents the mean statistics (over all threads).\n");

            } else if (ppThread.getNodeID() == -2) {

            } else if (ppThread.getNodeID() == -3) {
                ParaProf.helpWindow.writeText("This line represents the standard deviation of each function (over threads).\n");

            } else {
                ParaProf.helpWindow.writeText("n,c,t stands for: Node, Context and Thread.\n");
            }
            ParaProf.helpWindow.writeText("Right click to display options for viewing the data.");
            ParaProf.helpWindow.writeText("Left click to go directly to the Thread Data Window");
        }

        return "Right click for options";
    }

    public void reloadData() {
        threads.clear(); // help the GC
        threads = dataSorter.getAllFunctionProfiles();

        if (threads.size() > 1 && !window.getStackBars()) { // insert dummies
            List mean = ((PPThread) threads.get(0)).getFunctionList();
            for (int i = 0; i < threads.size(); i++) {
                PPThread thread = (PPThread) threads.get(i);

                List thisThread = thread.getFunctionList();

                int meanIndex = 0;

                int index = 0;

                while (meanIndex < mean.size()) {
                    Function meanComparison = ((PPFunctionProfile) mean.get(meanIndex)).getFunction();
                    //while (((PPFunctionProfile) thisThread.get(index)).getFunction() != meanComparison) {

                    if (index >= thisThread.size()) {
                        thisThread.add(index, null);
                    } else {
                        if (((PPFunctionProfile) thisThread.get(index)).getFunction() != meanComparison) {
                            thisThread.add(index, null);
                        }
                    }
                    index++;
                    meanIndex++;
                }
            }
        }
        fireModelChanged();
    }

    public String getOtherToolTopText(int row) {
        if (ParaProf.helpWindow.isShowing()) {
            ParaProf.helpWindow.clearText();
            ParaProf.helpWindow.writeText("Your mouse is over the misc. function section!\n");
            ParaProf.helpWindow.writeText("These are functions which have a non zero value,"
                    + " but whose screen representation is less than a pixel.\n");
            ParaProf.helpWindow.writeText("To view these function, right or left click to the left of"
                    + " this bar to bring up windows which will show more detailed information.");
        }

        return "Misc function section ... see help window for details";
    }

}
