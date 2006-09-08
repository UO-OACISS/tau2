package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;
import java.awt.event.MouseEvent;
import java.io.File;
import java.util.*;

import javax.swing.JComponent;

import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * Compares threads from (potentially) any trial
 * 
 * <P>CVS $Id: ComparisonBarChartModel.java,v 1.4 2006/09/08 00:22:10 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.4 $
 */
public class ComparisonBarChartModel extends AbstractBarChartModel {

    private List ppTrials = new ArrayList();
    private List threads = new ArrayList();

    private Map rowMap = new HashMap(); // maps function names (Strings) to RowBlobs
    private List rows = new ArrayList();

    private DataSorter dataSorter;
    private FunctionBarChartWindow window;

    private LegendModel legendModel;

    // A RowBlob is the data needed for one row.  If we're comparing trial A to trial B, then
    // a single RowBlob will contain two PPFunctionProfiles, one for each trial.
    private static class RowBlob extends ArrayList {
        String functionName;

        public RowBlob(String functionName) {
            this.functionName = functionName;
        }

        public String getFunctionName() {
            return functionName;
        }

        public void add(int index, Object element) {

            while (index >= this.size()) {
                this.add(null);
            }
            super.set(index, element);
        }
        
        public Object get(int index) {
            if (index >= size()) {
                return null;
            } else {
                return super.get(index);
            }
        }
        
    }

    public ComparisonBarChartModel(FunctionBarChartWindow window, ParaProfTrial ppTrial, Thread thread, DataSorter dataSorter) {
        this.window = window;
        this.dataSorter = dataSorter;
        addThread(ppTrial, thread);
    }


    public LegendModel getLegendModel() {
        if (legendModel == null) {
            legendModel = new LegendModel() {

                public int getNumElements() {
                    return ppTrials.size();
                }

                public String getLabel(int index) {
                    ParaProfTrial ppTrial = (ParaProfTrial) ppTrials.get(index);
                    Thread thread = (Thread) threads.get(index);

                    return ppTrial.getName() + " - " + ParaProfUtils.getThreadIdentifier(thread);

                }

                public Color getColor(int index) {
                    return ComparisonBarChartModel.this.getValueColor(0,index);
                }
            };
        }
        return legendModel;
    }

    public void addThread(ParaProfTrial ppTrial, Thread thread) {
        ppTrials.add(ppTrial);
        threads.add(thread);
    }


    public void reloadData() {
        rows.clear();
        rowMap.clear();

        ParaProfTrial selectedTrial = (ParaProfTrial) ppTrials.get(0);
        Thread selectedThread = (Thread) threads.get(0);

        dataSorter.setPpTrial(selectedTrial);
        List list = dataSorter.getFunctionProfiles(selectedThread);
        for (Iterator it = list.iterator(); it.hasNext();) {
            PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) it.next();
            RowBlob blob = new RowBlob(ppFunctionProfile.getDisplayName());
            rows.add(blob);
            blob.add(ppFunctionProfile);
            rowMap.put(ppFunctionProfile.getDisplayName(), blob);
        }

        // add all the others
        for (int i = 1; i < ppTrials.size(); i++) {
            ParaProfTrial ppTrial = (ParaProfTrial) ppTrials.get(i);
            Thread thread = (Thread) threads.get(i);

            // We now use the kludge wrapper so that the metric ids between trials get mapped
            // dataSorter.setPpTrial(ppTrial);
            DataSorter newDataSorter = new DataSorterWrapper(dataSorter, ppTrial);
            list = newDataSorter.getFunctionProfiles(thread);

            for (Iterator it = list.iterator(); it.hasNext();) {
                PPFunctionProfile fp = (PPFunctionProfile) it.next();
                if (ppTrial.displayFunction(fp.getFunction())) {
                    RowBlob blob = (RowBlob) rowMap.get(fp.getFunction().getName());
                    if (blob != null) {
                        blob.add(i, fp);
                    } else {
                        blob = new RowBlob(fp.getDisplayName());
                        rows.add(blob);
                        rowMap.put(fp.getFunction(), blob);
                        blob.add(i, fp);
                    }
                }
            }

        }

        fireModelChanged();
    }

    public int getNumRows() {
        return rows.size();
    }

    public int getSubSize() {
        return ppTrials.size();
        //return ((List) rows.get(0)).size();
    }

    public String getLabel(int row) {
        // TODO Auto-generated method stub
        return null;
    }

    public double getValue(int row, int subIndex) {
        RowBlob blob = (RowBlob) rows.get(row);
        PPFunctionProfile ppFp = (PPFunctionProfile) blob.get(subIndex);
        if (ppFp == null) {
            return -1;
        } else {
            return ppFp.getValue();
        }
    }

    public String getValueLabel(int row, int subIndex) {
        double value = getValue(row, subIndex);
        if (window.getDataSorter().getValueType() == ValueType.EXCLUSIVE_PERCENT
                || window.getDataSorter().getValueType() == ValueType.INCLUSIVE_PERCENT) {

            //s = (UtilFncs.adjustDoublePresision(value, 4)) + "%";
            return UtilFncs.getOutputString(0, value, 6) + "%";

        } else {
            return UtilFncs.getOutputString(window.units(), value, ParaProf.defaultNumberPrecision);
        }
    }

    public String getRowLabel(int row) {
        RowBlob blob = (RowBlob) rows.get(row);
        return blob.getFunctionName();
    }

    public Color getValueColor(int row, int subIndex) {
        // we use the "default" colors for our legend and bars
        List colorList = ParaProf.colorChooser.getColors();
        return (Color)colorList.get(subIndex % colorList.size());
    }

    public Color getValueHighlightColor(int row, int subIndex) {
        // TODO Auto-generated method stub
        return null;
    }

    public void fireValueClick(int row, int subIndex, MouseEvent e, JComponent owner) {
        // TODO Auto-generated method stub

    }

    public void fireRowLabelClick(int row, MouseEvent e, JComponent owner) {
        // TODO Auto-generated method stub

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

    public static void main(String[] args) {
        final ParaProf paraProf = new ParaProf();
        paraProf.startSystem();

        File[] files = new File[1];
        files[0] = new File("/home/amorris/data/packed/lu.C.512.ppk");
        ParaProfTrial ppTrial = loadTrial(files, 7);

        files[0] = new File("/home/amorris/data/packed/lu.A.128.ppk");
        ParaProfTrial lu128 = loadTrial(files, 7);

        FunctionBarChartWindow window = FunctionBarChartWindow.CreateComparisonWindow(ppTrial,
                ppTrial.getDataSource().getMeanData(), null);
        window.addThread(ppTrial, ppTrial.getDataSource().getThread(0, 0, 0));
        window.addThread(ppTrial, ppTrial.getDataSource().getThread(1,0,0));


        window.addThread(lu128, lu128.getDataSource().getMeanData());

        window.show();

    }

    public static ParaProfTrial loadTrial(File files[], int type) {
        ParaProfApplication application = ParaProf.applicationManager.addApplication();
        application.setName("New Application");

        ParaProfExperiment experiment = application.addExperiment();

        ParaProf.paraProfManagerWindow.addTrial(application, experiment, files, type, false, false);

        Vector trials = experiment.getTrials();

        ParaProfTrial ppTrial = (ParaProfTrial) trials.get(0);

        while (ppTrial.loading()) {
            sleep(500);
        }

        return ppTrial;
    }

    private static void sleep(int msec) {
        try {
            java.lang.Thread.sleep(msec);
        } catch (Exception e) {
            throw new RuntimeException("Exception while sleeping");
        }
    }

    public List getThreads() {
        return threads;
    }

    public List getPpTrials() {
        return ppTrials;
    }

    public void setThreads(List threads) {
        this.threads = threads;
    }
}
