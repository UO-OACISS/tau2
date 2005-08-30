package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;
import java.awt.event.MouseEvent;
import java.io.File;
import java.util.*;

import javax.swing.JComponent;

import edu.uoregon.tau.dms.dss.Thread;
import edu.uoregon.tau.dms.dss.UtilFncs;
import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.paraprof.enums.ValueType;

//public class ComparisonBarChartModel extends AbstractBarChartModel {
public class ComparisonBarChartModel extends AbstractBarChartModel {

    private List ppTrials = new ArrayList();
    private List threads = new ArrayList();
    private ParaProfTrial selectedTrial; // we sort by this one

    private Map blobMap = new HashMap(); // maps function names (Strings) to blobs

    private DataSorter dataSorter;
    private FunctionBarChartWindow window;

    private static class Blob extends ArrayList {
        String functionName;

        public Blob(String functionName) {
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

    private LegendModel legendModel;

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
                    return new Color((index*index*199 + 85) % 255, ((index * 337) + 135) % 255, ((index * 558) + 215) % 255);
                }
            };
        }
        return legendModel;
    }

    public void addThread(ParaProfTrial ppTrial, Thread thread) {
        ppTrials.add(ppTrial);
        threads.add(thread);
    }

    private List blobs = new ArrayList();

    public void reloadData() {
        blobs.clear();
        blobMap.clear();

        ParaProfTrial selectedTrial = (ParaProfTrial) ppTrials.get(0);
        Thread selectedThread = (Thread) threads.get(0);

        dataSorter.setPpTrial(selectedTrial);
        List list = dataSorter.getFunctionProfiles(selectedThread);

        for (Iterator it = list.iterator(); it.hasNext();) {
            PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) it.next();
            Blob blob = new Blob(ppFunctionProfile.getFunctionName());
            blobs.add(blob);
            blob.add(ppFunctionProfile);
            blobMap.put(ppFunctionProfile.getFunctionName(), blob);
        }

        // add all the others
        for (int i = 1; i < ppTrials.size(); i++) {
            ParaProfTrial ppTrial = (ParaProfTrial) ppTrials.get(i);
            Thread thread = (Thread) threads.get(i);

            dataSorter.setPpTrial(ppTrial);
            list = dataSorter.getFunctionProfiles(thread);

            for (Iterator it = list.iterator(); it.hasNext();) {
                PPFunctionProfile fp = (PPFunctionProfile) it.next();
                if (ppTrial.displayFunction(fp.getFunction())) {
                    Blob blob = (Blob) blobMap.get(fp.getFunction().getName());
                    if (blob != null) {
                        blob.add(i, fp);
                    } else {
                        blob = new Blob(fp.getFunctionName());
                        blobMap.put(fp.getFunction(), blob);
                        blob.add(i, fp);

                    }
                }
            }

        }

        fireModelChanged();
    }

    public int getNumRows() {
        return blobs.size();
    }

    public int getSubSize() {
        return ((List) blobs.get(0)).size();
    }

    public String getLabel(int row) {
        // TODO Auto-generated method stub
        return null;
    }

    public double getValue(int row, int subIndex) {
        Blob blob = (Blob) blobs.get(row);
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
        Blob blob = (Blob) blobs.get(row);
        return blob.getFunctionName();
    }

    public Color getValueColor(int row, int subIndex) {
        // TODO Auto-generated method stub
        return new Color((subIndex*subIndex*199 + 85) % 255, ((subIndex * 337) + 135) % 255, ((subIndex * 558) + 215) % 255);
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

        ParaProf.paraProfManagerWindow.addTrial(application, experiment, files, type, false);

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
}
