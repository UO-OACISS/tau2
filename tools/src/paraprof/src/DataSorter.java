package edu.uoregon.tau.paraprof;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.UserEventValueType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UserEvent;
import edu.uoregon.tau.perfdmf.UserEventProfile;

/**
 * DataSorter.java
 * This object manages data for the various windows giving them the capability to show only
 * functions that are in groups supposed to be shown. 
 *  
 * 
 * <P>CVS $Id: DataSorter.java,v 1.18 2009/10/27 00:13:48 amorris Exp $</P>
 * @author	Alan Morris, Robert Bell
 * @version	$Revision: 1.18 $
 */
public class DataSorter implements Comparator<FunctionProfile> {

    private ParaProfTrial ppTrial = null;

    private Metric selectedMetric;
    private Metric sortMetric;
    private ValueType valueType;
    private SortType sortType;
    private ValueType sortValueType;
    private UserEventValueType userEventValueType = UserEventValueType.NUMSAMPLES;

    private boolean descendingOrder;
    private boolean sortByVisible = true;

    private Function phase;

    private static SortType defaultSortType = SortType.MEAN_VALUE;
    private static ValueType defaultValueType = ValueType.EXCLUSIVE;
    private static boolean defaultSortOrder = true;
    
    private boolean selectedSnapshotOverride = false;
    private int selectedSnapshot = -1;

    public DataSorter(ParaProfTrial ppTrial) {
        this.ppTrial = ppTrial;
        this.selectedMetric = ppTrial.getDefaultMetric();
        this.sortMetric = ppTrial.getDefaultMetric();

        this.sortType = DataSorter.defaultSortType;
        this.valueType = DataSorter.defaultValueType;
        this.sortValueType = DataSorter.defaultValueType;
        this.descendingOrder = DataSorter.defaultSortOrder;
    }

    public boolean isTimeMetric() {
        return selectedMetric.isTimeMetric();
    }

    public boolean isDerivedMetric() {

        // We can't do this, HPMToolkit stuff has /'s and -'s all over the place
        //String metricName = this.getMetricName(this.getSelectedMetricID());
        //if (metricName.indexOf("*") != -1 || metricName.indexOf("/") != -1)
        //    return true;
        return selectedMetric.getDerivedMetric();
    }

    public List<Comparable> getUserEventProfiles(Thread thread) {

        UserEventProfile userEventProfile = null;

        List<Comparable> newList = new ArrayList<Comparable>();

        for (Iterator<UserEventProfile> e1 = thread.getUserEventProfiles(); e1.hasNext();) {
            userEventProfile = (UserEventProfile) e1.next();
            if (userEventProfile != null) {
                PPUserEventProfile ppUserEventProfile = new PPUserEventProfile(this, thread, userEventProfile);
                newList.add(ppUserEventProfile);
            }
        }
        Collections.sort(newList);
        return newList;
    }

    private List<Comparable> createFunctionProfileList(Thread thread, boolean callpath) {
        List<Comparable> newList = null;

        List<FunctionProfile> functionList = thread.getFunctionProfiles();
        newList = new ArrayList<Comparable>();

        for (int i = 0; i < functionList.size(); i++) {
            FunctionProfile functionProfile = (FunctionProfile) functionList.get(i);
            if (functionProfile != null) {
                if (callpath) {
                    if (functionProfile.getFunction().isPhaseMember(phase)) {
                        PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                        newList.add(ppFunctionProfile);
                    }
                } else {
                    if (ppTrial.displayFunction(functionProfile.getFunction())
                            && functionProfile.getFunction().isPhaseMember(phase)) {
                        PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                        newList.add(ppFunctionProfile);
                    }
                }
            }
        }
        Collections.sort(newList);
        return newList;
    }

    public List<Comparable> getCallPathFunctionProfiles(Thread thread) {
        return createFunctionProfileList(thread, true);
    }

    public List<Comparable> getFunctionProfiles(Thread thread) {
        return createFunctionProfileList(thread, false);
    }

    public List<FunctionProfile> getBasicFunctionProfiles(Thread thread) {

        List<FunctionProfile> newList = null;

        List functionList = thread.getFunctionProfiles();
        newList = new ArrayList<FunctionProfile>();

        for (int i = 0; i < functionList.size(); i++) {
            FunctionProfile functionProfile = (FunctionProfile) functionList.get(i);
            if (functionProfile != null && ppTrial.displayFunction(functionProfile.getFunction())) {
                newList.add(functionProfile);
            }
        }
        Collections.sort(newList, this);
        return newList;
    }

    public List<PPThread> getAllFunctionProfiles() {
        long time = System.currentTimeMillis();

        List<PPThread> threads = new ArrayList<PPThread>();
        Thread thread;
        PPThread ppThread;

        PPThread order = null;

        // if there is only one thread, don't show mean and stddev
        if (ppTrial.getDataSource().getAllThreads().size() > 1) {
            thread = ppTrial.getDataSource().getStdDevData();
            ppThread = new PPThread(thread, this.ppTrial);
            for (Iterator<FunctionProfile> it = thread.getFunctionProfiles().iterator(); it.hasNext();) {
                FunctionProfile functionProfile = (FunctionProfile) it.next();
                if (functionProfile != null && ppTrial.displayFunction(functionProfile.getFunction())
                        && functionProfile.getFunction().isPhaseMember(phase)) {
                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                    ppThread.addFunction(ppFunctionProfile);
                }
            }
            Collections.sort(ppThread.getFunctionList());
            threads.add(ppThread);

            thread = ppTrial.getDataSource().getMeanData();
            ppThread = new PPThread(thread, this.ppTrial);
            for (Iterator<FunctionProfile> it = thread.getFunctionProfiles().iterator(); it.hasNext();) {
                FunctionProfile functionProfile = (FunctionProfile) it.next();
                if (functionProfile != null && ppTrial.displayFunction(functionProfile.getFunction())
                        && functionProfile.getFunction().isPhaseMember(phase)) {
                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                    ppThread.addFunction(ppFunctionProfile);
                }
            }
            Collections.sort(ppThread.getFunctionList());
            threads.add(ppThread);

            order = ppThread;

            //            thread = ppTrial.getDataSource().getTotalData();
            //            ppThread = new PPThread(thread, this.ppTrial);
            //            for (Iterator e4 = thread.getFunctionProfiles().iterator(); e4.hasNext();) {
            //                FunctionProfile functionProfile = (FunctionProfile) e4.next();
            //                if (functionProfile != null && ppTrial.displayFunction(functionProfile.getFunction())
            //                        && functionProfile.getFunction().isPhaseMember(phase)) {
            //                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
            //                    ppThread.addFunction(ppFunctionProfile);
            //                }
            //            }
            //            Collections.sort(ppThread.getFunctionList());
            //            threads.add(ppThread);

        }

        for (Iterator<Thread> it = ppTrial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
            thread = (Thread) it.next();

            ppThread = new PPThread(thread, this.ppTrial);
            for (Iterator<PPFunctionProfile> it2 = order.getFunctionListIterator(); it2.hasNext();) {
                PPFunctionProfile orderfp = it2.next();

                FunctionProfile fp = thread.getFunctionProfile(orderfp.getFunction());

                if (fp != null) {
                    ppThread.addFunction(new PPFunctionProfile(this, thread, fp));
                }
            }
            if (ppThread.getFunctionList().size() > 0) {
                threads.add(ppThread);
            }

        }

        //        for (Iterator it = ppTrial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
        //            thread = (edu.uoregon.tau.perfdmf.Thread) it.next();
        //
        //            //Counts the number of ppFunctionProfiles that are actually added.
        //            //It is possible (because of selection criteria - groups for example) to filter
        //            //out all functions on a particular thread. The default at present is not to add.
        //
        //            int counter = 0; //Counts the number of PPFunctionProfile that are actually added.
        //            ppThread = new PPThread(thread, this.ppTrial);
        //
        //            //Do not add thread to the context until we have verified counter is not zero (done after next loop).
        //            //Now enter the thread data loops for this thread.
        //            for (Iterator e4 = thread.getFunctionProfiles().iterator(); e4.hasNext();) {
        //                FunctionProfile functionProfile = (FunctionProfile) e4.next();
        //                if (functionProfile != null && ppTrial.displayFunction(functionProfile.getFunction())
        //                        && functionProfile.getFunction().isPhaseMember(phase)) {
        //                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
        //                    ppThread.addFunction(ppFunctionProfile);
        //                    counter++;
        //                }
        //            }
        //
        //            // sort thread and add to the list
        //            if (counter != 0) {
        //                Collections.sort(ppThread.getFunctionList());
        //                threads.add(ppThread);
        //            }
        //        }

        //        if (ppTrial.getDataSource().getAllThreads().size() > 1 && threads.size() == 4) {
        //            threads.remove(0);
        //            threads.remove(0);
        //            threads.remove(0);
        //        }

        time = (System.currentTimeMillis()) - time;
        //System.out.println("Time for getAllFunctionProfiles : " + time);

        return threads;
    }

    private void addThread(List<List<FunctionProfile>> threads, List<FunctionProfile> order, Thread thread) {

        List<FunctionProfile> list = new ArrayList<FunctionProfile>();

        for (int i = 0, n = order.size(); i < n; i++) {
            FunctionProfile orderfp = order.get(i);

            FunctionProfile fp = thread.getFunctionProfile(orderfp.getFunction());

            if (fp != null) {
                list.add(fp);
            }
        }
        if (list.size() > 0) {
            threads.add(list);
        }

    }

    /**
     * Constructs a list of ordered lists of FunctionProfiles for all threads
     *
     * @return a list of lists of FunctionProfiles
     */
    public List<List<FunctionProfile>> getAllFunctionProfilesMinimal() {
        long time = System.currentTimeMillis();

        // a list of lists of FunctionProfiles
        List<List<FunctionProfile>> threads = new ArrayList<List<FunctionProfile>>();

        // a list of FunctionProfiles that represents the ordering
        List<FunctionProfile> order = new ArrayList<FunctionProfile>();

        List<FunctionProfile> meanProfiles = ppTrial.getDataSource().getMeanData().getFunctionProfiles();

        for (int i = 0, n = meanProfiles.size(); i < n; i++) {
            FunctionProfile functionProfile = (FunctionProfile) meanProfiles.get(i);
            if (functionProfile != null && ppTrial.displayFunction(functionProfile.getFunction())
                    && functionProfile.getFunction().isPhaseMember(phase)) {
                order.add(functionProfile);
            }
        }

        // sort the list
        Collections.sort(order, this);

        // add the pseudo-thread std. dev.
        addThread(threads, order, ppTrial.getDataSource().getStdDevData());

        // add the pseudo-thread total
        //addThread(threads, order, ppTrial.getDataSource().getTotalData());

        // add the mean thread, already sorted
        threads.add(order);

        // add all the other threads
        for (Iterator<Thread> it = ppTrial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();
            addThread(threads, order, thread);
        }

        time = (System.currentTimeMillis()) - time;
        //System.out.println("Time for getAllFunctionProfilesMinimal : " + time);

        return threads;
    }

    public FunctionOrdering getOrdering() {
        long time = System.currentTimeMillis();

        List<FunctionProfile> order = new ArrayList<FunctionProfile>();

        List<FunctionProfile> meanProfiles = ppTrial.getDataSource().getMeanData().getFunctionProfiles();

        for (int i = 0, n = meanProfiles.size(); i < n; i++) {
            FunctionProfile functionProfile = (FunctionProfile) meanProfiles.get(i);
            if (functionProfile != null && ppTrial.displayFunction(functionProfile.getFunction())
                    && functionProfile.getFunction().isPhaseMember(phase)) {
                order.add(functionProfile);
            }
        }

        Collections.sort(order, this);

        Function functions[] = new Function[order.size()];
        for (int i = 0, n = order.size(); i < n; i++) {
            functions[i] = order.get(i).getFunction();
        }

        FunctionOrdering fo = new FunctionOrdering(this);
        fo.setFunctions(functions);

        time = (System.currentTimeMillis()) - time;
        //System.out.println("Time for getOrdering : " + time);

        return fo;
    }

    public List<Thread> getThreads() {
        ArrayList<Thread> threads = new ArrayList<Thread>();
        if (ppTrial.getDataSource().getAllThreads().size() > 1) {
            threads.add(ppTrial.getDataSource().getStdDevData());
            //threads.add(ppTrial.getDataSource().getTotalData());
            threads.add(ppTrial.getDataSource().getMeanData());
        }

        // add all the other threads
        for (Iterator<Thread> it = ppTrial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();
            threads.add(thread);
        }
        return threads;
    }

    public List<PPFunctionProfile> getFunctionData(Function function, boolean includeMean, boolean includeStdDev) {
        List<PPFunctionProfile> newList = new ArrayList<PPFunctionProfile>();

        Thread thread;

        if (ppTrial.getDataSource().getAllThreads().size() > 1) {
            if (includeMean) {
                thread = ppTrial.getDataSource().getMeanData();
                FunctionProfile functionProfile = thread.getFunctionProfile(function);
                if (functionProfile != null) {
                    //Create a new thread data object.
                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                    newList.add(ppFunctionProfile);
                }
            }

            if (includeStdDev) {
                thread = ppTrial.getDataSource().getStdDevData();
                FunctionProfile functionProfile = thread.getFunctionProfile(function);
                if (functionProfile != null) {
                    //Create a new thread data object.
                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                    newList.add(ppFunctionProfile);
                }

                //                thread = ppTrial.getDataSource().getTotalData();
                //                functionProfile = thread.getFunctionProfile(function);
                //                if (functionProfile != null) {
                //                    //Create a new thread data object.
                //                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                //                    newList.add(ppFunctionProfile);
                //                }

            }
        }
        for (Iterator<Thread> it = ppTrial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
            thread = (Thread) it.next();
            FunctionProfile functionProfile = thread.getFunctionProfile(function);
            if (functionProfile != null) {
                //Create a new thread data object.
                PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                newList.add(ppFunctionProfile);
            }
        }
        Collections.sort(newList);
        //Collections.sort(newList, new AlphanumComparator());
        return newList;
    }

    public List<PPFunctionProfile> getFunctionAcrossPhases(Function function, Thread thread) {
        List<PPFunctionProfile> newList = new ArrayList<PPFunctionProfile>();

        String functionName = function.getName();
        if (function.isCallPathFunction()) {
            functionName = functionName.substring(functionName.indexOf("=>") + 2).trim();
        }

        for (Iterator<FunctionProfile> it = thread.getFunctionProfileIterator(); it.hasNext();) {
            FunctionProfile functionProfile = (FunctionProfile) it.next();
            if (functionProfile != null) {

                if (functionProfile.isCallPathFunction()) {
                    String name = functionProfile.getName();
                    name = name.substring(name.indexOf("=>") + 2).trim();
                    if (functionName.compareTo(name) == 0) {
                        PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                        newList.add(ppFunctionProfile);
                    }
                }
            }
        }
        Collections.sort(newList);
        return newList;
    }

    public List<PPUserEventProfile> getUserEventData(UserEvent userEvent) {
        List<PPUserEventProfile> newList = new ArrayList<PPUserEventProfile>();

        UserEventProfile userEventProfile;

        PPUserEventProfile ppUserEventProfile;

        Thread thread = ppTrial.getDataSource().getMeanData();
        userEventProfile = thread.getUserEventProfile(userEvent);
        if (userEventProfile != null) {
            ppUserEventProfile = new PPUserEventProfile(this, thread, userEventProfile);
            newList.add(ppUserEventProfile);
        }

        thread = ppTrial.getDataSource().getStdDevData();
        userEventProfile = thread.getUserEventProfile(userEvent);
        if (userEventProfile != null) {
            ppUserEventProfile = new PPUserEventProfile(this, thread, userEventProfile);
            newList.add(ppUserEventProfile);
        }

        for (Iterator<Thread> it = ppTrial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
            thread = (Thread) it.next();

            userEventProfile = thread.getUserEventProfile(userEvent);
            if (userEventProfile != null) {
                ppUserEventProfile = new PPUserEventProfile(this, thread, userEventProfile);
                newList.add(ppUserEventProfile);
            }
        }
        Collections.sort(newList);
        return newList;
    }

    //////////////////////////////////////////////////////////////////////////
    // Comparison stuff
    //////////////////////////////////////////////////////////////////////////

    public int compare(FunctionProfile arg0, FunctionProfile arg1) {
        FunctionProfile left = (FunctionProfile) arg0;
        FunctionProfile right = (FunctionProfile) arg1;
        if (descendingOrder) {
            return -performComparison(left, right);
        }
        return performComparison(left, right);
    }

    private int performComparison(FunctionProfile left, FunctionProfile right) {
        ValueType type = getSortValueType();

        if (sortType == SortType.NAME) {
            return getDisplayName(right).compareTo(getDisplayName(left));

        } else if (sortType == SortType.NCT) {
            return compareNCT(left, right);

        } else if (sortType == SortType.MEAN_VALUE) {
            return compareToHelper(type.getValue(left.getFunction().getMeanProfile(), getSortMetric(), getSelectedSnapshot()),
                    type.getValue(right.getFunction().getMeanProfile(), getSortMetric(), getSelectedSnapshot()),
                    left.getFunction().getMeanProfile(), right.getFunction().getMeanProfile());

        } else if (sortType == SortType.VALUE) {
            return compareToHelper(type.getValue(left, getSortMetric(), getSelectedSnapshot()), type.getValue(right,
                    getSortMetric(), getSelectedSnapshot()));
        } else {
            throw new ParaProfException("Unexpected sort type: " + sortType);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // Static Comparison stuff
    //////////////////////////////////////////////////////////////////////////

    // handles reversed callpaths
    public static String getDisplayName(FunctionProfile functionProfile) {
        if (ParaProf.preferences.getReversedCallPaths()) {
            return functionProfile.getFunction().getReversedName();
        } else {
            return functionProfile.getFunction().getName();
        }
    }

    private static int compareNCT(FunctionProfile left, FunctionProfile right) {
        Thread l = left.getThread();
        Thread r = right.getThread();
        if (l.getNodeID() != r.getNodeID()) {
            return l.getNodeID() - r.getNodeID();
        } else if (l.getContextID() != r.getContextID()) {
            return l.getContextID() - r.getContextID();
        } else {
            return l.getThreadID() - r.getThreadID();
        }
    }

    private static int compareToHelper(double d1, double d2) {
        double result = d1 - d2;
        if (result < 0.00) {
            return -1;
        } else if (result == 0.00) {
            return 0;
        } else {
            return 1;
        }
    }

    private static int compareToHelper(double d1, double d2, FunctionProfile f1, FunctionProfile f2) {
        double result = d1 - d2;
        if (result < 0.00) {
            return -1;
        } else if (result == 0.00) {
            // this is here to make sure that things get sorted the same for mean and other threads
            // in the case of callpath profiles, multiple functionProfiles may have the same values
            // we need them in the same order for everyone
            return f1.getFunction().compareTo(f2.getFunction());
        } else {
            return 1;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // Getters/Setters
    //////////////////////////////////////////////////////////////////////////

    public Function getPhase() {
        return phase;
    }

    public ParaProfTrial getPpTrial() {
        return ppTrial;
    }

    public void setPpTrial(ParaProfTrial ppTrial) {
        this.ppTrial = ppTrial;
    }

    public Metric getSortMetric() {
        if (getSortByVisible()) {
            return selectedMetric;
        } else {
            return sortMetric;
        }
    }

    public void setSortMetric(Metric sortMetric) {
        this.sortMetric = sortMetric;
    }

    public ValueType getSortValueType() {
        if (getSortByVisible()) {
            return valueType;
        } else {
            return sortValueType;
        }
    }

    public void setSortValueType(ValueType sortValueType) {
        this.sortValueType = sortValueType;
    }

    public boolean getSortByVisible() {
        return sortByVisible;
    }

    public void setSortByVisible(boolean sortByVisible) {
        this.sortByVisible = sortByVisible;
    }

    public void setSelectedMetric(Metric metric) {
        this.selectedMetric = metric;
    }

    public double getValue(FunctionProfile fp) {
        return getValueType().getValue(fp, selectedMetric, getSelectedSnapshot());
    }

    public double getValue(FunctionProfile fp, int snapshot) {
        return getValueType().getValue(fp, selectedMetric, snapshot);
    }

    
    public Metric getSelectedMetric() {
        return selectedMetric;
    }

    public void setDescendingOrder(boolean descendingOrder) {
        this.descendingOrder = descendingOrder;
    }

    public boolean getDescendingOrder() {
        return this.descendingOrder;
    }

    public void setSortType(SortType sortType) {
        this.sortType = sortType;
    }

    public SortType getSortType() {
        return sortType;
    }

    public void setValueType(ValueType valueType) {
        this.valueType = valueType;
    }

    public ValueType getValueType() {
        return valueType;
    }

    public void setPhase(Function phase) {
        this.phase = phase;
    }

    public UserEventValueType getUserEventValueType() {
        return userEventValueType;
    }

    public void setUserEventValueType(UserEventValueType userEventValueType) {
        this.userEventValueType = userEventValueType;
    }

    //////////////////////////////////////////////////////////////////////////
    // static default stuff
    //////////////////////////////////////////////////////////////////////////

    public static void setDefaultSortType(SortType sortType) {
        DataSorter.defaultSortType = sortType;
    }

    public static void setDefaultValueType(ValueType valueType) {
        DataSorter.defaultValueType = valueType;
    }

    public static void setDefaultSortOrder(boolean order) {
        DataSorter.defaultSortOrder = order;
    }

    public int getSelectedSnapshot() {
        if (selectedSnapshotOverride) {
            return selectedSnapshot;
        } else {
            return ppTrial.getSelectedSnapshot();
        }
    }

    public boolean getSelectedSnapshotOverride() {
        return selectedSnapshotOverride;
    }

    public void setSelectedSnapshotOverride(boolean selectedSnapshotOverride) {
        this.selectedSnapshotOverride = selectedSnapshotOverride;
    }

    public void setSelectedSnapshot(int selectedSnapshot) {
        this.selectedSnapshot = selectedSnapshot;
    }
}