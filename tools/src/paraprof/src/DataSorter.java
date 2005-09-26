package edu.uoregon.tau.paraprof;

import java.util.*;

import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.UserEventValueType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.perfdmf.Thread;

/**
 * DataSorter.java
 * This object manages data for the various windows giving them the capability to show only
 * functions that are in groups supposed to be shown. 
 *  
 * 
 * <P>CVS $Id: DataSorter.java,v 1.1 2005/09/26 21:12:04 amorris Exp $</P>
 * @author	Alan Morris, Robert Bell
 * @version	$Revision: 1.1 $
 */
public class DataSorter {

    private ParaProfTrial ppTrial = null;
    private double maxExclusiveSum = 0;
    private double maxExclusives[];

    private int selectedMetricID;
    private boolean descendingOrder;
    private boolean showAsPercent;
    private SortType sortType = SortType.MEAN_VALUE;
    private ValueType valueType = ValueType.EXCLUSIVE;
    private UserEventValueType userEventValueType = UserEventValueType.NUMSAMPLES;

    private Function phase;

    public DataSorter(ParaProfTrial ppTrial) {
        this.ppTrial = ppTrial;
        this.selectedMetricID = ppTrial.getDefaultMetricID();
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

    public boolean isTimeMetric() {
        String metricName = ppTrial.getMetricName(this.getSelectedMetricID());
        metricName = metricName.toUpperCase();
        if (metricName.indexOf("TIME") == -1) {
            return false;
        } else {
            return true;
        }
    }

    public boolean isDerivedMetric() {

        // We can't do this, HPMToolkit stuff has /'s and -'s all over the place
        //String metricName = this.getMetricName(this.getSelectedMetricID());
        //if (metricName.indexOf("*") != -1 || metricName.indexOf("/") != -1)
        //    return true;
        return ppTrial.getMetric(this.getSelectedMetricID()).getDerivedMetric();
    }

    public void setSelectedMetricID(int metric) {
        this.selectedMetricID = metric;
    }

    public int getSelectedMetricID() {
        return selectedMetricID;
    }

    public void setDescendingOrder(boolean descendingOrder) {
        this.descendingOrder = descendingOrder;
    }

    public boolean getDescendingOrder() {
        return this.descendingOrder;
    }

    public void setShowAsPercent(boolean showAsPercent) {
        this.showAsPercent = showAsPercent;
    }

    public boolean getShowAsPercent() {
        return showAsPercent;
    }

    public void setSortType(SortType sortType) {
        this.sortType = sortType;
    }

    public SortType getSortType() {
        return this.sortType;
    }

    public void setValueType(ValueType valueType) {
        this.valueType = valueType;
    }

    public ValueType getValueType() {
        return this.valueType;
    }

    public List getUserEventProfiles(edu.uoregon.tau.perfdmf.Thread thread) {

        UserEventProfile userEventProfile = null;
        List list = thread.getUserEventProfiles();

        List newList = new ArrayList();

        for (Iterator e1 = list.iterator(); e1.hasNext();) {
            userEventProfile = (UserEventProfile) e1.next();
            if (userEventProfile != null) {
                PPUserEventProfile ppUserEventProfile = new PPUserEventProfile(this, thread, userEventProfile);
                newList.add(ppUserEventProfile);
            }
        }
        Collections.sort(newList);
        return newList;
    }

    public List getFunctionProfiles(edu.uoregon.tau.perfdmf.Thread thread) {
        List newList = null;

        List functionList = thread.getFunctionProfiles();
        newList = new ArrayList();

        for (int i = 0; i < functionList.size(); i++) {
            FunctionProfile functionProfile = (FunctionProfile) functionList.get(i);
            if (functionProfile != null) {
                if (ppTrial.displayFunction(functionProfile.getFunction()) && functionProfile.getFunction().isPhaseMember(phase)) {
                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                    newList.add(ppFunctionProfile);
                }
            }
        }
        Collections.sort(newList);
        return newList;
    }

    public List getAllFunctionProfiles() {
        List threads = new ArrayList();
        edu.uoregon.tau.perfdmf.Thread thread;
        PPThread ppThread;

        // if there is only one thread, don't show mean and stddev
        if (ppTrial.getDataSource().getAllThreads().size() > 1) {
            thread = ppTrial.getDataSource().getStdDevData();
            ppThread = new PPThread(thread, this.ppTrial);
            for (Iterator e4 = thread.getFunctionProfiles().iterator(); e4.hasNext();) {
                FunctionProfile functionProfile = (FunctionProfile) e4.next();
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
            for (Iterator e4 = thread.getFunctionProfiles().iterator(); e4.hasNext();) {
                FunctionProfile functionProfile = (FunctionProfile) e4.next();
                if (functionProfile != null && ppTrial.displayFunction(functionProfile.getFunction())
                        && functionProfile.getFunction().isPhaseMember(phase)) {
                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                    ppThread.addFunction(ppFunctionProfile);
                }
            }
            Collections.sort(ppThread.getFunctionList());
            threads.add(ppThread);

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

        // reset this in case we are switching metrics
        maxExclusiveSum = 0;

        maxExclusives = new double[ppTrial.getDataSource().getNumFunctions()];

        for (Iterator it = ppTrial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
            thread = (edu.uoregon.tau.perfdmf.Thread) it.next();

            //Counts the number of ppFunctionProfiles that are actually added.
            //It is possible (because of selection criteria - groups for example) to filter
            //out all functions on a particular thread. The default at present is not to add.

            int counter = 0; //Counts the number of PPFunctionProfile that are actually added.
            ppThread = new PPThread(thread, this.ppTrial);

            double sum = 0.0;

            //Do not add thread to the context until we have verified counter is not zero (done after next loop).
            //Now enter the thread data loops for this thread.
            for (Iterator e4 = thread.getFunctionProfiles().iterator(); e4.hasNext();) {
                FunctionProfile functionProfile = (FunctionProfile) e4.next();
                if (functionProfile != null && ppTrial.displayFunction(functionProfile.getFunction())
                        && functionProfile.getFunction().isPhaseMember(phase)) {
                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                    ppThread.addFunction(ppFunctionProfile);
                    counter++;

                    sum += ppFunctionProfile.getExclusiveValue();

                    maxExclusives[functionProfile.getFunction().getID()] = Math.max(
                            maxExclusives[functionProfile.getFunction().getID()], ppFunctionProfile.getExclusiveValue());
                }
            }

            if (sum > maxExclusiveSum) {
                maxExclusiveSum = sum;
            }

            // sort thread and add to the list
            if (counter != 0) {
                Collections.sort(ppThread.getFunctionList());
                threads.add(ppThread);
            }
        }

//        if (ppTrial.getDataSource().getAllThreads().size() > 1 && threads.size() == 4) {
//            threads.remove(0);
//            threads.remove(0);
//            threads.remove(0);
//        }

        return threads;
    }

    public List getFunctionData(Function function, boolean includeMean, boolean includeStdDev) {
        List newList = new ArrayList();

        edu.uoregon.tau.perfdmf.Thread thread;

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
        for (Iterator it = ppTrial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
            thread = (edu.uoregon.tau.perfdmf.Thread) it.next();
            FunctionProfile functionProfile = thread.getFunctionProfile(function);
            if (functionProfile != null) {
                //Create a new thread data object.
                PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                newList.add(ppFunctionProfile);
            }
        }
        Collections.sort(newList);
        return newList;
    }

    public List getFunctionAcrossPhases(Function function, Thread thread) {
        List newList = new ArrayList();

        String functionName = function.getName();
        if (function.isCallPathFunction()) {
            functionName = functionName.substring(functionName.indexOf("=>")+2).trim();
        }
        
        for (Iterator it = thread.getFunctionProfileIterator(); it.hasNext();) {
            FunctionProfile functionProfile = (FunctionProfile) it.next();
            if (functionProfile != null) {

                if (functionProfile.isCallPathFunction()) {
                    String name = functionProfile.getName();
                    name = name.substring(name.indexOf("=>")+2).trim();
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

    public List getUserEventData(UserEvent userEvent) {
        List newList = new ArrayList();

        UserEventProfile userEventProfile;

        PPUserEventProfile ppUserEventProfile;

        for (Iterator it = ppTrial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
            edu.uoregon.tau.perfdmf.Thread thread = (edu.uoregon.tau.perfdmf.Thread) it.next();

            userEventProfile = thread.getUserEventProfile(userEvent);
            if (userEventProfile != null) {
                //Create a new thread data object.
                ppUserEventProfile = new PPUserEventProfile(this, thread, userEventProfile);
                newList.add(ppUserEventProfile);
            }
        }
        Collections.sort(newList);
        return newList;
    }

    // returns the maximum exclusive sum over all threads
    public double getMaxExclusiveSum() {
        return maxExclusiveSum;
    }

    public double[] getMaxExclusives() {
        return maxExclusives;
    }

    public Function getPhase() {
        return phase;
    }

    public ParaProfTrial getPpTrial() {
        return ppTrial;
    }

    public void setPpTrial(ParaProfTrial ppTrial) {
        this.ppTrial = ppTrial;
    }

}