package edu.uoregon.tau.paraprof;

import java.util.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.paraprof.enums.*;
/**
 * DataSorter.java
 * This object manages data for the various windows giving them the capability to show only
 * functions that are in groups supposed to be shown. 
 *  
 * 
 * <P>CVS $Id: DataSorter.java,v 1.9 2005/03/08 01:11:18 amorris Exp $</P>
 * @author	Alan Morris, Robert Bell
 * @version	$Revision: 1.9 $
 */
public class DataSorter {

    
    private ParaProfTrial trial = null;
    private double maxExclusiveSum = 0;
    private double maxExclusives[];

    private int selectedMetricID;
    private boolean descendingOrder;
    private boolean showAsPercent;
    private SortType sortType = SortType.MEAN_VALUE;
    private ValueType valueType = ValueType.EXCLUSIVE;
    private UserEventValueType userEventValueType= UserEventValueType.NUMSAMPLES;
    
    public DataSorter(ParaProfTrial trial) {
        this.trial = trial;
        this.selectedMetricID = trial.getDefaultMetricID();
    }


    public UserEventValueType getUserEventValueType() {
        return userEventValueType;
    }
    
    public void setUserEventValueType(UserEventValueType userEventValueType) {
        this.userEventValueType = userEventValueType;
    }
    
    public boolean isTimeMetric() {
        String metricName = trial.getMetricName(this.getSelectedMetricID());
        metricName = metricName.toUpperCase();
        if (metricName.indexOf("TIME") == -1)
            return false;
        else
            return true;
    }

    public boolean isDerivedMetric() {

        // We can't do this, HPMToolkit stuff has /'s and -'s all over the place
        //String metricName = this.getMetricName(this.getSelectedMetricID());
        //if (metricName.indexOf("*") != -1 || metricName.indexOf("/") != -1)
        //    return true;
        return trial.getMetric(this.getSelectedMetricID()).getDerivedMetric();
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
    
    
    public Vector getUserEventProfiles(int nodeID, int contextID, int threadID) {

        UserEventProfile userEventProfile = null;
        Vector list = ((edu.uoregon.tau.dms.dss.Thread) trial.getDataSource().getThread(nodeID, contextID,
                threadID)).getUserEventProfiles();

        Vector newList = new Vector();

        for (Enumeration e1 = list.elements(); e1.hasMoreElements();) {
            userEventProfile = (UserEventProfile) e1.nextElement();
            if (userEventProfile != null) {
                PPUserEventProfile ppUserEventProfile = new PPUserEventProfile(this, nodeID, contextID,
                        threadID, userEventProfile);
                newList.addElement(ppUserEventProfile);
            }
        }
        Collections.sort(newList);
        return newList;
    }

    public Vector getFunctionProfiles(int nodeID, int contextID, int threadID) {
        Vector newList = null;

        edu.uoregon.tau.dms.dss.Thread thread;

        if (nodeID == -1) { // mean
            thread = trial.getDataSource().getMeanData();
        } else {
            thread = trial.getDataSource().getThread(nodeID, contextID, threadID);
        }
        Vector functionList = thread.getFunctionProfiles();
        newList = new Vector();

        for (Enumeration e1 = functionList.elements(); e1.hasMoreElements();) {
            FunctionProfile functionProfile = (FunctionProfile) e1.nextElement();
            if (functionProfile != null) {
                if (trial.displayFunction(functionProfile.getFunction())) {
                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                    newList.addElement(ppFunctionProfile);
                }
            }
        }
        Collections.sort(newList);
        return newList;
    }

    public Vector getAllFunctionProfiles() {
        Vector newList = null;
        Vector threads = new Vector();

        edu.uoregon.tau.dms.dss.Thread thread = trial.getDataSource().getMeanData();

        PPThread ppThread = new PPThread(thread, this.trial);

        for (Enumeration e4 = thread.getFunctionProfiles().elements(); e4.hasMoreElements();) {
            FunctionProfile functionProfile = (FunctionProfile) e4.nextElement();
            if ((functionProfile != null) && (trial.displayFunction(functionProfile.getFunction()))) {
                PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                ppThread.addFunction(ppFunctionProfile);
            }
        }
        Collections.sort(ppThread.getFunctionList());
        threads.add(ppThread);

        // reset this in case we are switching metrics
        maxExclusiveSum = 0;

        maxExclusives = new double[trial.getDataSource().getNumFunctions()];

        for (Iterator it = trial.getDataSource().getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                    thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();

                    //Counts the number of ppFunctionProfiles that are actually added.
                    //It is possible (because of selection criteria - groups for example) to filter
                    //out all functions on a particular thread. The default at present is not to add.

                    int counter = 0; //Counts the number of PPFunctionProfile that are actually added.
                    ppThread = new PPThread(thread, this.trial);

                    double sum = 0.0;

                    //Do not add thread to the context until we have verified counter is not zero (done after next loop).
                    //Now enter the thread data loops for this thread.
                    for (Enumeration e4 = thread.getFunctionProfiles().elements(); e4.hasMoreElements();) {
                        FunctionProfile functionProfile = (FunctionProfile) e4.nextElement();
                        if ((functionProfile != null) && (trial.displayFunction(functionProfile.getFunction()))) {
                            PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread,
                                    functionProfile);
                            ppThread.addFunction(ppFunctionProfile);
                            counter++;

                            sum += ppFunctionProfile.getExclusiveValue();

                            maxExclusives[functionProfile.getFunction().getID()] = Math.max(
                                    maxExclusives[functionProfile.getFunction().getID()],
                                    ppFunctionProfile.getExclusiveValue());
                        }
                    }

                    if (sum > maxExclusiveSum) {
                        maxExclusiveSum = sum;
                    }

                    //Sort thread and add to context if required (see above for an explanation).
                    if (counter != 0) {
                        Collections.sort(ppThread.getFunctionList());
                        threads.add(ppThread);
                    }
                }
            }
        }
        return threads;
    }

    public Vector getFunctionData(Function function, boolean includeMean) {
        Vector newList = new Vector();

        edu.uoregon.tau.dms.dss.Thread thread;


        if (includeMean) {
            thread = trial.getDataSource().getMeanData();
            FunctionProfile functionProfile = thread.getFunctionProfile(function);
            if (functionProfile != null) {
                //Create a new thread data object.
                PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                newList.add(ppFunctionProfile);
            }
        }

        for (Iterator it = trial.getDataSource().getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                    thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();
                    FunctionProfile functionProfile = thread.getFunctionProfile(function);
                    if (functionProfile != null) {
                        //Create a new thread data object.
                        PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(this, thread, functionProfile);
                        newList.add(ppFunctionProfile);
                    }
                }
            }
        }
        Collections.sort(newList);
        return newList;
    }

    public Vector getUserEventData(UserEvent userEvent) {
        Vector newList = new Vector();

        UserEventProfile userEventProfile;

        PPUserEventProfile ppUserEventProfile;

        for (Iterator it = trial.getDataSource().getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                    edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();

                    userEventProfile = thread.getUserEventProfile(userEvent);
                    if (userEventProfile != null) {
                        //Create a new thread data object.
                        ppUserEventProfile = new PPUserEventProfile(this, node.getNodeID(),
                                context.getContextID(), thread.getThreadID(), userEventProfile);
                        newList.add(ppUserEventProfile);
                    }
                }
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
    
   
}