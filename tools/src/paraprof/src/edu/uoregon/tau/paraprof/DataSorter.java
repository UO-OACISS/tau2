package edu.uoregon.tau.paraprof;

import java.util.*;
import edu.uoregon.tau.dms.dss.*;

public class DataSorter {

    public DataSorter(ParaProfTrial trial) {
        this.trial = trial;
    }

    public Vector getUserEventProfiles(int nodeID, int contextID, int threadID, int sortType) {

        UserEventProfile userEventProfile = null;
        Vector list = ((edu.uoregon.tau.dms.dss.Thread) trial.getNCT().getThread(nodeID, contextID, threadID)).getUsereventList();

        Vector newList = new Vector();

        for (Enumeration e1 = list.elements(); e1.hasMoreElements();) {
            userEventProfile = (UserEventProfile) e1.nextElement();
            if (userEventProfile != null) {
                //        if (trialData.displayMapping(userEventProfile.getMappingID())) {
                PPUserEventProfile ppUserEventProfile = new PPUserEventProfile(trial, nodeID, contextID,
                        threadID, userEventProfile);
                ppUserEventProfile.setSortType(sortType);
                newList.addElement(ppUserEventProfile);
                //      }
            }
        }
        Collections.sort(newList);
        return newList;
    }

    public Vector getFunctionProfiles(int nodeID, int contextID, int threadID, int sortType) {
        TrialData trialData = trial.getTrialData();
        Vector newList = null;

        edu.uoregon.tau.dms.dss.Thread thread;

        if (nodeID == -1) { // mean
            thread = trial.getDataSource().getMeanData();
        } else {
            thread = trial.getNCT().getThread(nodeID, contextID, threadID);
        }
        Vector functionList = thread.getFunctionList();
        newList = new Vector();

        for (Enumeration e1 = functionList.elements(); e1.hasMoreElements();) {
            FunctionProfile functionProfile = (FunctionProfile) e1.nextElement();
            if (functionProfile != null) {
                if (trialData.displayFunction(functionProfile.getFunction())) {
                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(trial, nodeID, contextID,
                            threadID, functionProfile);
                    ppFunctionProfile.setSortType(sortType);
                    newList.addElement(ppFunctionProfile);
                }
            }
        }
        Collections.sort(newList);
        return newList;
    }

    public Vector getAllFunctionProfiles(int sortType) {
        Vector newList = null;
        TrialData trialData = trial.getTrialData();
        Vector threads = new Vector();

        edu.uoregon.tau.dms.dss.Thread thread = trial.getDataSource().getMeanData();

        PPThread ppThread = new PPThread(thread);

        for (Enumeration e4 = thread.getFunctionList().elements(); e4.hasMoreElements();) {
            FunctionProfile functionProfile = (FunctionProfile) e4.nextElement();
            if ((functionProfile != null) && (trialData.displayFunction(functionProfile.getFunction()))) {
                PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(trial, -1, -1, -1, functionProfile);
                ppFunctionProfile.setSortType(sortType);
                ppThread.addFunction(ppFunctionProfile);
            }
        }
        Collections.sort(ppThread.getFunctionList());
        threads.add(ppThread);

        // reset this in case we are switching metrics
        maxExclusiveSum = 0;

        for (Enumeration e1 = trial.getNCT().getNodes().elements(); e1.hasMoreElements();) {
            Node node = (Node) e1.nextElement();
            for (Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements();) {
                Context context = (Context) e2.nextElement();
                for (Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements();) {
                    //Counts the number of ppFunctionProfiles that are actually added.
                    //It is possible (because of selection criteria - groups for example) to filter
                    //out all mappings on a particular thread. The default at present is not to add.

                    int counter = 0; //Counts the number of PPFunctionProfile that are actually added.
                    thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
                    ppThread = new PPThread(thread);

                    double sum = 0.0;

                    //Do not add thread to the context until we have verified counter is not zero (done after next loop).
                    //Now enter the thread data loops for this thread.
                    for (Enumeration e4 = thread.getFunctionList().elements(); e4.hasMoreElements();) {
                        FunctionProfile functionProfile = (FunctionProfile) e4.nextElement();
                        if ((functionProfile != null)
                                && (trialData.displayFunction(functionProfile.getFunction()))) {
                            PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(trial,
                                    node.getNodeID(), context.getContextID(), thread.getThreadID(),
                                    functionProfile);
                            ppFunctionProfile.setSortType(sortType);
                            ppThread.addFunction(ppFunctionProfile);
                            counter++;

                            sum += ppFunctionProfile.getExclusiveValue();

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

    public Vector getFunctionData(Function function, int sortType, boolean mean) {
        Vector newList = new Vector();

        Node node;
        Context context;
        edu.uoregon.tau.dms.dss.Thread thread;
        FunctionProfile functionProfile;

        PPFunctionProfile ppFunctionProfile;

        if (mean) {
            // add mean first
            thread = trial.getDataSource().getMeanData();
            functionProfile = thread.getFunctionProfile(function);
            if (functionProfile != null) {
                //Create a new thread data object.
                ppFunctionProfile = new PPFunctionProfile(trial, -1, -1, -1, functionProfile);
                ppFunctionProfile.setSortType(sortType);
                newList.add(ppFunctionProfile);
            }
        }

        for (Enumeration e1 = trial.getNCT().getNodes().elements(); e1.hasMoreElements();) {
            node = (Node) e1.nextElement();
            for (Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements();) {
                context = (Context) e2.nextElement();
                for (Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements();) {
                    thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
                    functionProfile = thread.getFunctionProfile(function);
                    if (functionProfile != null) {
                        //Create a new thread data object.
                        ppFunctionProfile = new PPFunctionProfile(trial, node.getNodeID(),
                                context.getContextID(), thread.getThreadID(), functionProfile);
                        ppFunctionProfile.setSortType(sortType);
                        newList.add(ppFunctionProfile);
                    }
                }
            }
        }
        Collections.sort(newList);
        return newList;
    }

    public Vector getUserEventData(UserEvent userEvent, int sortType) {
        Vector newList = new Vector();

        Node node;
        Context context;
        edu.uoregon.tau.dms.dss.Thread thread;
        UserEventProfile userEventProfile;

        PPUserEventProfile ppUserEventProfile;

        for (Enumeration e1 = trial.getNCT().getNodes().elements(); e1.hasMoreElements();) {
            node = (Node) e1.nextElement();
            for (Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements();) {
                context = (Context) e2.nextElement();
                for (Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements();) {
                    thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();

                    userEventProfile = thread.getUserEvent(userEvent.getID());
                    if (userEventProfile != null) {
                        //Create a new thread data object.
                        ppUserEventProfile = new PPUserEventProfile(trial, node.getNodeID(),
                                context.getContextID(), thread.getThreadID(), userEventProfile);
                        ppUserEventProfile.setSortType(sortType);
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

    private ParaProfTrial trial = null;
    private double maxExclusiveSum = 0;
}