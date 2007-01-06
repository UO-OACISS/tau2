package edu.uoregon.tau.perfdmf;

import java.util.*;
import java.io.*;

/**
 * This class represents a Thread.  It contains an array of FunctionProfiles and 
 * UserEventProfiles as well as maximum data (e.g. max exclusive value for all functions on 
 * this thread). 
 *  
 * <P>CVS $Id: Thread.java,v 1.5 2007/01/06 04:40:58 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.5 $
 * @see		Node
 * @see		Context
 * @see		FunctionProfile
 * @see		UserEventProfile
 */
public class Thread implements Comparable {

    private int nodeID, contextID, threadID;
    private List functionProfiles = new ArrayList();
    private List userEventProfiles = new ArrayList();
    private double[] maxData;
    private double maxNumCalls = 0;
    private double maxNumSubr = 0;
    private boolean trimmed = false;
    private boolean relationsBuilt = false;
    private int numMetrics = 1;
    private static final int METRIC_SIZE = 7;
    
    public static final int MEAN = -1;
    public static final int TOTAL = -2;
    public static final int STDDEV = -3;
    
    private List snapshots = new ArrayList();

    //    
    //    public Thread(int nodeID, int contextID, int threadID) {
    //        this(nodeID, contextID, threadID, 1);
    //    }

    public Thread(int nodeID, int contextID, int threadID, int numMetrics) {
        numMetrics = Math.max(numMetrics, 1);
        this.nodeID = nodeID;
        this.contextID = contextID;
        this.threadID = threadID;
        maxData = new double[numMetrics * METRIC_SIZE];
        this.numMetrics = numMetrics;
    }

    public String toString() {
        if (nodeID == -1) {
            return "Mean";
        }
        if (nodeID == -3) {
            return "Standard Deviation";
        }
        return "n,c,t " + nodeID + "," + contextID + "," + threadID;
    }

    public int getNodeID() {
        return nodeID;
    }

    public int getContextID() {
        return contextID;
    }

    public int getThreadID() {
        return threadID;
    }

    public int getNumMetrics() {
        return numMetrics;
    }

    public void addMetric() {
        int currentLength = maxData.length;
        double[] newArray = new double[currentLength + METRIC_SIZE];

        for (int i = 0; i < currentLength; i++) {
            newArray[i] = maxData[i];
        }
        maxData = newArray;
        numMetrics++;

        for (Iterator it = getFunctionProfiles().iterator(); it.hasNext();) {
            FunctionProfile fp = (FunctionProfile) it.next();
            if (fp != null) { // fp == null would mean this thread didn't call this function
                fp.addMetric();
            }
        }
    }

  

    public Snapshot addSnapshot(String name) {
        Snapshot snapshot = new Snapshot(name, snapshots.size());
        snapshots.add(snapshot);
        
        
        if (snapshots.size() > 1) {
            for (Iterator e6 = functionProfiles.iterator(); e6.hasNext();) {
                FunctionProfile fp = (FunctionProfile) e6.next();
                if (fp != null) { // fp == null would mean this thread didn't call this function
                    fp.addSnapshot();
                }
            }
        }
        
        return snapshot;
    }

    public List getSnapshots() {
        return snapshots;
    }

    public int getNumSnapshots() {
        return Math.max(1,snapshots.size());
    }

    public void addFunctionProfile(FunctionProfile fp) {
        int id = fp.getFunction().getID();
        // increase the size of the functionProfiles list if necessary
        while (id >= functionProfiles.size()) {
            functionProfiles.add(null);
        }

        functionProfiles.set(id, fp);
        fp.setThread(this);
    }

    public void addUserEventProfile(UserEventProfile uep) {
        int id = uep.getUserEvent().getID();
        // increase the userEventProfiles vector size if necessary

        while (id >= userEventProfiles.size()) {
            userEventProfiles.add(null);
        }

        userEventProfiles.set(id, uep);
    }

    public FunctionProfile getFunctionProfile(Function function) {
        if ((functionProfiles != null) && (function.getID() < functionProfiles.size()))
            return (FunctionProfile) functionProfiles.get(function.getID());
        return null;
    }

    public List getFunctionProfiles() {
        return functionProfiles;
    }

    public Iterator getFunctionProfileIterator() {
        return functionProfiles.iterator();
    }

    public UserEventProfile getUserEventProfile(UserEvent userEvent) {
        if ((userEventProfiles != null) && (userEvent.getID() < userEventProfiles.size()))
            return (UserEventProfile) userEventProfiles.get(userEvent.getID());
        return null;
    }

    public List getUserEventProfiles() {
        return userEventProfiles;
    }

    private void setMaxInclusive(int metric, double inDouble) {
        this.insertDouble(metric, 0, inDouble);
    }

    public double getMaxInclusive(int metric) {
        return this.getDouble(metric, 0);
    }

    private void setMaxExclusive(int metric, double inDouble) {
        this.insertDouble(metric, 1, inDouble);
    }

    public double getMaxExclusive(int metric) {
        return this.getDouble(metric, 1);
    }

    private void setMaxInclusivePercent(int metric, double inDouble) {
        this.insertDouble(metric, 2, inDouble);
    }

    public double getMaxInclusivePercent(int metric) {
        return this.getDouble(metric, 2);
    }

    private void setMaxExclusivePercent(int metric, double inDouble) {
        this.insertDouble(metric, 3, inDouble);
    }

    public double getMaxExclusivePercent(int metric) {
        return this.getDouble(metric, 3);
    }

    private void setMaxInclusivePerCall(int metric, double inDouble) {
        this.insertDouble(metric, 4, inDouble);
    }

    public double getMaxInclusivePerCall(int metric) {
        return this.getDouble(metric, 4);
    }

    private void setMaxExclusivePerCall(int metric, double inDouble) {
        this.insertDouble(metric, 5, inDouble);
    }

    public double getMaxExclusivePerCall(int metric) {
        return this.getDouble(metric, 5);
    }

    public void setPercentDivider(int metric, double inDouble) {
        this.insertDouble(metric, 6, inDouble);
    }

    public double getPercentDivider(int metric) {
        return this.getDouble(metric, 6);
    }

    private void setMaxNumCalls(double inDouble) {
        maxNumCalls = inDouble;
    }

    public double getMaxNumCalls() {
        return maxNumCalls;
    }

    private void setMaxNumSubr(double inDouble) {
        maxNumSubr = inDouble;
    }

    public double getMaxNumSubr() {
        return maxNumSubr;
    }

    // Since per thread callpath relations are build on demand, the following four functions tell whether this
    // thread's callpath information has been set yet.  This way, we only compute it once.
    public void setTrimmed(boolean b) {
        trimmed = b;
    }

    public boolean trimmed() {
        return trimmed;
    }

    public void setRelationsBuilt(boolean b) {
        relationsBuilt = b;
    }

    public boolean relationsBuilt() {
        return relationsBuilt;
    }

    public int compareTo(Object obj) {
        return threadID - ((Thread) obj).getThreadID();
    }

    //    public String toString() {
    //        return this.getClass().getName() + ": " + this.getNodeID() + "," + this.getContextID() + ","
    //                + this.getThreadID();
    //    }

    public void setThreadData(int metric) {
        setThreadValues(metric, metric);
    }

    public void setThreadDataAllMetrics() {
        setThreadValues(0, this.getNumMetrics() - 1);
    }

    private void insertDouble(int metric, int offset, double inDouble) {
        int actualLocation = (metric * METRIC_SIZE) + offset;
        maxData[actualLocation] = inDouble;
    }

    private double getDouble(int metric, int offset) {
        int actualLocation = (metric * METRIC_SIZE) + offset;
        return maxData[actualLocation];
    }

    // compute max values and percentages for threads (not mean/total)
    private void setThreadValues(int startMetric, int endMetric) {
        for (int metric = startMetric; metric <= endMetric; metric++) {
            double maxInclusive = 0.0;
            double maxExclusive = 0.0;
            double maxInclusivePerCall = 0.0;
            double maxExclusivePerCall = 0.0;
            double maxNumCalls = 0;
            double maxNumSubr = 0;

            for (Iterator it = this.getFunctionProfileIterator(); it.hasNext();) {
                FunctionProfile fp = (FunctionProfile) it.next();
                if (fp != null) {
                    if (fp.getFunction().isPhase()) {
                        maxExclusive = Math.max(maxExclusive, fp.getInclusive(metric));
                        maxExclusivePerCall = Math.max(maxExclusivePerCall, fp.getInclusivePerCall(metric));
                    } else {
                        maxExclusive = Math.max(maxExclusive, fp.getExclusive(metric));
                        maxExclusivePerCall = Math.max(maxExclusivePerCall, fp.getExclusivePerCall(metric));
                    }
                    maxInclusive = Math.max(maxInclusive, fp.getInclusive(metric));
                    maxInclusivePerCall = Math.max(maxInclusivePerCall, fp.getInclusivePerCall(metric));
                    maxNumCalls = Math.max(maxNumCalls, fp.getNumCalls());
                    maxNumSubr = Math.max(maxNumSubr, fp.getNumSubr());
                }
            }

            this.setMaxInclusive(metric, maxInclusive);
            this.setMaxExclusive(metric, maxExclusive);
            this.setMaxInclusivePerCall(metric, maxInclusivePerCall);
            this.setMaxExclusivePerCall(metric, maxExclusivePerCall);
            this.setMaxNumCalls(maxNumCalls);
            this.setMaxNumSubr(maxNumSubr);

            double maxInclusivePercent = 0.0;
            double maxExclusivePercent = 0.0;

            for (Iterator it = this.getFunctionProfileIterator(); it.hasNext();) {
                FunctionProfile fp = (FunctionProfile) it.next();
                if (fp != null) {

                    // Note: Assumption is made that the max inclusive value is the value required to calculate
                    // percentage (ie, divide by). Thus, we are assuming that the sum of the exclusive
                    // values is equal to the max inclusive value. This is a reasonable assumption. This also gets
                    // us out of sticky situations when call path data is present (this skews attempts to calculate
                    // the total exclusive value unless checks are made to ensure that we do not include call path objects).

                    Function function = fp.getFunction();
                    if (this.getNodeID() > -1) { // don't do this for mean/total
                        double inclusiveMax = this.getMaxInclusive(metric);

                        setPercentDivider(metric, inclusiveMax / 100.0);
                        
                        if (inclusiveMax != 0) {
                            double exclusivePercent = (fp.getExclusive(metric) / inclusiveMax) * 100;
                            double inclusivePercent = (fp.getInclusive(metric) / inclusiveMax) * 100;

                            fp.setExclusivePercent(metric, exclusivePercent);
                            fp.setInclusivePercent(metric, inclusivePercent);
                            //function.setMaxExclusivePercent(metric, Math.max(function.getMaxExclusivePercent(metric), exclusivePercent));
                            //function.setMaxInclusivePercent(metric, Math.max(function.getMaxInclusivePercent(metric), inclusivePercent));
                        }
                    }

                    maxExclusivePercent = Math.max(maxExclusivePercent, fp.getExclusivePercent(metric));
                    maxInclusivePercent = Math.max(maxInclusivePercent, fp.getInclusivePercent(metric));
                }
            }

            this.setMaxInclusivePercent(metric, maxInclusivePercent);
            this.setMaxExclusivePercent(metric, maxExclusivePercent);
        }
    }

}