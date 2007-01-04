package edu.uoregon.tau.perfdmf;

import java.util.*;
import java.text.*;

/**
 * This class represents a single function profile on a single thread.
 *
 * <P>CVS $Id: FunctionProfile.java,v 1.3 2007/01/04 01:34:36 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.3 $
 * @see		Function
 */
public class FunctionProfile implements Comparable {

    // this is a private static class to save memory when callpath data is not needed
    // we need only one empty pointer instead of four
    private static class CallPathData {
        public Set childProfiles;
        public Set parentProfiles;
        public Map childProfileCallPathSets;
        public Map parentProfileCallPathSets;
    }

    private static final int METRIC_SIZE = 4;
    private static final int CALL_SIZE = 2;

    private static final int INCLUSIVE = 3;
    private static final int EXCLUSIVE = 0;
    private static final int INCLUSIVE_PERCENT = 2;
    private static final int EXCLUSIVE_PERCENT = 1;

    private Function function;
    private double[] data;
    private double[] calls;

    private int numMetrics;

    private CallPathData callPathData;
    private Thread thread;

    public FunctionProfile(Function function) {
        this(function, 1);
    }

    public FunctionProfile(Function function, int numMetrics) {
        this(function, numMetrics, 1);
    }

    public FunctionProfile(Function function, int numMetrics, int snapshots) {
        this.numMetrics = numMetrics;
        data = new double[numMetrics * METRIC_SIZE * snapshots];
        calls = new double[CALL_SIZE * snapshots];
        this.function = function;
    }

    public Function getFunction() {
        return function;
    }

    public String getName() {
        return function.getName();
    }

    private int getNumSnapshots() {
        return calls.length / CALL_SIZE;
    }

    public void setInclusive(int metric, double value) {
        this.putDouble(getNumSnapshots() - 1, metric, INCLUSIVE, value);
    }

    public void setInclusive(int snapshot, int metric, double value) {
        this.putDouble(snapshot, metric, INCLUSIVE, value);
    }

    public double getInclusive(int metric) {
        return this.getDouble(getNumSnapshots() - 1, metric, INCLUSIVE);
    }

    public double getInclusive(int metric, int snapshot) {
        return this.getDouble(snapshot, metric, INCLUSIVE);
    }

    public void setExclusive(int metric, double value) {
        //System.out.println(getName() + " : setExclusiveA(snapshot = " + (getNumSnapshots() - 1) + ", metric = " + metric + ", value = " + value);
        this.putDouble(getNumSnapshots() - 1, metric, EXCLUSIVE, value);
    }

    public void setExclusive(int snapshot, int metric, double value) {
        //System.out.println(getName() + " : setExclusiveB(snapshot = " + snapshot + ", metric = " + metric + ", value = " + value);
        this.putDouble(snapshot, metric, EXCLUSIVE, value);
    }

    public double getExclusive(int snapshot, int metric) {
        return this.getDouble(snapshot, metric, EXCLUSIVE);
    }

    public double getExclusive(int metric) {
        //        if (function.isPhase()) {
        //            return this.getDouble(metric, 0);
        //        } else {
        //            return this.getDouble(metric, 1);
        //        }
        return this.getDouble(getNumSnapshots() - 1, metric, EXCLUSIVE);
    }

    public void setInclusivePercent(int metric, double value) {
        this.putDouble(getNumSnapshots() - 1, metric, INCLUSIVE_PERCENT, value);
    }

    public double getInclusivePercent(int metric) {
        return this.getDouble(getNumSnapshots() - 1, metric, INCLUSIVE_PERCENT);
    }

    public void setExclusivePercent(int metric, double value) {
        this.putDouble(getNumSnapshots() - 1, metric, EXCLUSIVE_PERCENT, value);
    }

    public double getExclusivePercent(int metric) {
        return this.getDouble(getNumSnapshots() - 1, metric, EXCLUSIVE_PERCENT);
    }

    public void setNumCalls(double value) {
        calls[0] = value;
    }

    public double getNumCalls() {
        return calls[0];
    }

    public void setNumSubr(double value) {
        calls[1] = value;
    }

    public double getNumSubr() {
        return calls[1];
    }

    public double getInclusivePerCall(int metric) {
        if (this.getNumCalls() == 0) {
            return 0;
        }
        return this.getInclusive(metric) / this.getNumCalls();
    }

    public double getExclusivePerCall(int metric) {
        if (this.getNumCalls() == 0) {
            return 0;
        }
        return this.getExclusive(metric) / this.getNumCalls();
    }

    // unused profile calls
    //    public void addCall(double exclusive, double inclusive) {
    //        if (calls == null)
    //            calls = new ArrayList();
    //        double[] arr = new double[2];
    //        arr[0] = exclusive;
    //        arr[1] = inclusive;
    //        calls.add(arr);
    //    }

    public int getNumMetrics() {
        //return doubleList.length / METRIC_SIZE;
        return numMetrics;
    }

    public void addMetric() {
        int newMetricSize = numMetrics + 1;
        int currentLength = data.length;
        int numSnapshots = calls.length / CALL_SIZE;

        double[] newArray = new double[newMetricSize * METRIC_SIZE * numSnapshots];

        for (int s = 0; s < numSnapshots; s++) {

            int source = (s * METRIC_SIZE * numMetrics);
            int dest = (s * METRIC_SIZE * newMetricSize);
            for (int m = 0; m < METRIC_SIZE * numMetrics; m++) {
                newArray[dest + m] = data[source + m];
            }

        }

        data = newArray;
        numMetrics = newMetricSize;
    }

    public void addSnapshot() {

        int currentLength = data.length;
        double[] newArray = new double[currentLength + (METRIC_SIZE * numMetrics)];

        for (int i = 0; i < currentLength; i++) {
            newArray[i] = data[i];
        }
        data = newArray;

        int newCallsLength = calls.length + CALL_SIZE;
        double[] newCalls = new double[newCallsLength];

        for (int i = 0; i < calls.length; i++) {
            newCalls[i] = calls[i];
        }

        calls = newCalls;
    }

    // call path section
    public void addChildProfile(FunctionProfile child, FunctionProfile callpath) {
        // example:
        // callpath: a => b => c => d
        // child: d
        // this: c
        CallPathData callPathData = getCallPathData();

        if (callPathData.childProfiles == null)
            callPathData.childProfiles = new TreeSet();
        callPathData.childProfiles.add(child);

        if (callPathData.childProfileCallPathSets == null)
            callPathData.childProfileCallPathSets = new TreeMap();

        // we maintain a set of callpaths for each child, retrieve the set for this child
        Set callPathSet = (Set) callPathData.childProfileCallPathSets.get(child);

        if (callPathSet == null) {
            callPathSet = new TreeSet();
            callPathData.childProfileCallPathSets.put(child, callPathSet);
        }

        callPathSet.add(callpath);
    }

    public void addParentProfile(FunctionProfile parent, FunctionProfile callpath) {
        // example:
        // callpath: a => b => c => d
        // parent: c
        // this: d

        CallPathData callPathData = getCallPathData();

        if (callPathData.parentProfiles == null)
            callPathData.parentProfiles = new TreeSet();
        callPathData.parentProfiles.add(parent);

        if (callPathData.parentProfileCallPathSets == null)
            callPathData.parentProfileCallPathSets = new TreeMap();

        // we maintain a set of callpaths for each child, retrieve the set for this child
        Set callPathSet = (Set) callPathData.parentProfileCallPathSets.get(parent);

        if (callPathSet == null) {
            callPathSet = new TreeSet();
            callPathData.parentProfileCallPathSets.put(parent, callPathSet);
        }

        callPathSet.add(callpath);
    }

    public Iterator getChildProfiles() {
        CallPathData callPathData = getCallPathData();
        if (callPathData.childProfiles != null)
            return callPathData.childProfiles.iterator();
        return new UtilFncs.EmptyIterator();
    }

    public Iterator getParentProfiles() {
        CallPathData callPathData = getCallPathData();
        if (callPathData.parentProfiles != null)
            return callPathData.parentProfiles.iterator();
        return new UtilFncs.EmptyIterator();
    }

    public Iterator getParentProfileCallPathIterator(FunctionProfile parent) {
        CallPathData callPathData = getCallPathData();
        if (callPathData.parentProfileCallPathSets == null)
            return new UtilFncs.EmptyIterator();
        return ((Set) callPathData.parentProfileCallPathSets.get(parent)).iterator();
    }

    public Iterator getChildProfileCallPathIterator(FunctionProfile child) {
        CallPathData callPathData = getCallPathData();
        if (callPathData.childProfileCallPathSets == null)
            return new UtilFncs.EmptyIterator();
        return ((Set) callPathData.childProfileCallPathSets.get(child)).iterator();
    }

    /**
     * Passthrough to the actual function's isCallPathFunction
     * 
     * @return		whether or not this function is a callpath (contains '=>')
     */
    public boolean isCallPathFunction() {
        return function.isCallPathFunction();
    }

    private void putDouble(int snapshot, int metric, int offset, double inDouble) {
        int actualLocation = (snapshot * METRIC_SIZE * numMetrics) + (metric * METRIC_SIZE) + offset;
        data[actualLocation] = inDouble;
    }

    private double getDouble(int snapshot, int metric, int offset) {
        int location = (snapshot * METRIC_SIZE * numMetrics) + (metric * METRIC_SIZE) + offset;
        return data[location];
    }

    public int compareTo(Object inObject) {
        return this.function.compareTo(((FunctionProfile) inObject).function);
    }

    public String toString() {
        return "A FunctionProfile for " + function.toString();
    }

    private CallPathData getCallPathData() {
        if (callPathData == null) {
            callPathData = new CallPathData();
        }
        return callPathData;
    }

    public Thread getThread() {
        return thread;
    }

    public void setThread(Thread thread) {
        this.thread = thread;
    }
}