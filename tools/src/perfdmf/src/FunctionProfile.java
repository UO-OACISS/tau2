package edu.uoregon.tau.perfdmf;

import java.util.*;

/**
 * This class represents a single function profile on a single thread.
 *
 * <P>CVS $Id: FunctionProfile.java,v 1.15 2010/01/14 01:29:16 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.15 $
 * @see		Function
 */
public class FunctionProfile {

    // this is a private static class to save memory when callpath data is not needed
    // we need only one empty pointer instead of four
    private static class CallPathData {
        public Set<FunctionProfile> childProfiles;
        public Set<FunctionProfile> parentProfiles;
        public Map<FunctionProfile, Set<FunctionProfile>> childProfileCallPathSets;
        public Map<FunctionProfile, Set<FunctionProfile>> parentProfileCallPathSets;
    }

    private static final int METRIC_SIZE = 2;

    private static final int CALLS = 0;
    private static final int SUBR = 1;
    private static final int INCLUSIVE = 2;
    private static final int EXCLUSIVE = 3;

    private Function function;
    private Thread thread;
    private double[] data;
    private CallPathData callPathData;

    public FunctionProfile(Function function) {
        this(function, 1);
    }

    public FunctionProfile(Function function, int numMetrics) {
        this(function, numMetrics, 1);
    }

    public FunctionProfile(Function function, int numMetrics, int snapshots) {
        numMetrics = Math.max(numMetrics, 1);
        data = new double[((numMetrics + 1) * METRIC_SIZE + 2) * snapshots];

        this.function = function;
    }

    public Function getFunction() {
        return function;
    }

    public String getName() {
        return function.getName();
    }

    public void setInclusive(int metric, double value) {
        this.putDouble(metric, INCLUSIVE, value);
    }

    public void setInclusive(int snapshot, int metric, double value) {
        this.putDouble(snapshot, metric, INCLUSIVE, value);
    }

    public double getInclusive(int metric) {
        return this.getDouble(metric, INCLUSIVE);
    }

    public double getInclusive(int snapshot, int metric) {
        return this.getDouble(snapshot, metric, INCLUSIVE);
    }

    public void setExclusive(int metric, double value) {
        this.putDouble(metric, EXCLUSIVE, value);
    }

    public void setExclusive(int snapshot, int metric, double value) {
        this.putDouble(snapshot, metric, EXCLUSIVE, value);
    }

    public double getExclusive(int snapshot, int metric) {
        return this.getDouble(snapshot, metric, EXCLUSIVE);
    }

    public double getExclusive(int metric) {
        double value = this.getDouble(metric, EXCLUSIVE);
        return value;
    }

    // TODO: A bunch of these methods are not using the snapshot
    public double getInclusivePercent(int snapshot, int metric) {
        if (snapshot == -1) {
            snapshot = thread.getNumSnapshots() - 1;
        }
        if (thread.getNodeID() >= 0) {
            double dividend = thread.getPercentDivider(metric, snapshot);
            if (dividend == 0) {
                return 0;
            }
            return getInclusive(snapshot, metric) / dividend;
        } else if (thread.getNodeID() == Thread.TOTAL || thread.getNodeID() == Thread.MEAN) {
            double dividend = thread.getPercentDivider(metric, snapshot);
            if (dividend == 0) {
                return 0;
            }
            return function.getTotalProfile().getInclusive(snapshot, metric) / dividend;
        } else if (thread.getNodeID() == Thread.STDDEV) {
            return getInclusive(snapshot, metric) / function.getMeanInclusive(metric) * 100.0;
        }
        throw new RuntimeException("Bad Thread ID = " + thread);

    }

    public double getInclusivePercent(int metric) {
        return getInclusivePercent(-1, metric);
    }

    // TODO: A bunch of these methods are not using the snapshot
    public double getExclusivePercent(int snapshot, int metric) {
        if (snapshot == -1) {
            snapshot = thread.getNumSnapshots() - 1;
        }
        if (thread.getNodeID() >= 0) {
            double dividend = thread.getPercentDivider(metric, snapshot);
            if (dividend == 0) {
                return 0;
            }
            return getExclusive(snapshot, metric) / dividend;
        } else if (thread.getNodeID() == Thread.TOTAL || thread.getNodeID() == Thread.MEAN) {
            double dividend = thread.getPercentDivider(metric, snapshot);
            if (dividend == 0) {
                return 0;
            }
            if (thread.getNodeID() == Thread.TOTAL) {
                return function.getTotalProfile().getExclusive(snapshot, metric) / dividend;
            } else {
                dividend /= thread.getDataSource().getAllThreads().size();
                return function.getMeanProfile().getExclusive(snapshot, metric) / dividend;
            }
        } else if (thread.getNodeID() == Thread.STDDEV) {
            return getExclusive(metric) / function.getMeanExclusive(metric) * 100.0;
        }
        throw new RuntimeException("Bad Thread ID = " + thread);
    }

    public double getExclusivePercent(int metric) {
        return getExclusivePercent(-1, metric);
    }

    public void setNumCalls(int snapshot, double value) {
        this.putDouble(snapshot, 0, CALLS, value);
    }

    public void setNumCalls(double value) {
        this.putDouble(0, CALLS, value);
    }

    public double getNumCalls() {
        return getNumCalls(-1);
    }

    public double getNumCalls(int snapshot) {
        if (snapshot == -1) {
            snapshot = thread.getNumSnapshots() - 1;
        }
        return getDouble(snapshot, 0, CALLS);
    }

    public void setNumSubr(int snapshot, double value) {
        this.putDouble(snapshot, 0, SUBR, value);
    }

    public void setNumSubr(double value) {
        this.putDouble(0, SUBR, value);
    }

    public double getNumSubr() {
        return getNumSubr(-1);
    }

    public double getNumSubr(int snapshot) {
        if (snapshot == -1) {
            snapshot = thread.getNumSnapshots() - 1;
        }
        return getDouble(snapshot, 0, SUBR);
    }

    public double getInclusivePerCall(int metric) {
        return getInclusivePerCall(-1, metric);
    }

    public double getInclusivePerCall(int snapshot, int metric) {
        if (snapshot == -1) {
            snapshot = thread.getNumSnapshots() - 1;
        }
        if (this.getNumCalls(snapshot) == 0) {
            return 0;
        }
        return this.getInclusive(snapshot, metric) / this.getNumCalls(snapshot);
    }

    public double getExclusivePerCall(int metric) {
        return getExclusivePerCall(-1, metric);
    }

    public double getExclusivePerCall(int snapshot, int metric) {
        if (snapshot == -1) {
            snapshot = thread.getNumSnapshots() - 1;
        }
        if (this.getNumCalls(snapshot) == 0) {
            return 0;
        }
        return this.getExclusive(snapshot, metric) / this.getNumCalls(snapshot);
    }

    public void addMetric() {
        int numMetrics = thread.getNumMetrics() - 1;
        int numSnapshots = thread.getNumSnapshots();
        int newMetricSize = numMetrics + 1;

        double[] newArray = new double[(newMetricSize + 1) * METRIC_SIZE * numSnapshots];

        for (int s = 0; s < numSnapshots; s++) {
            int source = (s * METRIC_SIZE * (numMetrics + 1));
            int dest = (s * METRIC_SIZE * (newMetricSize + 1));
            for (int m = 0; m < METRIC_SIZE * (numMetrics + 1); m++) {
                newArray[dest + m] = data[source + m];
            }
        }
        data = newArray;
    }

    public void addSnapshot() {
        //        int newCallsLength = thread.getNumSnapshots() * CALL_SIZE;
        //        if (newCallsLength > calls.length) {
        //            // could only do this with Java 1.6 :(
        //            //calls = Arrays.copyOf(calls, (int)(newCallsLength*1.5));
        //            double[] newCalls = new double[(int) (newCallsLength * 1.5)];
        //            System.arraycopy(calls, 0, newCalls, 0, calls.length);
        //            calls = newCalls;
        //        }

        int numMetrics = thread.getNumMetrics();
        int newLength = thread.getNumSnapshots() * ((METRIC_SIZE * numMetrics) + 2);
        if (newLength > data.length) {
            // could only do this with Java 1.6 :(
            //data = Arrays.copyOf(data, (int)(newLength*1.5));
            double[] newArray = new double[(int) (newLength * 1.5)];
            System.arraycopy(data, 0, newArray, 0, data.length);
            data = newArray;
        }
    }

    // call path section
    public void addChildProfile(FunctionProfile child, FunctionProfile callpath) {
        // example:
        // callpath: a => b => c => d
        // child: d
        // this: c
        CallPathData callPathData = getCallPathData();

        if (callPathData.childProfiles == null)
            callPathData.childProfiles = new HashSet<FunctionProfile>();
        callPathData.childProfiles.add(child);

        if (callPathData.childProfileCallPathSets == null)
            callPathData.childProfileCallPathSets = new HashMap<FunctionProfile, Set<FunctionProfile>>();

        // we maintain a set of callpaths for each child, retrieve the set for this child
        Set<FunctionProfile> callPathSet = callPathData.childProfileCallPathSets.get(child);

        if (callPathSet == null) {
            callPathSet = new HashSet<FunctionProfile>();
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
            callPathData.parentProfiles = new HashSet<FunctionProfile>();
        callPathData.parentProfiles.add(parent);

        if (callPathData.parentProfileCallPathSets == null)
            callPathData.parentProfileCallPathSets = new HashMap<FunctionProfile, Set<FunctionProfile>>();

        // we maintain a set of callpaths for each child, retrieve the set for this child
        Set<FunctionProfile> callPathSet = callPathData.parentProfileCallPathSets.get(parent);

        if (callPathSet == null) {
            callPathSet = new HashSet<FunctionProfile>();
            callPathData.parentProfileCallPathSets.put(parent, callPathSet);
        }

        callPathSet.add(callpath);
    }

    public Iterator<FunctionProfile> getChildProfiles() {
        CallPathData callPathData = getCallPathData();
        if (callPathData.childProfiles != null)
            return callPathData.childProfiles.iterator();
        return new UtilFncs.EmptyIterator();
    }

    public Iterator<FunctionProfile> getParentProfiles() {
        CallPathData callPathData = getCallPathData();
        if (callPathData.parentProfiles != null)
            return callPathData.parentProfiles.iterator();
        return new UtilFncs.EmptyIterator();
    }

    public Iterator getParentProfileCallPathIterator(FunctionProfile parent) {
        CallPathData callPathData = getCallPathData();
        if (callPathData.parentProfileCallPathSets == null)
            return new UtilFncs.EmptyIterator();
        return callPathData.parentProfileCallPathSets.get(parent).iterator();
    }

    public Iterator getChildProfileCallPathIterator(FunctionProfile child) {
        CallPathData callPathData = getCallPathData();
        if (callPathData.childProfileCallPathSets == null)
            return new UtilFncs.EmptyIterator();
        return callPathData.childProfileCallPathSets.get(child).iterator();
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
        int numMetrics = thread.getNumMetrics();
        int location = (snapshot * (METRIC_SIZE * (numMetrics + 1))) + (metric * METRIC_SIZE) + offset;
        data[location] = inDouble;
    }

    private void putDouble(int metric, int offset, double inDouble) {
        int snapshot = thread.getNumSnapshots() - 1;
        int numMetrics = thread.getNumMetrics();
        int location = (snapshot * (METRIC_SIZE * (numMetrics + 1))) + (metric * METRIC_SIZE) + offset;
        data[location] = inDouble;
    }

    private double getDouble(int snapshot, int metric, int offset) {
        if (snapshot == -1) {
            snapshot = thread.getNumSnapshots() - 1;
        }

        int numMetrics = thread.getNumMetrics();
        int location = (snapshot * (METRIC_SIZE * (numMetrics + 1))) + (metric * METRIC_SIZE) + offset;
        return data[location];
    }

    private double getDouble(int metric, int offset) {
        // use the last snapshot (final value)
        int snapshot = thread.getNumSnapshots() - 1;
        int numMetrics = thread.getNumMetrics();
        int location = (snapshot * (METRIC_SIZE * (numMetrics + 1))) + (metric * METRIC_SIZE) + offset;
        return data[location];
    }

    public String toString() {
        return thread + " : " + function;
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