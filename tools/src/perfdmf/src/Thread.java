package edu.uoregon.tau.perfdmf;

import java.util.*;

import edu.uoregon.tau.common.TauRuntimeException;

/**
 * This class represents a Thread.  It contains an array of FunctionProfiles and 
 * UserEventProfiles as well as maximum data (e.g. max exclusive value for all functions on 
 * this thread). 
 *  
 * <P>CVS $Id: Thread.java,v 1.14 2008/05/14 23:14:01 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.14 $
 * @see		Node
 * @see		Context
 * @see		FunctionProfile
 * @see		UserEventProfile
 */
public class Thread implements Comparable {

    private int nodeID, contextID, threadID;
    private List functionProfiles = new ArrayList();
    private List userEventProfiles = new ArrayList();
    private boolean trimmed;
    private boolean relationsBuilt;
    private int numMetrics;

    public static final int MEAN = -1;
    public static final int TOTAL = -2;
    public static final int STDDEV = -3;

    private List snapshots = new ArrayList();
    private Map metaData = new TreeMap();

    private boolean firstSnapshotFound;

    // two dimensional, snapshots x metrics
    private ThreadData[][] threadData;

    private long startTime;
    private DataSource dataSource;

    private static class ThreadData {

        public double maxNumCalls;
        public double maxNumSubr;

        public double maxInclusive;
        public double maxInclusivePercent;
        public double maxInclusivePerCall;
        public double maxExclusive;
        public double maxExclusivePercent;
        public double maxExclusivePerCall;

        public double percentDivider;
    }

    public Thread(int nodeID, int contextID, int threadID, int numMetrics, DataSource dataSource) {
        numMetrics = Math.max(numMetrics, 1);
        this.nodeID = nodeID;
        this.contextID = contextID;
        this.threadID = threadID;
        //maxData = new double[numMetrics * METRIC_SIZE];
        this.numMetrics = numMetrics;
        this.dataSource = dataSource;
        if (dataSource == null) {
            throw new TauRuntimeException("Error: dataSource should never be null in Thread constructor");
        }
        
        recreateData();

        // create the first snapshot
        Snapshot snapshot = new Snapshot("", snapshots.size());
        snapshots.add(snapshot);
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
        numMetrics++;

        recreateData();

        for (Iterator it = getFunctionProfiles().iterator(); it.hasNext();) {
            FunctionProfile fp = (FunctionProfile) it.next();
            if (fp != null) { // fp == null would mean this thread didn't call this function
                fp.addMetric();
            }
        }
    }

    public Snapshot addSnapshot(String name) {

        if (!firstSnapshotFound) {
            firstSnapshotFound = true;
            Snapshot snapshot = (Snapshot) snapshots.get(0);
            snapshot.setName(name);
            return snapshot;
        }
        Snapshot snapshot = new Snapshot(name, snapshots.size());
        snapshots.add(snapshot);

        if (snapshots.size() > 1) {
            for (Iterator e6 = functionProfiles.iterator(); e6.hasNext();) {
                FunctionProfile fp = (FunctionProfile) e6.next();
                if (fp != null) { // fp == null would mean this thread didn't call this function
                    fp.addSnapshot();
                }
            }
            for (Iterator it = userEventProfiles.iterator(); it.hasNext();) {
                UserEventProfile uep = (UserEventProfile) it.next();
                if (uep != null) {
                    uep.addSnapshot();
                }
            }
        }

        recreateData();

        return snapshot;
    }

    private void recreateData() {
        int numSnapshots = getNumSnapshots();
        int numMetrics = getNumMetrics();
        threadData = new ThreadData[getNumSnapshots()][getNumMetrics()];
        for (int s = 0; s < getNumSnapshots(); s++) {
            for (int m = 0; m < getNumMetrics(); m++) {
                threadData[s][m] = new ThreadData();
            }
        }
    }

    public List getSnapshots() {
        return snapshots;
    }

    public int getNumSnapshots() {
        return Math.max(1, snapshots.size());
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
    
    public FunctionProfile getOrCreateFunctionProfile(Function function, int numMetrics) {
        FunctionProfile fp = getFunctionProfile(function);
        if (fp == null) {
            fp = new FunctionProfile(function, numMetrics);
            addFunctionProfile(fp);
            return fp;
        } else {
            return fp;
        }
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

    // Since per thread callpath relations are built on demand, the following four functions tell whether this
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

    public void setThreadData(int metric) {
        setThreadValues(metric, metric, 0, getNumSnapshots() - 1);
    }

    public void setThreadDataAllMetrics() {
        setThreadValues(0, this.getNumMetrics() - 1, 0, getNumSnapshots() - 1);
    }

    // compute max values and percentages for threads (not mean/total)
    private void setThreadValues(int startMetric, int endMetric, int startSnapshot, int endSnapshot) {

        String startString = (String) getMetaData().get("Starting Timestamp");
        if (startString != null) {
            setStartTime(Long.parseLong(startString));
        }

        for (int snapshot = startSnapshot; snapshot <= endSnapshot; snapshot++) {
            for (int metric = startMetric; metric <= endMetric; metric++) {
                ThreadData data = threadData[snapshot][metric];
                double maxInclusive = 0;
                double maxExclusive = 0;
                double maxInclusivePerCall = 0;
                double maxExclusivePerCall = 0;
                double maxNumCalls = 0;
                double maxNumSubr = 0;

                for (Iterator it = this.getFunctionProfileIterator(); it.hasNext();) {
                    FunctionProfile fp = (FunctionProfile) it.next();
                    if (fp == null) {
                        continue;
                    }
                    if (fp.getFunction().isPhase()) {
                        maxExclusive = Math.max(maxExclusive, fp.getInclusive(snapshot, metric));
                        maxExclusivePerCall = Math.max(maxExclusivePerCall, fp.getInclusivePerCall(snapshot, metric));
                    } else {
                        maxExclusive = Math.max(maxExclusive, fp.getExclusive(snapshot, metric));
                        maxExclusivePerCall = Math.max(maxExclusivePerCall, fp.getExclusivePerCall(snapshot, metric));
                    }
                    maxInclusive = Math.max(maxInclusive, fp.getInclusive(snapshot, metric));
                    maxInclusivePerCall = Math.max(maxInclusivePerCall, fp.getInclusivePerCall(snapshot, metric));
                    maxNumCalls = Math.max(maxNumCalls, fp.getNumCalls(snapshot));
                    maxNumSubr = Math.max(maxNumSubr, fp.getNumSubr(snapshot));
                }

                data.maxExclusive = maxExclusive;
                data.maxInclusive = maxInclusive;
                data.maxExclusivePerCall = maxExclusivePerCall;
                data.maxInclusivePerCall = maxInclusivePerCall;
                data.maxNumCalls = maxNumCalls;
                data.maxNumSubr = maxNumSubr;

                // Note: Assumption is made that the max inclusive value is the value required to calculate
                // percentage (ie, divide by). Thus, we are assuming that the sum of the exclusive
                // values is equal to the max inclusive value. This is a reasonable assumption. This also gets
                // us out of sticky situations when call path data is present (this skews attempts to calculate
                // the total exclusive value unless checks are made to ensure that we do not 
                // include call path objects).
                if (this.getNodeID() > -1) { // don't do this for mean/total
                    data.percentDivider = data.maxInclusive / 100.0;
                }

                double maxInclusivePercent = 0;
                double maxExclusivePercent = 0;
                for (Iterator it = this.getFunctionProfileIterator(); it.hasNext();) {
                    FunctionProfile fp = (FunctionProfile) it.next();
                    if (fp == null) {
                        continue;
                    }
                    maxExclusivePercent = Math.max(maxExclusivePercent, fp.getExclusivePercent(snapshot, metric));
                    maxInclusivePercent = Math.max(maxInclusivePercent, fp.getInclusivePercent(snapshot, metric));
                }

                data.maxExclusivePercent = maxExclusivePercent;
                data.maxInclusivePercent = maxInclusivePercent;
            }
        }
    }

    public Map getMetaData() {
        return metaData;
    }

    public double getMaxInclusive(int metric, int snapshot) {
        if (snapshot == -1) {
            snapshot = getNumSnapshots() - 1;
        }
        return threadData[snapshot][metric].maxInclusive;
    }

    public double getMaxExclusive(int metric, int snapshot) {
        return threadData[snapshot][metric].maxExclusive;
    }

    public double getMaxInclusivePercent(int metric, int snapshot) {
        return threadData[snapshot][metric].maxInclusivePercent;
    }

    public double getMaxExclusivePercent(int metric, int snapshot) {
        return threadData[snapshot][metric].maxExclusivePercent;
    }

    public double getMaxInclusivePerCall(int metric, int snapshot) {
        return threadData[snapshot][metric].maxInclusivePerCall;
    }

    public double getMaxExclusivePerCall(int metric, int snapshot) {
        return threadData[snapshot][metric].maxExclusivePerCall;
    }

    public void setPercentDivider(int metric, int snapshot, double divider) {
        threadData[snapshot][metric].percentDivider = divider;
    }

    public double getPercentDivider(int metric, int snapshot) {
        return threadData[snapshot][metric].percentDivider;
    }

    public double getMaxNumCalls(int snapshot) {
        return threadData[snapshot][0].maxNumCalls;
    }

    public double getMaxNumSubr(int snapshot) {
        return threadData[snapshot][0].maxNumSubr;
    }

    public long getStartTime() {
        return startTime;
    }

    public void setStartTime(long startTime) {
        this.startTime = startTime;
    }

    public DataSource getDataSource() {
        return dataSource;
    }

}