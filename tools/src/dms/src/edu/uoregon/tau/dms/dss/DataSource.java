package edu.uoregon.tau.dms.dss;

import java.util.Enumeration;
import java.util.Iterator;
import java.util.Vector;

import java.io.*;
import java.sql.*
;
public abstract class DataSource {

    public DataSource() {
        this.setTrialData(new TrialData());
        this.setNCT(new NCT());
    }

    abstract public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException;

    abstract public int getProgress();

    abstract public void cancelLoad();

    public boolean debug() {
        return UtilFncs.debug;
    }

    private boolean profileStatsPresent = false;
    private boolean profileCallsPresent = false;
    private boolean userEventsPresent = false;
    private boolean callPathDataPresent = false;
    private boolean aggregatesPresent = false;
    private boolean groupNamesPresent = false;

    // data structures
    private TrialData trialData = null;
    private NCT nct = null;
    protected Thread meanData = null;

    public Thread getMeanData() {
        return meanData;
    }

    protected Vector metrics = null;
    private int[] maxNCT = null;

    // temp

    protected boolean firstMetric;

    protected void setFirstMetric(boolean firstMetric) {
        this.firstMetric = firstMetric;
    }

    protected boolean firstMetric() {
        return firstMetric;
    }

    /**
     * Sets this data session's NCT object.
     * 
     * @param nct
     *            NCT object.
     */
    public void setNCT(NCT nct) {
        this.nct = nct;
    }

    /**
     * Gets this data session's NCT object.
     * 
     * @return An NCT object.
     */
    public NCT getNCT() {
        return nct;
    }

    /**
     * Sets this data session's TrialData object.
     * 
     * @param trialData
     *            TrialData object.
     */
    public void setTrialData(TrialData trialData) {
        this.trialData = trialData;
    }

    /**
     * Gets this data session's TrialData object.
     * 
     * @return A TrialData object.
     */
    public TrialData getTrialData() {
        return trialData;
    }

    protected void setUserEventsPresent(boolean userEventsPresent) {
        this.userEventsPresent = userEventsPresent;
    }

    protected void setCallPathDataPresent(boolean callPathDataPresent) {
        this.callPathDataPresent = callPathDataPresent;
    }

    protected void setProfileStatsPresent(boolean profileStatsPresent) {
        this.profileStatsPresent = profileStatsPresent;
    }

    protected void setProfileCallsPresent(boolean profileCallsPresent) {
        this.profileCallsPresent = profileCallsPresent;
    }

    protected void setAggregatesPresent(boolean aggregatesPresent) {
        this.aggregatesPresent = aggregatesPresent;
    }

    protected void setGroupNamesPresent(boolean groupNamesPresent) {
        this.groupNamesPresent = groupNamesPresent;
    }

    public boolean profileStatsPresent() {
        return profileStatsPresent;
    }

    public boolean profileCallsPresent() {
        return profileCallsPresent();
    }

    public boolean aggregatesPresent() {
        return aggregatesPresent;
    }

    public boolean groupNamesPresent() {
        return groupNamesPresent;
    }

    public boolean userEventsPresent() {
        return userEventsPresent;
    }

    public boolean callPathDataPresent() {
        return callPathDataPresent;
    }

    protected boolean groupCheck = false;

    protected void setGroupCheck(boolean groupCheck) {
        this.groupCheck = groupCheck;
    }

    protected boolean groupCheck() {
        return groupCheck;
    }

    //Gets the maximum id reached for all nodes, context, and threads.
    //This takes into account that id values might not be contiguous (ie, we do
    // not
    //simply get the maximum number seen. For example, there might be only one
    // profile
    //in the system for n,c,t of 0,1,234. We do not want to just return [1,1,1]
    // representing
    //the number of items, but the actual id values which are the largest (ie,
    // return [0,1,234]).
    public int[] getMaxNCTNumbers() {
        if (maxNCT == null) {
            maxNCT = new int[3];
            for (int i = 0; i < 3; i++) {
                maxNCT[i] = 0;
            }
            for (Enumeration e1 = (this.getNCT().getNodes()).elements(); e1.hasMoreElements();) {
                Node node = (Node) e1.nextElement();
                if (node.getNodeID() > maxNCT[0])
                    maxNCT[0] = node.getNodeID();
                for (Enumeration e2 = (node.getContexts()).elements(); e2.hasMoreElements();) {
                    Context context = (Context) e2.nextElement();
                    if (context.getContextID() > maxNCT[1])
                        maxNCT[1] = context.getContextID();
                    for (Enumeration e3 = (context.getThreads()).elements(); e3.hasMoreElements();) {
                        edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
                        if (thread.getThreadID() > maxNCT[2])
                            maxNCT[2] = thread.getThreadID();
                    }
                }
            }

        }
        return maxNCT;
    }

    // Metrics Stuff

    /**
     * Set a Vector of metrics for this DataSession. The DataSession object will
     * maintain a reference to the Vector of metric values. To clear this
     * reference, call setMetric(Metric) with null.
     * 
     * @param metrics
     *            Vector of metric values to be saved.
     */
    public void setMetrics(Vector metrics) {
        this.metrics = metrics;
    }

    /**
     * Adds a metric to this data sessions metrics. The DataSession object will
     * maintain a reference to the Vector of metric values. To clear this
     * reference, call setMetric(String) with null.
     * 
     * @param metric
     *            name of metric.
     * 
     * @return Metric the newly added metric.
     */
    public int addMetric(Metric metric) {
        if (this.metrics == null) {
            this.metrics = new Vector();
        }

        //Try getting the metric.

        // int index = metrics.indexOf(metric);
        // didn't work. so do it manually.
        int index = -1;
        Enumeration e = metrics.elements();
        for (int i = 0; e.hasMoreElements(); i++) {
            Metric m = (Metric) e.nextElement();
            if (m.equals(metric)) {
                index = i;
                break;
            }
        }
        if (index == -1) {
            metric.setID(this.getNumberOfMetrics());
            metrics.add(metric);
        } else {
            metric.setID(((Metric) (metrics.elementAt(index))).getID());
        }
        return metric.getID();
    }

    public Metric addMetric(String metricName) {
        Metric metric = new Metric();
        metric.setName(metricName);
        addMetric(metric);
        return metric;
    }

    /**
     * Get a Vector of metric values for this DataSession. The DataSession
     * object will maintain a reference to the Vector of metric values. To clear
     * this reference, call setMetric(String) with null.
     * 
     * @return Vector of metric values
     */
    public Vector getMetrics() {
        return this.metrics;
    }

    /**
     * Get the metric with the given id.. The DataSession object will maintain a
     * reference to the Vector of metric values. To clear this reference, call
     * setMetric(String) with null.
     * 
     * @param metricID
     *            metric id.
     * 
     * @return Metric with given id.
     */
    public Metric getMetric(int metricID) {
        //Try getting the matric.
        if ((this.metrics != null) && (metricID < this.metrics.size()))
            return (Metric) this.metrics.elementAt(metricID);
        else
            return null;
    }

    /**
     * Get the metric name corresponding to the given id. The DataSession object
     * will maintain a reference to the Vector of metric values. To clear this
     * reference, call setMetric(String) with null.
     * 
     * @param metricID
     *            metric id.
     * 
     * @return The metric name as a String.
     */
    public String getMetricName(int metricID) {
        //Try getting the metric name.
        if ((this.metrics != null) && (metricID < this.metrics.size()))
            return ((Metric) this.metrics.elementAt(metricID)).getName();
        else
            return null;
    }

    /**
     * Get the number of metrics. The DataSession object will maintain a
     * reference to the Vector of metric values. To clear this reference, call
     * setMetric(String) with null.
     * 
     * @return Returns the number of metrics as an int.
     */
    public int getNumberOfMetrics() {
        //Try getting the matric name.
        if (this.metrics != null)
            return metrics.size();
        else
            return -1;
    }

    // Private Methods

    /*
     * After loading all data, this function should be called to generate all
     * the derived data
     */
    public void generateDerivedData() {
        for (Enumeration e = this.getNCT().getThreads().elements(); e.hasMoreElements();) {
            ((Thread) e.nextElement()).setThreadDataAllMetrics();
        }
        this.setMeanData(0, this.getNumberOfMetrics() - 1);
        this.meanData.setThreadDataAllMetrics();
    }

    public void setMeanData(int startMetric, int endMetric) {

        /*
         * Given, excl, incl, call, subr for each thread 
         * 
         * for each thread: 
         *   for each function:
         *     inclpercent = incl / (max(all incls for this thread)) * 100 
         *     exclpercent = excl / (max(all incls for this thread)) * 100 
         *     inclpercall = incl / call 
         * 
         * for the total: 
         *   for each function: 
         *     incl = sum(all threads, incl) 
         *     excl = sum(all threads, excl) 
         *     call = sum(all threads, call) 
         *     subr = sum(all threads, subr) 
         *     inclpercent = incl / (max(all incls for total)) * 100 
         *     exclpercent = excl / (max(all incls for total)) * 100 
         *     inclpercall = incl / call

         * 
         *     inclpercent = incl / sum(max incl for each thread) * 100 
         *     exclpercent = excl / sum(max incl for each thread) * 100 

         * 
         * for the mean: 
         *   for each function: 
         *     incl = total(incl) / numThreads 
         *     excl = total(excl) / numThreads
         *     call = total(call) / numThreads 
         *     subr = total(subr) / numThreads
         *     inclpercent = total.inclpercent 
         *     exclpercent = total.exclpercent
         *     inclpercall = total.inclpercall
         */

        int numMetrics = this.getNumberOfMetrics();

        double[] exclSum = new double[numMetrics];
        double[] inclSum = new double[numMetrics];
        double[] maxInclSum = new double[numMetrics];
        double callSum = 0;
        double subrSum = 0;

        double topLevelInclSum[] = new double[numMetrics];

        for (int i = 0; i < numMetrics; i++) {
            maxInclSum[i] = 0;
        }

        TrialData trialData = this.getTrialData();
        Iterator l = trialData.getFunctions();

        if (meanData == null) {
            meanData = new Thread(-1, -1, -1, numMetrics);
            meanData.initializeFunctionList(this.getTrialData().getNumFunctions());
        }

        for (int i = 0; i < numMetrics; i++) {
            for (Enumeration e1 = this.getNCT().getNodes().elements(); e1.hasMoreElements();) {
                Node node = (Node) e1.nextElement();
                for (Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements();) {
                    Context context = (Context) e2.nextElement();
                    for (Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements();) {
                        edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
                        topLevelInclSum[i] += thread.getMaxInclusive(i);
                    }
                }
            }
        }

        while (l.hasNext()) { // for each function
            Function function = (Function) l.next();

            // get/create the FunctionProfile for mean
            FunctionProfile meanProfile = meanData.getFunctionProfile(function);
            if (meanProfile == null) {
                meanProfile = new FunctionProfile(function, numMetrics);
                meanData.addFunctionProfile(meanProfile, function.getID());
            }

            function.setMeanProfile(meanProfile);

            callSum = 0;
            subrSum = 0;
            for (int i = 0; i < numMetrics; i++) {
                exclSum[i] = 0;
                inclSum[i] = 0;
            }

            // this must be stored somewhere else, but I'm going to compute it
            // since I don't know where
            int numThreads = 0;

            for (Enumeration e1 = this.getNCT().getNodes().elements(); e1.hasMoreElements();) {
                Node node = (Node) e1.nextElement();
                for (Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements();) {
                    Context context = (Context) e2.nextElement();
                    for (Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements();) {
                        edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
                        FunctionProfile functionProfile = thread.getFunctionProfile(function);

                        if (functionProfile != null) { // only if this function was called for this nct

                            for (int i = startMetric; i <= endMetric; i++) {

                                exclSum[i] += functionProfile.getExclusive(i);
                                inclSum[i] += functionProfile.getInclusive(i);

                                // the same for every metric
                                if (i == 0) {
                                    callSum += functionProfile.getNumCalls();
                                    subrSum += functionProfile.getNumSubr();
                                }
                            }
                        }
                        numThreads++;
                    }
                }
            }

            // we don't want to set the calls and subroutines if we're just computing mean data for a derived metric!
            if (startMetric == 0) {
                // set the totals for all but percentages - need to do those later...
                function.setTotalNumCalls(callSum);
                function.setTotalNumSubr(subrSum);

                // mean is just the total / numThreads
                meanProfile.setNumCalls((double) callSum / numThreads);
                meanProfile.setNumSubr((double) subrSum / numThreads);
            }

            for (int i = startMetric; i <= endMetric; i++) {
                function.setTotalExclusive(i, exclSum[i]);
                function.setTotalInclusive(i, inclSum[i]);
                function.setTotalInclusivePerCall(i, inclSum[i] / function.getTotalNumCalls());

                // mean data computed as above in comments

                meanProfile.setExclusive(i, exclSum[i] / numThreads);
                meanProfile.setInclusive(i, inclSum[i] / numThreads);
                meanProfile.setInclusivePerCall(i, inclSum[i] / numThreads / meanProfile.getNumCalls());

                if (inclSum[i] > maxInclSum[i]) {
                    maxInclSum[i] = inclSum[i];
                }
            }

        }

        // now compute percentages since we now have max(all incls for total)
        // for each function
        l = trialData.getFunctions();
        while (l.hasNext()) {

            Function function = (Function) l.next();

            for (int i = startMetric; i <= endMetric; i++) {
                if (maxInclSum[i] != 0) {
                    //                    function.setTotalInclusivePercent(i, function.getTotalInclusive(i)
                    //                            / maxInclSum[i] * 100);
                    //                    function.setTotalExclusivePercent(i, function.getTotalExclusive(i)
                    //                            / maxInclSum[i] * 100);
                    function.setTotalInclusivePercent(i, function.getTotalInclusive(i)
                            / topLevelInclSum[i] * 100);
                    function.setTotalExclusivePercent(i, function.getTotalExclusive(i)
                            / topLevelInclSum[i] * 100);
                }

            }
            function.setMeanValuesSet(true);
        }
    }

}