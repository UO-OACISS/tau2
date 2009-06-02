package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.*;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

/**
 * This class represents a data source.  After loading, data is availiable through the
 * public methods.
 *  
 * <P>CVS $Id: DataSource.java,v 1.45 2009/06/02 19:59:26 amorris Exp $</P>
 * @author  Robert Bell, Alan Morris
 * @version $Revision: 1.45 $
 */
public abstract class DataSource {

    public static final SimpleDateFormat dateTime = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");

    public static final int PPK = 0;
    public static final int TAUPROFILE = 1;
    public static final int DYNAPROF = 2;
    public static final int MPIP = 3;
    public static final int HPM = 4;
    public static final int GPROF = 5;
    public static final int PSRUN = 6;
    public static final int PPROF = 7;
    public static final int CUBE = 8;
    public static final int HPCTOOLKIT = 9;
    public static final int SNAP = 10;
    public static final int OMPP = 11;
    public static final int PERIXML = 12;
    public static final int GPTL = 13; // General Purpose Timing Library - Jim Rosinski
    public static final int PARAVER = 14; // Statistical output from Paraver - Jesus LeBarta
    public static final int IPM = 15; // Data from IPM/NERSC
    public static final int GYRO = 100;
    public static final int GAMESS = 101; // application log data

    public static String formatTypeStrings[] = { "ParaProf Packed Profile", "Tau profiles", "Dynaprof", "MpiP", "HPMToolkit",
            "Gprof", "PSRun", "Tau pprof.dat", "Cube", "HPCToolkit", "TAU Snapshot", "ompP", "PERI-XML",
            "General Purpose Timing Library (GPTL)", "Paraver", "IPM" };

    private static boolean meanIncludeNulls = true;

    private boolean userEventsPresent = false;
    private boolean callPathDataPresent = false;
    private boolean groupNamesPresent = false;
    private boolean phasesPresent = false;
    private Function topLevelPhase;

    // data structures
    private List metrics = null;
    protected Thread meanData = null;
    protected Thread totalData = null;
    protected Thread stddevData = null;
    private Map nodes = new TreeMap();
    private Map functions = new TreeMap();
    private Map groups = new TreeMap();
    private Map userEvents = new TreeMap();
    private List allThreads;

    private boolean generateIntermediateCallPathData;
    private boolean reverseDataAvailable;

    // just a holder for the output of getMaxNCTNumbers(), makes subsequent calls instantaneous
    private int[] maxNCT = null;

    abstract public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException;

    abstract public int getProgress();

    abstract public void cancelLoad();

    protected volatile boolean reloading;

    protected Map metaData = new TreeMap();
    protected Map uncommonMetaData = new TreeMap();

    private File metadataFile;
    private StringBuffer metadataString = new StringBuffer();

    private boolean wellBehavedSnapshots;
    private long avgStartTime;

    protected boolean monitored;

    protected boolean hasThreads = true;
    protected boolean hasContexts = true;
    protected boolean hasMPI = false;

    public DataSource() {
    // nothing
    }

    // by default no files
    public List getFiles() {
        return new ArrayList();
    }

    public boolean isReloading() {
        return reloading;
    }

    public boolean reloadData() throws Exception {
        if (this.reloading) {
            return false;
        }
        this.reloading = true;
        //System.out.println("=> Begin Reloading");
        cleanData();
        load();
        //System.out.println("=> End Reloading");
        this.reloading = false;
        return true;
    }

    protected void cleanData() {
        for (Iterator it = this.getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();
            for (Iterator it2 = thread.getFunctionProfileIterator(); it2.hasNext();) {
                FunctionProfile fp = (FunctionProfile) it2.next();
                if (fp != null) {
                    for (int m = 0; m < this.getNumberOfMetrics(); m++) {
                        fp.setExclusive(m, 0);
                        fp.setInclusive(m, 0);
                    }
                    fp.setNumSubr(0);
                    fp.setNumCalls(0);
                }
            }
        }
    }

    public void setMonitored(boolean monitored) {
        this.monitored = monitored;
    }

    public Thread getMeanData() {
        return meanData;
    }

    public Thread getStdDevData() {
        return stddevData;
    }

    public Thread getTotalData() {
        return totalData;
    }

    private void setCallPathDataPresent(boolean callPathDataPresent) {
        this.callPathDataPresent = callPathDataPresent;
    }

    public boolean getCallPathDataPresent() {
        return callPathDataPresent;
    }

    protected void setGroupNamesPresent(boolean groupNamesPresent) {
        this.groupNamesPresent = groupNamesPresent;
    }

    public boolean getGroupNamesPresent() {
        return groupNamesPresent;
    }

    protected void setUserEventsPresent(boolean userEventsPresent) {
        this.userEventsPresent = userEventsPresent;
    }

    public boolean getUserEventsPresent() {
        return userEventsPresent;
    }

    public Function addFunction(String name) {
        return this.addFunction(name, this.getNumberOfMetrics());
    }

    public Function addFunction(String name, int numMetrics) {
        name = name.trim();
        Function function = (Function) functions.get(name);

        // return the function if found
        if (function != null) {
            return function;
        }

        // otherwise, add it and return it
        function = new Function(name, functions.size(), numMetrics);
        functions.put(name, function);
        return function;
    }

    public Function getFunction(String name) {
        return (Function) functions.get(name.trim());
    }

    public int getNumFunctions() {
        return functions.size();
    }

    public Iterator getFunctions() {
        return functions.values().iterator();
    }

    public UserEvent addUserEvent(String name) {
        Object obj = userEvents.get(name);

        if (obj != null) {
            return (UserEvent) obj;
        }

        UserEvent userEvent = new UserEvent(name, userEvents.size() + 1);
        userEvents.put(name, userEvent);
        setUserEventsPresent(true);
        return userEvent;
    }

    public UserEvent getUserEvent(String name) {
        return (UserEvent) userEvents.get(name);
    }

    public int getNumUserEvents() {
        return userEvents.size();
    }

    public Iterator getUserEvents() {
        return userEvents.values().iterator();
    }

    public Group getGroup(String name) {
        return (Group) groups.get(name);
    }

    public Group addGroup(String name) {
        name = name.trim();
        if (name == "TAU_MPI") {
            hasMPI = true;
        }
        Object obj = groups.get(name);

        if (obj != null) {
            return (Group) obj;
        }

        Group group = new Group(name, groups.size() + 1);
        groups.put(name, group);
        return group;
    }

    protected void addGroups(String groupNames, Function function) {
        if (groupNames == null) {
            return;
        }
        StringTokenizer st = new StringTokenizer(groupNames, "|");
        while (st.hasMoreTokens()) {
            String groupName = st.nextToken();
            if (groupName != null) {
                Group group = this.addGroup(groupName.trim());
                function.addGroup(group);
            }
        }
        this.setGroupNamesPresent(true);
    }

    public int getNumGroups() {
        return groups.size();
    }

    public Iterator getGroups() {
        return groups.values().iterator();
    }

    /**
     * Retrieves the highest value found for each of node, context thread.  For example, 
     * if the two threads in the system are (1,0,512) and (512,0,1), it will return [512,0,512].
     * 
     * @return int[3] 3 numbers indicating the largest values for node, context, and thread respectively
     */
    public int[] getMaxNCTNumbers() {
        if (maxNCT == null) {
            maxNCT = new int[3];

            for (Iterator it = this.getNodes(); it.hasNext();) {
                Node node = (Node) it.next();
                maxNCT[0] = Math.max(maxNCT[0], node.getNodeID());
                for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                    Context context = (Context) it2.next();
                    maxNCT[1] = Math.max(maxNCT[1], context.getContextID());
                    for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                        Thread thread = (Thread) it3.next();
                        maxNCT[2] = Math.max(maxNCT[2], thread.getThreadID());
                    }
                }
            }
        }
        return maxNCT;
    }

    public int getNumThreads() {
        int numThreads = 0;
        for (Iterator it = this.getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                    it3.next();
                    numThreads++;
                }
            }
        }
        return numThreads;
    }

    /**
     * Set the List of Metrics for this DataSource.
     * 
     * @param metrics
     *            List of Metrics
     */
    public void setMetrics(List metrics) {
        this.metrics = metrics;
    }

    /**
     * Adds a metric to the DataSource's metrics list.
     * 
     * @param metric
     *            Metric to be added
     */
    public void addMetric(Metric metric) {
        if (this.metrics == null) {
            this.metrics = new ArrayList();
        } else {
            for (Iterator it = getAllThreads().iterator(); it.hasNext();) {
                Thread thread = (Thread) it.next();
                thread.addMetric();
            }

            if (meanData != null) {
                meanData.addMetric();
                totalData.addMetric();
                stddevData.addMetric();
            }
        }

        metric.setID(this.getNumberOfMetrics());
        metrics.add(metric);
    }

    /**
     * Adds a metric to the DataSource's metrics List (given as a String).
     * 
     * @param metric
     *            Name of metric to be added
     */
    public Metric addMetric(String metricName) {
        if (metrics != null) {
            for (Iterator it = metrics.iterator(); it.hasNext();) {
                Metric metric = (Metric) it.next();
                if (metric.getName().equals(metricName)) {
                    return metric;
                }
            }
        }

        Metric metric = new Metric();
        metric.setName(metricName);
        addMetric(metric);
        return metric;
    }

    /**
     * Adds a metric to the DataSource's metrics List (given as a String).
     * need to create this method from DataSource.  The DataSource
     * assumes that we are processing in this order: metric,thread,event.
     * The GPTL data is in thread,event,metric order.
     * 
     * @param metric
     *            Name of metric to be added
     * @param thread
     *            Current thread that is being processed
     */
    public Metric addMetric(String metricName, Thread thread) {
        Metric metric = null;
        // if the metric list exists, and has values, search for this metric
        if (metrics == null) {
            metrics = new ArrayList();
        } else {
            for (Iterator it = metrics.iterator(); it.hasNext();) {
                Metric tmp = (Metric) it.next();
                if (tmp.getName().equals(metricName)) {
                    metric = tmp;
                    break;
                }
            }
        }

        // did we find the metric?  If not, create a new one
        if (metric == null) {
            metric = new Metric();
            metric.setName(metricName);
            metric.setID(this.getNumberOfMetrics());
            metrics.add(metric);
        }

        // if the current thread doesn't have this metric, add it
        if (thread.getNumMetrics() < this.getNumberOfMetrics()) {
            thread.addMetric();
        }

        return metric;
    }

    /**
     * Get a the List of Metrics
     * 
     * @return List of Metrics
     */
    public List getMetrics() {
        return metrics;
    }

    /**
     * Clear the list of metrics
     * 
     */
    public void clearMetrics() {
        metrics = null;
    }

    public Metric getMetric(String name) {
        for (Iterator it = metrics.iterator(); it.hasNext();) {
            Metric metric = (Metric) it.next();
            if (metric.getName().equals(name)) {
                return metric;
            }
        }
        return null;
    }

    /**
     * Get the metric with the given id. 
     * 
     * @param metricID
     *            metric id.
     * 
     * @return Metric with given id.
     */
    public Metric getMetric(int metricID) {
        if ((this.metrics != null) && (metricID < this.metrics.size())) {
            return (Metric) this.metrics.get(metricID);
        } else {
            return null;
        }
    }

    /**
     * Get the metric name corresponding to the given id. The DataSession object
     * will maintain a reference to the List of metric values. To clear this
     * reference, call setMetric(String) with null.
     * 
     * @param metricID
     *            metric id.
     * 
     * @return The metric name as a String.
     */
    public String getMetricName(int metricID) {
        if ((this.metrics != null) && (metricID < this.metrics.size())) {
            return ((Metric) this.metrics.get(metricID)).getName();
        } else {
            return null;
        }
    }

    /**
     * Get the number of metrics. The DataSession object will maintain a
     * reference to the List of metric values. To clear this reference, call
     * setMetric(String) with null.
     * 
     * @return Returns the number of metrics as an int.
     */
    public int getNumberOfMetrics() {
        //Try getting the metric name.
        if (this.metrics != null) {
            return metrics.size();
        } else {
            return 0;
        }
    }

    private void generateBonusCallPathData() {
        /*
         * The "bonus" data is necessary for the reverse call tree
         * With normal callpath data we have:
         *     A => B1 => C => D
         *     A => B2 => C => D
         *     D
         *     
         * Here, D is duplicate data, it is the sum of the other two
         * 
         * The Bonus data is additional duplicate data:
         *     B1 => C => D
         *     B2 => C => D
         *     C => D
         *     
         * It provides an aggregate view from each root for all parents
         */
        if (!getCallPathDataPresent()) {
            return;
        }

        List functions = new ArrayList();

        for (Iterator l = this.getFunctions(); l.hasNext();) {
            Function function = (Function) l.next();
            functions.add(function);
        }

        // make sure that the allThreads list is initialized;
        this.initAllThreadsList();

        int numThreads = allThreads.size();

        Group derivedGroup = this.addGroup("TAU_CALLPATH_DERIVED");
        reverseDataAvailable = true;

        for (Iterator l = functions.iterator(); l.hasNext();) {
            Function function = (Function) l.next();

            if (function.isCallPathFunction() && CallPathUtilFuncs.containsDoublePath(function.getName())) {

                String bonusName = function.getName();
                bonusName = bonusName.substring(bonusName.indexOf("=>") + 2);
                while (bonusName.indexOf("=>") != -1) {

                    Function bonusFunction = this.getFunction(bonusName);
                    if (bonusFunction == null) {
                        bonusFunction = addFunction(bonusName);
                        for (Iterator g = function.getGroups().iterator(); g.hasNext();) {
                            bonusFunction.addGroup((Group) g.next());
                        }
                    }
                    bonusFunction.addGroup(derivedGroup);
                    bonusName = bonusName.substring(bonusName.indexOf("=>") + 2);

                    for (int i = 0; i < numThreads; i++) {
                        Thread thread = (Thread) allThreads.get(i);

                        FunctionProfile functionProfile = thread.getFunctionProfile(function);
                        if (functionProfile != null) {
                            FunctionProfile bfp = thread.getFunctionProfile(bonusFunction);
                            if (bfp == null) {
                                bfp = new FunctionProfile(bonusFunction, getNumberOfMetrics(), thread.getNumSnapshots());
                                thread.addFunctionProfile(bfp);
                            }

                            for (int metric = 0; metric < this.getNumberOfMetrics(); metric++) {

                                bfp.setExclusive(metric, bfp.getExclusive(metric) + functionProfile.getExclusive(metric));
                                bfp.setInclusive(metric, bfp.getInclusive(metric) + functionProfile.getInclusive(metric));
                            }
                            bfp.setNumCalls(bfp.getNumCalls() + functionProfile.getNumCalls());
                            bfp.setNumSubr(bfp.getNumSubr() + functionProfile.getNumSubr());

                        }
                    }
                }
            }
        }

    }

    private void addDerivedSnapshots(Thread thread, Thread derivedThread) {
        if (wellBehavedSnapshots) {

            derivedThread.setStartTime(avgStartTime);

            for (Iterator it = thread.getSnapshots().iterator(); it.hasNext();) {
                Snapshot snapshot = (Snapshot) it.next();
                Snapshot derivedSnapshot = derivedThread.addSnapshot(snapshot.getName());
                derivedSnapshot.setTimestamp(snapshot.getTimestamp());
            }
        }
    }

    /*
     * After loading all data, this function should be called to generate all
     * the derived data
     */
    public void generateDerivedData() {
        //long time = System.currentTimeMillis();

        // reset the list of all threads so that it will be recreated
        allThreads = null;

        if (CallPathUtilFuncs.checkCallPathsPresent(getFunctions())) {
            setCallPathDataPresent(true);
        }

        if (generateIntermediateCallPathData) {
            generateBonusCallPathData();
        }

        checkForPhases();

        // initialize to the first thread
        int numDerivedSnapshots = ((Thread) this.getAllThreads().get(0)).getNumSnapshots();

        long sumStartTime = 0;
        for (Iterator it = this.getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();
            thread.setThreadDataAllMetrics();

            int numSnapshots = thread.getNumSnapshots();
            if (numSnapshots != numDerivedSnapshots) {
                numDerivedSnapshots = -1;
            }
            String startString = (String) thread.getMetaData().get("Starting Timestamp");
            if (startString != null) {
                sumStartTime += Long.parseLong(startString);
            }
        }

        if (numDerivedSnapshots > 1) {
            // only true when all threads have the same number of snapshots
            // otherwise we make no derived snapshots
            wellBehavedSnapshots = true;
            avgStartTime = (long) ((double) sumStartTime / this.getAllThreads().size());
        }

        try {
            this.generateStatistics(0, this.getNumberOfMetrics() - 1);
        } catch (Exception e) {
            e.printStackTrace();
        }
        this.meanData.setThreadDataAllMetrics();
        this.totalData.setThreadDataAllMetrics();
        this.stddevData.setThreadDataAllMetrics();

        this.generateUserEventStatistics();

        finishPhaseAnalysis();

        //time = (System.currentTimeMillis()) - time;
        //System.out.println("Time to process (in milliseconds): " + time);

    }

    private double computeStdDev(double sumSqr, double mean, int count) {
        double stdDev = 0;
        if (count > 1) {
            stdDev = java.lang.Math.sqrt(java.lang.Math.abs((sumSqr / count) - (mean * mean)));
        }

        return stdDev;
    }

    private void generateUserEventStatistics() {
        // make sure that the allThreads list is initialized;
        this.initAllThreadsList();
        int numThreads = allThreads.size();
        int numSnapshots = meanData.getNumSnapshots();

        int numProfiles[] = new int[getNumUserEvents() + 1];

        for (int snapshot = 0; snapshot < numSnapshots; snapshot++) {

            for (int i = 0; i < numThreads; i++) { // for each thread
                Thread thread = (Thread) allThreads.get(i);
                for (Iterator it = thread.getUserEventProfiles(); it.hasNext();) {
                    UserEventProfile uep = (UserEventProfile) it.next();
                    UserEvent ue = uep.getUserEvent();

                    // get/create the userEventProfile for mean
                    UserEventProfile meanProfile = meanData.getUserEventProfile(ue);
                    if (meanProfile == null) {
                        meanProfile = new UserEventProfile(ue, numSnapshots);
                        meanData.addUserEventProfile(meanProfile);
                    }

                    // get/create the userEventProfile for total
                    UserEventProfile totalProfile = totalData.getUserEventProfile(ue);
                    if (totalProfile == null) {
                        totalProfile = new UserEventProfile(ue, numSnapshots);
                        totalData.addUserEventProfile(totalProfile);
                    }

                    // get/create the userEventProfile for stddev
                    UserEventProfile stddevProfile = stddevData.getUserEventProfile(ue);
                    if (stddevProfile == null) {
                        stddevProfile = new UserEventProfile(ue, numSnapshots);
                        stddevData.addUserEventProfile(stddevProfile);
                    }

                    numProfiles[ue.getID()]++;

                    totalProfile.setNumSamples(totalProfile.getNumSamples() + uep.getNumSamples(snapshot), snapshot);
                    totalProfile.setMaxValue(totalProfile.getMaxValue() + uep.getMaxValue(snapshot), snapshot);
                    totalProfile.setMinValue(totalProfile.getMinValue() + uep.getMinValue(snapshot), snapshot);
                    totalProfile.setMeanValue(totalProfile.getMeanValue() + uep.getMeanValue(snapshot), snapshot);
                    totalProfile.setStdDev(totalProfile.getStdDev() + uep.getStdDev(snapshot), snapshot);

                    stddevProfile.setNumSamples(stddevProfile.getNumSamples()
                            + (uep.getNumSamples(snapshot) * uep.getNumSamples(snapshot)), snapshot);
                    stddevProfile.setMaxValue(stddevProfile.getMaxValue()
                            + (uep.getMaxValue(snapshot) * uep.getMaxValue(snapshot)), snapshot);
                    stddevProfile.setMinValue(stddevProfile.getMinValue()
                            + (uep.getMinValue(snapshot) * uep.getMinValue(snapshot)), snapshot);
                    stddevProfile.setMeanValue(stddevProfile.getMeanValue()
                            + (uep.getMeanValue(snapshot) * uep.getMeanValue(snapshot)), snapshot);
                    stddevProfile.setStdDev(stddevProfile.getStdDev() + (uep.getStdDev(snapshot) * uep.getStdDev(snapshot)),
                            snapshot);

                }
            }

            for (Iterator it = this.getUserEvents(); it.hasNext();) {
                UserEvent ue = (UserEvent) it.next();

                UserEventProfile meanProfile = meanData.getUserEventProfile(ue);
                UserEventProfile totalProfile = totalData.getUserEventProfile(ue);
                UserEventProfile stddevProfile = stddevData.getUserEventProfile(ue);

                int divider = numThreads;
                if (!meanIncludeNulls) { // do we include null values as zeroes in the computation or not?
                    divider = numProfiles[ue.getID()];
                }

                meanProfile.setNumSamples((totalProfile.getNumSamples() / divider), snapshot);
                meanProfile.setMaxValue(totalProfile.getMaxValue(snapshot) / divider, snapshot);
                meanProfile.setMinValue(totalProfile.getMinValue(snapshot) / divider, snapshot);
                meanProfile.setMeanValue(totalProfile.getMeanValue(snapshot) / divider, snapshot);
                meanProfile.setStdDev(totalProfile.getStdDev(snapshot) / divider, snapshot);

                stddevProfile.setNumSamples(computeStdDev(stddevProfile.getNumSamples(snapshot),
                        meanProfile.getNumSamples(snapshot), divider), snapshot);
                stddevProfile.setMaxValue(computeStdDev(stddevProfile.getMaxValue(snapshot), meanProfile.getMaxValue(snapshot),
                        divider), snapshot);
                stddevProfile.setMinValue(computeStdDev(stddevProfile.getMinValue(snapshot), meanProfile.getMinValue(snapshot),
                        divider), snapshot);
                stddevProfile.setMeanValue(computeStdDev(stddevProfile.getMeanValue(snapshot),
                        meanProfile.getMeanValue(snapshot), divider), snapshot);
                stddevProfile.setStdDev(
                        computeStdDev(stddevProfile.getStdDev(snapshot), meanProfile.getStdDev(snapshot), divider), snapshot);
            }
        }

    }

    public void generateStatistics(int startMetric, int endMetric) {

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
         *     inclpercent = incl / (sum(max(all incls for each thread)) * 100 
         *     exclpercent = excl / (sum(max(all incls for each thread)) * 100 
         *     inclpercall = incl / call
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
        Thread firstThread = (Thread) getAllThreads().get(0);
        if (meanData == null) {
            meanData = new Thread(-1, -1, -1, numMetrics, this);
            addDerivedSnapshots(firstThread, meanData);
        }

        if (totalData == null) {
            totalData = new Thread(-2, -2, -2, numMetrics, this);
            addDerivedSnapshots(firstThread, totalData);
        }

        if (stddevData == null) {
            stddevData = new Thread(-3, -3, -3, numMetrics, this);
            addDerivedSnapshots(firstThread, stddevData);
        }

        double[] exclSum = new double[numMetrics];
        double[] inclSum = new double[numMetrics];
        double[] exclSumSqr = new double[numMetrics];
        double[] inclSumSqr = new double[numMetrics];


        // make sure that the allThreads list is initialized;
        this.initAllThreadsList();

        for (int snapshot = 0; snapshot < meanData.getNumSnapshots(); snapshot++) {

            // must always iterate through all metrics regardless to find the top level timers, I think???
            for (int i = startMetric; i <= endMetric; i++) { // for each metric
                double topLevelInclSum[] = new double[numMetrics];
                for (Iterator it = allThreads.iterator(); it.hasNext();) { // for each thread
                    Thread thread = (Thread) it.next();
                    if (wellBehavedSnapshots) {
                        topLevelInclSum[i] += thread.getMaxInclusive(i, snapshot);
                    } else { // pick the last from each thread
                        topLevelInclSum[i] += thread.getMaxInclusive(i, thread.getNumSnapshots() - 1);
                    }
                }

                totalData.setPercentDivider(i, snapshot, topLevelInclSum[i] / 100.0);
                meanData.setPercentDivider(i, snapshot, topLevelInclSum[i] / 100.0);
                stddevData.setPercentDivider(i, snapshot, topLevelInclSum[i] / 100.0);
            }

            for (Iterator l = this.getFunctions(); l.hasNext();) { // for each function
                Function function = (Function) l.next();

                // get/create the FunctionProfile for mean
                FunctionProfile meanProfile = meanData.getFunctionProfile(function);
                if (meanProfile == null) {
                    meanProfile = new FunctionProfile(function, numMetrics, meanData.getNumSnapshots());
                    meanData.addFunctionProfile(meanProfile);
                }
                function.setMeanProfile(meanProfile);

                // get/create the FunctionProfile for total
                FunctionProfile totalProfile = totalData.getFunctionProfile(function);
                if (totalProfile == null) {
                    totalProfile = new FunctionProfile(function, numMetrics, meanData.getNumSnapshots());
                    totalData.addFunctionProfile(totalProfile);
                }
                function.setTotalProfile(totalProfile);

                // get/create the FunctionProfile for stddev
                FunctionProfile stddevProfile = stddevData.getFunctionProfile(function);
                if (stddevProfile == null) {
                    stddevProfile = new FunctionProfile(function, numMetrics, meanData.getNumSnapshots());
                    stddevData.addFunctionProfile(stddevProfile);
                }
                function.setStddevProfile(stddevProfile);

                int numEvents = 0;
                double callSum = 0;
                double subrSum = 0;
                double callSumSqr = 0;
                double subrSumSqr = 0;
                for (int i = 0; i < numMetrics; i++) {
                    exclSum[i] = 0;
                    inclSum[i] = 0;
                    exclSumSqr[i] = 0;
                    inclSumSqr[i] = 0;
                }

                int numThreads = allThreads.size();

                for (int i = 0; i < numThreads; i++) { // for each thread
                    Thread thread = (Thread) allThreads.get(i);
                    FunctionProfile functionProfile = thread.getFunctionProfile(function);

                    int s = snapshot;
                    if (!wellBehavedSnapshots) {
                        s = thread.getNumSnapshots() - 1;
                    }
                    if (functionProfile != null) { // only if this function was called for this nct
                        numEvents++;
                        for (int m = startMetric; m <= endMetric; m++) {

                            exclSum[m] += functionProfile.getExclusive(s, m);
                            inclSum[m] += functionProfile.getInclusive(s, m);
                            exclSumSqr[m] += functionProfile.getExclusive(s, m) * functionProfile.getExclusive(s, m);
                            inclSumSqr[m] += functionProfile.getInclusive(s, m) * functionProfile.getInclusive(s, m);

                            // the same for every metric
                            if (m == 0) {
                                callSum += functionProfile.getNumCalls(s);
                                subrSum += functionProfile.getNumSubr(s);
                                callSumSqr += functionProfile.getNumCalls(s) * functionProfile.getNumCalls(s);
                                subrSumSqr += functionProfile.getNumSubr(s) * functionProfile.getNumSubr(s);
                            }
                        }
                    }
                }

                int divider = numThreads;
                if (!meanIncludeNulls) { // do we include null values as zeroes in the computation or not?
                    divider = numEvents;
                }

                // we don't want to set the calls and subroutines if we're just computing mean data for a derived metric!
                if (startMetric == 0) {

                    totalProfile.setNumCalls(snapshot, callSum);
                    totalProfile.setNumSubr(snapshot, subrSum);

                    // mean is just the total / divider
                    meanProfile.setNumCalls(snapshot, (double) callSum / divider);
                    meanProfile.setNumSubr(snapshot, (double) subrSum / divider);

                    double stdDev = 0;
                    if (divider > 1) {
                        stdDev = java.lang.Math.sqrt(java.lang.Math.abs((callSumSqr / (divider))
                                - (meanProfile.getNumCalls(snapshot) * meanProfile.getNumCalls(snapshot))));
                    }
                    stddevProfile.setNumCalls(snapshot, stdDev);

                    stdDev = 0;
                    if (divider > 1) {
                        stdDev = java.lang.Math.sqrt(java.lang.Math.abs((subrSumSqr / (divider))
                                - (meanProfile.getNumSubr(snapshot) * meanProfile.getNumSubr(snapshot))));
                    }
                    stddevProfile.setNumSubr(snapshot, stdDev);

                }

                for (int m = startMetric; m <= endMetric; m++) {

                    totalProfile.setExclusive(snapshot, m, exclSum[m]);
                    totalProfile.setInclusive(snapshot, m, inclSum[m]);

                    // mean data computed as above in comments
                    meanProfile.setExclusive(snapshot, m, exclSum[m] / divider);
                    meanProfile.setInclusive(snapshot, m, inclSum[m] / divider);

                    double stdDev = 0;
                    if (divider > 1) {

                        // see http://cuwu.editthispage.com/stories/storyReader$13 for why I don't multiply by n/(n-1)

                        //stdDev = java.lang.Math.sqrt(((double) divider / (divider - 1))
                        //        * java.lang.Math.abs((exclSumSqr[i] / (divider))
                        //                - (meanProfile.getExclusive(i) * meanProfile.getExclusive(i))));
                        stdDev = java.lang.Math.sqrt(java.lang.Math.abs((exclSumSqr[m] / (divider))
                                - (meanProfile.getExclusive(snapshot, m) * meanProfile.getExclusive(snapshot, m))));
                    }
                    stddevProfile.setExclusive(snapshot, m, stdDev);

                    stdDev = 0;
                    if (divider > 1) {
                        stdDev = java.lang.Math.sqrt(java.lang.Math.abs((inclSumSqr[m] / (divider))
                                - (meanProfile.getInclusive(snapshot, m) * meanProfile.getInclusive(snapshot, m))));
                    }
                    stddevProfile.setInclusive(snapshot, m, stdDev);

                }
            }
        }

        // Now we count up whether we have threads and contexts or not, this is for display purposes in ParaProf
        int[] nct = getMaxNCTNumbers();
        hasContexts = nct[1] > 0;
        hasThreads = nct[2] > 0;

    }

    /**
     * Creates and then adds a node with the given id to the the list of nodes. 
     * The postion in which the node is added is determined by given id.
     * A node is not added if the id is < 0, or that node id is already
     * present. Adds do not have to be consecutive (ie., nodes can be added out of order).
     * The node created will have an id matching the given id.
     *
     * @param    nodeID The id of the node to be added.
     * @return    The Node that was added.
     */
    public Node addNode(int nodeID) {
        Object obj = nodes.get(new Integer(nodeID));

        // return the Node if found
        if (obj != null) {
            return (Node) obj;
        }

        // otherwise, add it and return it
        Node node = new Node(nodeID, this);
        nodes.put(new Integer(nodeID), node);
        return node;
    }

    public Thread addThread(int nodeID, int contextID, int threadID) {
        Node node = addNode(nodeID);
        Context context = node.addContext(contextID);
        Thread thread = context.addThread(threadID, this.getNumberOfMetrics());
        // reset the list of all threads so that it will be recreated
        allThreads = null;
        return thread;
    }

    /**
     * Gets the node with the specified node id.  If the node is not found, the function returns null.
     *
     * @param    nodeID The id of the node sought.
     * @return    The node found (or null if it was not).
     */
    public Node getNode(int nodeID) {
        return (Node) nodes.get(new Integer(nodeID));
    }

    /**
     * Returns the number of nodes in this NCT object.
     *
     * @return    The number of nodes.
     */
    public int getNumberOfNodes() {
        return nodes.size();
    }

    /**
     * Returns the list of nodes in this object as an Iterator.
     *
     * @return    An Iterator over node objects.
     */
    public Iterator getNodes() {
        return nodes.values().iterator();
    }

    //Returns the total number of contexts in this trial.
    public int getTotalNumberOfContexts() {
        int totalNumberOfContexts = -1;
        for (Iterator it = this.getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            totalNumberOfContexts += (node.getNumberOfContexts());
        }
        return totalNumberOfContexts;
    }

    //Returns the number of contexts on the specified node.
    public int getNumberOfContexts(int nodeID) {
        return ((Node) getNode(nodeID)).getNumberOfContexts();
    }

    //Returns all the contexts on the specified node.
    public Iterator getContexts(int nodeID) {
        Node node = getNode(nodeID);
        if (node != null) {
            return node.getContexts();
        }
        return null;
    }

    //Returns the context on the specified node.
    public Context getContext(int nodeID, int contextID) {
        Node node = getNode(nodeID);
        if (node != null) {
            return node.getContext(contextID);
        }
        return null;
    }

    //Returns the total number of threads in this trial.
    public int getTotalNumberOfThreads() {
        int totalNumberOfThreads = 0;
        for (Iterator it = this.getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                totalNumberOfThreads += (context.getNumberOfThreads());
            }
        }
        return totalNumberOfThreads;
    }

    //Returns the number of threads on the specified node,context.
    public int getNumberOfThreads(int nodeID, int contextID) {
        return (this.getContext(nodeID, contextID)).getNumberOfThreads();
    }

    public List getThreads() {
        List list = new ArrayList();
        for (Iterator it = this.getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                    Thread thread = (Thread) it3.next();
                    list.add(thread);
                }
            }
        }
        return list;
    }

    public Thread getThread(int nodeID, int contextID, int threadID) {
        if (nodeID == -1) {
            return this.getMeanData();
        } else if (nodeID == -3) {
            return this.getStdDevData();
        }

        Context context = this.getContext(nodeID, contextID);
        Thread thread = null;
        if (context != null)
            thread = context.getThread(threadID);
        return thread;
    }

    private void initAllThreadsList() {
        allThreads = new ArrayList();
        for (Iterator it = this.getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                    Thread thread = (Thread) it3.next();
                    allThreads.add(thread);
                }
            }
        }
    }

    public List getAllThreads() {
        if (allThreads == null) {
            initAllThreadsList();
        }
        return allThreads;
    }

    /**
     * Changes whether or not functions which do not call a particular function 
     * are included as a 0 in the computation of statistics (mean, std. dev., etc)
     * 
     * This does not affect how trials uploaded to the database are handled
     * 
     * @param meanIncludeNulls true to include nulls as 0's the computation, false otherwise
     */
    public static void setMeanIncludeNulls(boolean meanIncludeNulls) {
        DataSource.meanIncludeNulls = meanIncludeNulls;
    }

    protected void checkForPhases() {

        Group tau_phase = this.getGroup("TAU_PHASE");

        if (tau_phase != null) {
            phasesPresent = true;

            for (Iterator it = this.getFunctions(); it.hasNext();) {
                Function function = (Function) it.next();

                if (function.isGroupMember(tau_phase)) {
                    function.setPhase(true);
                    function.setActualPhase(function);
                }
            }

            for (Iterator it = this.getFunctions(); it.hasNext();) {
                Function function = (Function) it.next();

                int location = function.getName().indexOf("=>");

                if (location > 0) {
                    // split "A => B"
                    String phaseRoot = UtilFncs.getLeftSide(function.getName());
                    String phaseChild = UtilFncs.getRightSide(function.getName());

                    Function f = this.getFunction(phaseChild);
                    if (f.isPhase()) {
                        function.setPhase(true);
                        function.setActualPhase(f);
                    }

                    function.setParentPhase(this.getFunction(phaseRoot));
                }
            }

        }
    }

    protected void finishPhaseAnalysis() {

        if (phasesPresent) {
            Group tau_phase = this.getGroup("TAU_PHASE");
            ArrayList phases = new ArrayList();

            for (Iterator it = this.getFunctions(); it.hasNext();) {
                Function function = (Function) it.next();
                if (function.isGroupMember(tau_phase)) {
                    phases.add(function);
                }
            }

            // there must be at least one
            if (phases.size() == 0) {
                throw new RuntimeException("Error: TAU_PHASE found, but no phases!");
            }

            // try to find the "top level phase", usually 'main'
            topLevelPhase = (Function) phases.get(0);
            for (Iterator it = phases.iterator(); it.hasNext();) {
                Function function = (Function) it.next();
                if (function.getMeanInclusive(0) > topLevelPhase.getMeanInclusive(0)) {
                    topLevelPhase = function;
                }
            }
        }
    }

    public boolean getPhasesPresent() {
        return phasesPresent;
    }

    /**
     * Returns the top level phase, usually 'main'.
     * 
     * @return the top level phase
     */
    public Function getTopLevelPhase() {
        return topLevelPhase;
    }

    public boolean getGenerateIntermediateCallPathData() {
        return generateIntermediateCallPathData;
    }

    public void setGenerateIntermediateCallPathData(boolean generateIntermediateCallPathData) {
        this.generateIntermediateCallPathData = generateIntermediateCallPathData;
    }

    public boolean getReverseDataAvailable() {
        return reverseDataAvailable;
    }

    public Map getMetaData() {
        return metaData;
    }

    public void setMetaData(Map metaData) {
        this.metaData = metaData;
    }

    public void buildXMLMetaData() {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        try {
            // create the XML DOM document builder
            DocumentBuilder builder = factory.newDocumentBuilder();
            // create a new XML document
            Document document = builder.newDocument();

            // create the root node.  Do it this way, because the level 1
            // xerces DOM parser/write doesn't support namespaces. Grrr.
            Element root = (Element) document.createElement("tau:metadata");
            root.setAttribute("xmlns:tau", "http://www.cs.uoregon.edu/research/tau");
            document.appendChild(root);

            Element master = null;
            List nodeProfiles = new ArrayList();
            List profilesAdded = new ArrayList();

            // create the common attribute node
            if (metaData.size() > 0) {
                master = (Element) document.createElement("tau:CommonProfileAttributes");
                root.appendChild(master);

                // output the first thread of name / value pairs, like this:
                // <attribute><name>xxx</name><value>yyy</value></attribute>
                for (Iterator it2 = metaData.keySet().iterator(); it2.hasNext();) {
                    String name = (String) it2.next();
                    String value = (String) metaData.get(name);
                    Element attribute = (Element) document.createElement("tau:attribute");
                    master.appendChild(attribute);
                    Element attrName = (Element) document.createElement("tau:name");
                    attribute.appendChild(attrName);
                    attrName.appendChild(document.createTextNode(name));
                    Element attrValue = (Element) document.createElement("tau:value");
                    attribute.appendChild(attrValue);
                    attrValue.appendChild(document.createTextNode(value));
                }
            }

            // for all threads of execution, output the attributes
            // that are different for one or more threads
            for (Iterator it = getAllThreads().iterator(); it.hasNext();) {
                Thread thread = (Thread) it.next();
                // output the first thread of name / value pairs, like this:
                // <attribute><name>xxx</name><value>yyy</value></attribute>
                Element delta = (Element) document.createElement("tau:ProfileAttributes");
                // give the record attributes, so we know which thread of execution
                // it belongs to
                delta.setAttribute("node", Integer.toString(thread.getNodeID()));
                delta.setAttribute("context", Integer.toString(thread.getContextID()));
                delta.setAttribute("thread", Integer.toString(thread.getThreadID()));

                boolean addit = false;

                for (Iterator it2 = thread.getMetaData().keySet().iterator(); it2.hasNext();) {
                    String name = (String) it2.next();
                    String value = (String) thread.getMetaData().get(name);
                    // if this name/value pair is not in the master, then 
                    // append it to the tree.
                    if (!metaData.containsKey(name)) {
                        Element attribute = (Element) document.createElement("tau:attribute");
                        delta.appendChild(attribute);
                        Element attrName = (Element) document.createElement("tau:name");
                        attribute.appendChild(attrName);
                        attrName.appendChild(document.createTextNode(name));
                        Element attrValue = (Element) document.createElement("tau:value");
                        attribute.appendChild(attrValue);
                        attrValue.appendChild(document.createTextNode(value));
                        addit = true;
                    }
                }
                nodeProfiles.add(delta);

                if (addit) {
                    // don't add it to the tree, unless it has items that differ from
                    // the master record
                    root.appendChild(delta);
                    profilesAdded.add(new Boolean(true));
                } else {
                    profilesAdded.add(new Boolean(false));
                }

            }

            // if the user also specified an XML file on the command line
            // with XML data, then merge that tree into our tree
            if (this.metadataFile != null) {
                Document oldDocument = builder.parse(metadataFile);
                // get the root elements, so we can move it
                Element oldRoot = oldDocument.getDocumentElement();
                // here's the magic step!
                org.w3c.dom.Node imported = document.importNode(oldRoot, true);
                System.out.println(imported.getPrefix());
                System.out.println(imported.getNodeName());
                if (master != null && imported.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE
                        && imported.getNodeName().equals("tau:metadata")) {
                    // extract it out, and add it to our tree!
                    NodeList nodes = imported.getChildNodes();
                    int nodeIndex = 0;
                    for (int i = 0; i < nodes.getLength(); i++) {
                        org.w3c.dom.Node cpa = nodes.item(i);
                        // System.out.println("\t" + cpa.getNodeName());
                        if (cpa.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE
                                && cpa.getNodeName().equals("tau:CommonProfileAttributes")) {

                            NodeList attrs = cpa.getChildNodes();
                            // System.out.println("length: " + attrs.getLength());
                            for (int j = 0; j < attrs.getLength(); j++) {
                                // System.out.println("\t\tj:" + j);
                                org.w3c.dom.Node tmp = attrs.item(j);
                                if (tmp.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE
                                        && tmp.getNodeName().equals("tau:attribute")) {
                                    // System.out.println("\t\tname:" + tmp.getNodeName());
                                    master.appendChild(tmp);
                                }
                                // System.out.println("\t\tend j:" + j);
                            }

                            //root.appendChild(cpa);
                        } else if (cpa.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE
                                && cpa.getNodeName().equals("tau:ProfileAttributes")) {
                            Element currentDelta = (Element) nodeProfiles.get(nodeIndex);
                            if (((Boolean) profilesAdded.get(nodeIndex)).booleanValue()) {

                                NodeList attrs = cpa.getChildNodes();
                                // System.out.println("length: " + attrs.getLength());
                                for (int j = 0; j < attrs.getLength(); j++) {
                                    // System.out.println("\t\tj:" + j);
                                    org.w3c.dom.Node tmp = attrs.item(j);
                                    if (tmp.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE
                                            && tmp.getNodeName().equals("tau:attribute")) {
                                        // System.out.println("\t\tname:" + tmp.getNodeName());
                                        currentDelta.appendChild(tmp);
                                    }
                                    // System.out.println("\t\tend j:" + j);
                                }
                            } else {
                                root.appendChild(cpa);
                            }
                            nodeIndex++;
                        }
                    }
                } else {
                    // add the root of the second document to our root
                    root.appendChild(imported);
                }
            }

            // normalize all whitespace in the file
            document.getDocumentElement().normalize();

            // Use a Transformer for output
            TransformerFactory tFactory = TransformerFactory.newInstance();
            Transformer transformer = tFactory.newTransformer();

            // output the file to a string buffer.
            DOMSource source = new DOMSource(document);
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            StreamResult result = new StreamResult(stream);
            transformer.transform(source, result);
            // don't output the XML if there isn't anything.
            if (root.hasChildNodes()) {
                metadataString.append(stream.toString());
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void aggregateMetaData() {
        Thread node0 = (Thread) getAllThreads().get(0);

        // must have at least one thread
        if (node0 == null) {
            return;
        }

        // First, add all name/value pairs from the first node (any node, really)
        for (Iterator it = node0.getMetaData().keySet().iterator(); it.hasNext();) {
            String name = (String) it.next();
            String value = (String) node0.getMetaData().get(name);
            metaData.put(name, value);
        }

        // Now iterate through all nodes and remove from the master set (metaData) any that differ
        for (Iterator it = getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();
            for (Iterator it2 = thread.getMetaData().keySet().iterator(); it2.hasNext();) {
                String name = (String) it2.next();
                String value = (String) thread.getMetaData().get(name);

                String trialValue = (String) metaData.get(name);
                if (trialValue == null || !value.equals(trialValue)) {
                    metaData.remove(name);
                    uncommonMetaData.put(name, value);
                }
            }
        }

        // Now remove the normalized name/value pairs from the thread-specific structures
        for (Iterator it = getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();
            for (Iterator it2 = metaData.keySet().iterator(); it2.hasNext();) {
                String name = (String) it2.next();
                thread.getMetaData().remove(name);
            }
        }

    }

    /**
     * If the user passes in a metadata file, parse it into the trial.
     * 
     * @param metadataFileName
     * @throws IOException
     */
    public void setMetadataFile(String metadataFileName) throws IOException {
        this.metadataFile = new File(metadataFileName);
        if (!this.metadataFile.exists())
            throw new FileNotFoundException("The file " + metadataFileName + " does not exist.");
        if (!this.metadataFile.canRead())
            throw new IOException("The file " + metadataFileName + " does not have read permission.");
        if (!this.metadataFile.isFile())
            throw new FileNotFoundException(metadataFileName + " is not a valid file.");
        return;
    }

    public String getMetadataString() {
        return metadataString.toString();
    }

    public boolean getWellBehavedSnapshots() {
        return wellBehavedSnapshots;
    }

    public Map getUncommonMetaData() {
        return uncommonMetaData;
    }

    public boolean getHasThreads() {
        return hasThreads;
    }

    public boolean getHasContexts() {
        return hasContexts;
    }

    public final static int EXEC_TYPE_SINGLE = 0;
    public final static int EXEC_TYPE_MPI = 1;
    public final static int EXEC_TYPE_THREADED = 2;
    public final static int EXEC_TYPE_HYBRID = 3;
    public final static int EXEC_TYPE_OTHER = 4;

    public int getExecutionType() {
        if (getAllThreads().size() == 1) {
            if (hasMPI) {
                return EXEC_TYPE_MPI;
            } else {
                return EXEC_TYPE_SINGLE;
            }
        }

        if (getHasContexts() == false && getHasThreads() == false) {
            return EXEC_TYPE_MPI;
        }
        if (getHasContexts() == false) {
            return EXEC_TYPE_HYBRID;
        }
        return EXEC_TYPE_OTHER;
    }

}