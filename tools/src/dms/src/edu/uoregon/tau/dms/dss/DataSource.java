package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.io.*;
import java.sql.*;

/**
 * This class represents a data source.  After loading, data is availiable through the
 * public methods.
 *  
 * <P>CVS $Id: DataSource.java,v 1.8 2005/01/31 23:10:25 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.8 $
 * @see		TrialData
 * @see		NCT
 */
public abstract class DataSource {

    private boolean userEventsPresent = false;
    private boolean callPathDataPresent = false;
    private boolean groupNamesPresent = false;

    // data structures
    private Vector metrics = null;
    protected Thread meanData = null;
    protected Thread totalData = null;
    private Map nodes = new TreeMap();

    private Map functions = new TreeMap();
    private Map groups = new TreeMap();
    private Map userEvents = new TreeMap();

    // just a holder for the output of getMaxNCTNumbers(), makes subsequent calls instantaneous
    private int[] maxNCT = null;

    public DataSource() {
    }

    abstract public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException;

    abstract public int getProgress();

    abstract public void cancelLoad();

    public Thread getMeanData() {
        return meanData;
    }

    public Thread getTotalData() {
        return totalData;
    }

    protected void setCallPathDataPresent(boolean callPathDataPresent) {
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

    public Function addFunction(String name, int numMetrics) {
        name = name.trim();
        Object obj = functions.get(name);

        // return the function if found
        if (obj != null)
            return (Function) obj;

        // otherwise, add it and return it
        Function function = new Function(name, functions.size(), numMetrics);
        functions.put(name, function);
        return function;
    }

    public Function getFunction(String name) {
        Function f = (Function) functions.get(name.trim());
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

        if (obj != null)
            return (UserEvent) obj;

        UserEvent userEvent = new UserEvent(name, userEvents.size() + 1);
        userEvents.put(name, userEvent);
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

    public Group addGroup(String name) {
        Object obj = groups.get(name);

        if (obj != null)
            return (Group) obj;

        Group group = new Group(name, groups.size() + 1);
        groups.put(name, group);
        return group;
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
                if (node.getNodeID() > maxNCT[0])
                    maxNCT[0] = node.getNodeID();
                for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                    Context context = (Context) it2.next();
                    if (context.getContextID() > maxNCT[1])
                        maxNCT[1] = context.getContextID();
                    for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                        edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();
                        if (thread.getThreadID() > maxNCT[2])
                            maxNCT[2] = thread.getThreadID();
                    }
                }
            }

        }
        return maxNCT;
    }

    /**
     * Set the Vector of Metrics for this DataSource.
     * 
     * @param metrics
     *            Vector of Metrics
     */
    public void setMetrics(Vector metrics) {
        this.metrics = metrics;
    }

    /**
     * Adds a metric to the DataSource's metrics vector.
     * 
     * @param metric
     *            Metric to be added
     */
    public void addMetric(Metric metric) {
        if (this.metrics == null) {
            this.metrics = new Vector();
        }

        metric.setID(this.getNumberOfMetrics());
        metrics.add(metric);
    }

    /**
     * Adds a metric to the DataSource's metrics vector (given as a String).
     * 
     * @param metric
     *            Name of metric to be added
     */
    public Metric addMetric(String metricName) {
        if (metrics != null) {
            for (Iterator it = metrics.iterator(); it.hasNext();) {
                Metric metric = (Metric) it.next();
                if (metric.getName().equals(metricName))
                    return metric;
            }
        }

        Metric metric = new Metric();
        metric.setName(metricName);
        addMetric(metric);
        return metric;
    }

    /**
     * Get a the Vector of Metrics
     * 
     * @return Vector of Metrics
     */
    public Vector getMetrics() {
        return this.metrics;
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
        //Try getting the metric name.
        if (this.metrics != null)
            return metrics.size();
        else
            return -1;
    }

    /*
     * After loading all data, this function should be called to generate all
     * the derived data
     */
    protected void generateDerivedData() {
        for (Enumeration e = this.getThreads().elements(); e.hasMoreElements();) {
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

        double[] exclSum = new double[numMetrics];
        double[] inclSum = new double[numMetrics];
        double callSum = 0;
        double subrSum = 0;

        double topLevelInclSum[] = new double[numMetrics];

        Iterator l = this.getFunctions();

        if (meanData == null) {
            meanData = new Thread(-1, -1, -1, numMetrics);
        }

        if (totalData == null) {
            totalData = new Thread(-2, -2, -2, numMetrics);
        }

        // must always iterate through all metrics regardless to find the top level timers, I think???
        for (int i = 0; i < numMetrics; i++) {

            for (Iterator it = this.getNodes(); it.hasNext();) {
                Node node = (Node) it.next();
                for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                    Context context = (Context) it2.next();
                    for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                        Thread thread = (Thread) it3.next();
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
                meanData.addFunctionProfile(meanProfile);
            }
            function.setMeanProfile(meanProfile);

            // get/create the FunctionProfile for total
            FunctionProfile totalProfile = totalData.getFunctionProfile(function);
            if (totalProfile == null) {
                totalProfile = new FunctionProfile(function, numMetrics);
                totalData.addFunctionProfile(totalProfile);
            }
            function.setTotalProfile(totalProfile);

            callSum = 0;
            subrSum = 0;
            for (int i = 0; i < numMetrics; i++) {
                exclSum[i] = 0;
                inclSum[i] = 0;
            }

            // this must be stored somewhere else, but I'm going to compute it since I don't know where
            int numThreads = 0;

            for (Iterator it = this.getNodes(); it.hasNext();) {
                Node node = (Node) it.next();
                for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                    Context context = (Context) it2.next();
                    for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                        edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();
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

                totalProfile.setNumCalls(callSum);
                totalProfile.setNumSubr(subrSum);

                // mean is just the total / numThreads
                meanProfile.setNumCalls((double) callSum / numThreads);
                meanProfile.setNumSubr((double) subrSum / numThreads);
            }

            for (int i = startMetric; i <= endMetric; i++) {

                totalProfile.setExclusive(i, exclSum[i]);
                totalProfile.setInclusive(i, inclSum[i]);
                if (totalProfile.getNumCalls() != 0)
                    totalProfile.setInclusivePerCall(i, inclSum[i] / totalProfile.getNumCalls());

                // mean data computed as above in comments
                meanProfile.setExclusive(i, exclSum[i] / numThreads);
                meanProfile.setInclusive(i, inclSum[i] / numThreads);

                if (meanProfile.getNumCalls() != 0)
                    meanProfile.setInclusivePerCall(i, inclSum[i] / numThreads / meanProfile.getNumCalls());

                if (topLevelInclSum[i] != 0) {
                    totalProfile.setInclusivePercent(i, totalProfile.getInclusive(i) / topLevelInclSum[i] * 100);
                    totalProfile.setExclusivePercent(i, totalProfile.getExclusive(i) / topLevelInclSum[i] * 100);
                    meanProfile.setInclusivePercent(i, totalProfile.getInclusivePercent(i));
                    meanProfile.setExclusivePercent(i, totalProfile.getExclusivePercent(i));
                }
            }
        }
    }

    /**
     * Creates and then adds a node with the given id to the the list of nodes. 
     * The postion in which the node is added is determined by given id.
     * A node is not added if the id is < 0, or that node id is already
     * present. Adds do not have to be consecutive (ie., nodes can be added out of order).
     * The node created will have an id matching the given id.
     *
     * @param	nodeID The id of the node to be added.
     * @return	The Node that was added.
     */
    public Node addNode(int nodeID) {
        Object obj = nodes.get(new Integer(nodeID));

        // return the Node if found
        if (obj != null)
            return (Node) obj;

        // otherwise, add it and return it
        Node node = new Node(nodeID);
        nodes.put(new Integer(nodeID), node);
        return node;
    }

    /**
     * Gets the node with the specified node id.  If the node is not found, the function returns null.
     *
     * @param	nodeID The id of the node sought.
     * @return	The node found (or null if it was not).
     */
    public Node getNode(int nodeID) {
        return (Node) nodes.get(new Integer(nodeID));
    }

    /**
     * Returns the number of nodes in this NCT object.
     *
     * @return	The number of nodes.
     */
    public int getNumberOfNodes() {
        return nodes.size();
    }

    /**
     * Returns the list of nodes in this object as a Vector.
     *
     * @return	A Vector of node objects.
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
        int totalNumberOfThreads = -1;
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

    public Vector getThreads() {
        Vector vector = new Vector();
        for (Iterator it = this.getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                    edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();
                    vector.add(thread);
                }
            }
        }
        return vector;
    }

    public Vector getThreads(int nodeID) {
        Vector vector = new Vector();
        Node node = this.getNode(nodeID);
        for (Iterator it2 = node.getContexts(); it2.hasNext();) {
            Context context = (Context) it2.next();
            for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();
                vector.add(thread);
            }
        }
        return vector;
    }

    public Iterator getThreads(int nodeID, int contextID) {
        Context context = this.getContext(nodeID, contextID);
        if (context != null)
            return context.getThreads();
        return null;
    }

    public Thread getThread(int nodeID, int contextID, int threadID) {
        Vector vector = null;
        Context context = this.getContext(nodeID, contextID);
        Thread thread = null;
        if (context != null)
            thread = context.getThread(threadID);
        return thread;
    }

    protected boolean debug() {
        //return true;
        return UtilFncs.debug;
    }
}