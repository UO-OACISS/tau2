package edu.uoregon.tau.perfdmf;

import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;

/**
 * This class represents a Context.  It contains a set of Threads, a nodeID and an contextID.
 *  
 * <P>CVS $Id: Context.java,v 1.4 2008/10/31 00:45:01 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.4 $
 * @see		Node
 * @see		Thread
 */
public class Context implements Comparable<Context> {

    private int nodeID = -1;
    private int contextID = -1;
    private Map<Integer, Thread> threads = new TreeMap<Integer, Thread>();
    private DataSource dataSource;

    /**
     * Creates a Context with the given IDs.  This constructor is not public because Contexts should 
     * only be created by Node.addContext(...)
     * 
     * @param nodeID		ID of the node this context is a member of
     * @param contextID		ID of this context
     */
    public Context(int nodeID, int contextID, DataSource dataSource) {
        this.nodeID = nodeID;
        this.contextID = contextID;
        this.dataSource = dataSource;
    }

    /**
     * Returns the Contexts NodeID, the ID of the Node this Context belongs to.
     *
     * @return				NodeID of this Context.
     */
    public int getNodeID() {
        return nodeID;
    }

    /**
     * Returns the Context's ID.
     *
     * @return				the ID of this Context
     */
    public int getContextID() {
        return contextID;
    }

    /**
     * Creates a thread with the specified threadID (with default capacity 1) and adds it to the 
     * list of threads.  If a Thread with the given ID already exists, nothing is added.
     * @param 	threadID	ID for new Thread
     * @return				Newly added (or existing) Thread
     */
    public Thread addThread(int threadID) {
        return addThread(threadID, dataSource.getNumberOfMetrics());
    }

    /**
     * Creates a thread with the specified threadID and capacity (number of metrics) and adds it 
     * to the list of threads.  If a Thread with the given ID already exists, nothing is added.
     * @param 	threadID	ID for new Thread
     * @param 	capacity	Number of metrics to allocate space for
     * @return				Newly added (or existing) Thread
     */
    public Thread addThread(int threadID, int capacity) {
        Object obj = threads.get(new Integer(threadID));

        // return the Node if found
        if (obj != null) {
            return (Thread) obj;
        }

        // otherwise, add it and return it
        Thread thread = new Thread(this.nodeID, this.contextID, threadID, capacity, dataSource);
        threads.put(new Integer(threadID), thread);
        return thread;
    }

    /**
     * Returns an iterator over the Threads.
     * @return				Iterator over this Context's Threads 
     */
    public Iterator<Thread> getThreads() {
        return threads.values().iterator();
    }

    /**
     * Returns the Thread with the given id.  
     * @param 	threadID	ID of thread		
     * @return				Requested Thread, or null if not found
     */
    public Thread getThread(int threadID) {
        return threads.get(new Integer(threadID));
    }

    /**
     * Returns the number of Threads for this Context. 
     * @return				Number of Threads
     */
    public int getNumberOfThreads() {
        return threads.size();
    }

    /**
     * Compares this Context to another Context or Integer.
     */
    public int compareTo(Context obj) {
        return contextID - obj.getContextID();
    }
}
