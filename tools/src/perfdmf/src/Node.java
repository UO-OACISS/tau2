package edu.uoregon.tau.perfdmf;

import java.util.*;

/**
 * This class represents a Node.  It contains a set of Contexts and an ID.
 *  
 * <P>CVS $Id: Node.java,v 1.4 2009/06/01 18:38:26 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.4 $
 * @see		DataSource
 * @see		Context
 */
public class Node implements Comparable<Node> {

    private int nodeID = -1;
    private Map<Integer, Context> contexts = new TreeMap<Integer, Context>();
    private DataSource dataSource;

    /**
     * Creates a Node with the given ID.  This constructor is not public because Nodes should 
     * only be created by DataSource.addNode(...)
     * @param 	nodeID		ID of this Node
     */
    Node(int nodeID, DataSource dataSource) {
        this.nodeID = nodeID;
        this.dataSource = dataSource;
    }

    /**
     * Returns the Node's ID.
     * @return				the ID of this node
     */
    public int getNodeID() {
        return nodeID;
    }

    /**
     * Creates a Context with the given ID and adds it to the set of Contexts for this node.
     * If a Context with the given ID already exists, nothing is added.
     * @param 	contextID	ID for new Context
     * @return				Newly added (or existing) Context
     */
    public Context addContext(int contextID) {
        Object obj = contexts.get(new Integer(contextID));

        // return the Node if found
        if (obj != null)
            return (Context) obj;

        // otherwise, add it and return it
        Context context = new Context(this.nodeID, contextID, dataSource);
        contexts.put(new Integer(contextID), context);
        return context;
    }

    /**
     * Returns an iterator over the Contexts.
     * @return				Iterator over this Node's Contexts 
     */
    public Iterator<Context> getContexts() {
        return contexts.values().iterator();
    }

    /**
     * Returns the context with the given id.  
     * @param 	contextID	ID of context		
     * @return				Requested Context, or null if not found
     */
    public Context getContext(int contextID) {
        return contexts.get(new Integer(contextID));
    }

    /**
     * Returns the number of contexts for this node. 
     * @return				Number of Contexts
     */
    public int getNumberOfContexts() {
        return contexts.size();
    }

    /**
     * Compares this Node to another Node or Integer.
     */
    public int compareTo(Node obj) {
        return nodeID -  obj.getNodeID();
    }

    public List<Thread> getThreads() {
        List<Thread> list = new ArrayList<Thread>();
        for (Iterator<Context> it = getContexts(); it.hasNext();) {
            Context context = it.next();
            for (Iterator<Thread> it2 = context.getThreads(); it2.hasNext();) {
                Thread thread = it2.next();
                list.add(thread);
            }
        }
        return list;
    }

}
