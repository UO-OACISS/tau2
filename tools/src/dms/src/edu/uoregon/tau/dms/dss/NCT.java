/**
 * 
 * An object representation of the node, context, thread heirarchical method
 * of representing performance data.
 *
 * <p>
 * To do: Class is complete.
 *
 *
 *
 *
 * <P>CVS $Id: NCT.java,v 1.2 2004/05/05 23:16:28 khuck Exp $</P>
 * @author	Robert Bell
 * @version	2.0
 * @since	0.1
 * @see		Node
 * @see		Context
 * @see		Thread
 */


package edu.uoregon.tau.dms.dss;

import java.util.*;

public class NCT{

    //######
    //Node methods.
    //######

    /**
     * Adds the node given to the the list of nodes. The postion in which the node
     * is added is determined by the node's id (obtained from Node.getNodeID()).
     * A node is not added if the node's id is < 0, or the node is already
     * present. Adds do not have to be consecutive (ie., nodes can be added out of order).
     *
     * @param	node The node to be inserted.
     */
    public boolean addNode(Node node){
	boolean result = false;
	try{
	    if(node.getNodeID()>=0){
		//Find the position in which this node should be added to
		//the list of nodes.
		int pos = this.getNodePosition(node);
		if(pos<0){
		    nodes.insertElementAt(node, (-(pos+1)));
		    result = true;
		}
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "N1");
	}
	return result;
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
    public Node addNode(int nodeID){
	Node node = null;
	try{
	    if(nodeID>=0){
		//Find the position in which this node should be added to
		//the list of nodes.
		int pos = this.getNodePosition(new Integer(nodeID));
		if(pos<0){
		    node = new Node(nodeID);
		    nodes.insertElementAt(node, (-(pos+1)));
		}
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "N2");
	}
	return node;
    }
    
    /**
     * Returns the number of nodes in this NCT object.
     *
     * @return	The number of nodes.
     */
    public int getNumberOfNodes(){
	return nodes.size();}

    /**
     * Returns the list of nodes in this object as a Vector.
     *
     * @return	A Vector of node objects.
     */
    public Vector getNodes(){
	return nodes;}

    /**
     * Gets the node with the specified node id.  If the node is not found, the function returns null.
     *
     * @param	nodeID The id of the node sought.
     * @return	The node found (or null if it was not).
     */
    public Node getNode(int nodeID){
	Node node = null;
	try{
	    int pos = getNodePosition(new Integer(nodeID));
	    if(pos>=0)
		node = (Node) nodes.elementAt(pos);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "N3");
	}
	return node;
    }

    //######
    //Private node methods.
    //######

    //Gets the position in the nodes list of the node with the passed in id (specified with and Integer).
    private int getNodePosition(Integer integer){
	return Collections.binarySearch(nodes, integer);}

    //Gets the position in the nodes list of the passed in node.
    private int getNodePosition(Node node){
	return Collections.binarySearch(nodes, node);}
    //######
    //End - Private node methods.
    //######

    //######
    //End - Node methods.
    //######

    //######
    //Context methods.
    //######
    //Returns the total number of contexts in this trial.
    public int getTotalNumberOfContexts(){
	int totalNumberOfContexts = -1;
	for(Enumeration e = this.getNodes().elements(); e.hasMoreElements() ;){
	    Node node = (Node) e.nextElement();
	    totalNumberOfContexts+=(node.getNumberOfContexts());
	}
	return totalNumberOfContexts;
    }

    //Returns the number of contexts on the specified node.
    public int getNumberOfContexts(int nodeID){
	return ((Node) nodes.elementAt(getNodePosition(new Integer(nodeID)))).getNumberOfContexts();}

    //Returns all the contexts on the specified node.
    public Vector getContexts(int nodeID){
	Vector vector = null;
	int pos = getNodePosition(new Integer(nodeID));
	if(pos>=0)
	    vector = ((Node) nodes.elementAt(pos)).getContexts();
	return vector;
    }

    //Returns the context on the specified node.
    public Context getContext(int nodeID, int contextID){
	Context context = null;
	int pos = getNodePosition(new Integer(nodeID));
	if(pos>=0)
	    context = ((Node) nodes.elementAt(pos)).getContext(contextID);
	return context;	       
    }
    //######
    //End - Context methods.
    //######

    //######
    //Thread methods.
    //######
    //Returns the total number of threads in this trial.
    public int getTotalNumberOfThreads(){
	int totalNumberOfThreads = -1;
	for(Enumeration e1 = this.getNodes().elements(); e1.hasMoreElements() ;){
	    Node node = (Node) e1.nextElement();
	    for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		Context context = (Context) e2.nextElement();
		totalNumberOfThreads+=(context.getNumberOfThreads());
	    }
	}
	return totalNumberOfThreads;
    }

    //Returns the number of threads on the specified node,context.
    public int getNumberOfThreads(int nodeID, int contextID){
	return (this.getContext(nodeID,contextID)).getNumberOfThreads();}

    public Vector getThreads(int nodeID, int contextID){
	Vector vector = null;
	Context context = this.getContext(nodeID,contextID);
	if(context!=null)
	    vector = context.getThreads();
	return vector;
    }

    public Thread getThread(int nodeID, int contextID, int threadID){
	Vector vector = null;
	Context context = this.getContext(nodeID,contextID);
	Thread thread = null;
	if(context!=null)
	    thread = context.getThread(threadID);
	return thread;
    }
    //######
    //End - Thread methods.
    //######

    //Instance data.
    Vector nodes = new Vector();
}

