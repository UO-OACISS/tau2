/* 
   NCT.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

/*
  To do: 
  1) Consider some improvements with regard to getNumberOfContexts/Threads.
  Currently the paradigm is for one of flexibility. If this becomes a problem,
  consider introducing some lookup improvements.
*/

package paraprof;

import java.util.*;

public class NCT{
    public NCT(){}

    //######
    //Node methods.
    //######
    //Adds the specified node to the list of nodes.
    public void addNode(Node node){
	try{
	    if(node.getNodeID()<0){
		System.out.println("Error - Invalid node id (id less than zero). Node not added!");
		return;
	    }

	    int pos = this.getNodePosition(node);
	    if(pos>=0)
		System.out.println("Error - Node already present. Node not added!");
	    else
		nodes.insertElementAt(node, (-(pos+1)));
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "N1");
	}
    }

    //Creates a node with the specified node id and adds it to the list of nodes.
    public Node addNode(int nodeID){
	Node node = null;
	try{
	    if(nodeID<0){
		System.out.println("Error - Invalid node id (id less than zero). Node not added!");
		return null;
	    }

	    int pos = this.getNodePosition(new Integer(nodeID));
	    if(pos>=0)
		System.out.println("Error - Node already present. Node not added!");
	    else{
		node = new Node(nodeID);
		nodes.insertElementAt(node, (-(pos+1)));
	    }
	    return node;
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "N2");
	}
	return node;
    }
    
    public int getNumberOfNodes(){
	return nodes.size();}

    public Vector getNodes(){
	return nodes;}

    //Gets the node with the specified node id.  If the node is not found, the function returns
    //null.
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

    private int getNodePosition(Integer integer){
	return Collections.binarySearch(nodes, integer);}

    private int getNodePosition(Node node){
	return Collections.binarySearch(nodes, node);}
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

