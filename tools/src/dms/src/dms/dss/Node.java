/* 
   Name: Node.java
   Author:     Robert Bell
   Description:  
*/

package dms.dss;

import java.util.*;

public class Node implements Comparable{
    public Node(){
	contexts = new Vector();}

    public Node(int nodeID){
	this.nodeID = nodeID;
	contexts = new Vector();
    }

    public void setNodeID(int nodeID){
	this.nodeID = nodeID;}

    public int getNodeID(){
	return nodeID;}
    
    //Adds the specified context to the list of contexts. 
    public void addContext(Context context){
	try{
	    if(context.getContextID()<0){
		System.out.println("Error - Invalid context id (id less than zero). Context not added!");
		return;
	    }

	    int pos = this.getContextPosition(context);
	    if(pos>=0)
		System.out.println("Error - Context already present. Context not added!");
	    else
		contexts.insertElementAt(context, (-(pos+1)));
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "N1");
	}
    }

    //Creates a context with the specified context id and adds it to the list of contexts.
    public Context addContext(int contextID){
	Context context = null;
	try{
	    if(contextID<0){
		System.out.println("Error - Invalid context id (id less than zero). Context not added!");
		return null;
	    }

	    int pos = this.getContextPosition(new Integer(contextID));
	    if(pos>=0)
		System.out.println("Error - Context already present. Context not added!");
	    else{
		context = new Context(nodeID, contextID);
		contexts.insertElementAt(context, (-(pos+1)));
	    }
	    return context;
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "N2");
	}
	return context;
    }
    
    public Vector getContexts(){
	return contexts;}

    //Gets the context with the specified context id.  If the context is not found, the function returns
    //null.
    public Context getContext(int contextID){
	Context context = null;
	try{
	    int pos = getContextPosition(new Integer(contextID));
	    if(pos>=0)
		context = (Context) contexts.elementAt(pos);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "N3");
	}
	return context;
    }

    public int getNumberOfContexts(){
	return contexts.size();}

    private int getContextPosition(Integer integer){
	return Collections.binarySearch(contexts, integer);}

    private int getContextPosition(Context context){
	return Collections.binarySearch(contexts, context);}

    public int compareTo(Object obj){
	if(obj instanceof Integer)
	    return nodeID - ((Integer)obj).intValue();
	else
	    return nodeID - ((Node)obj).getNodeID();
    }
    
    //Instance data.
    int nodeID = -1;
    Vector contexts;   
}
