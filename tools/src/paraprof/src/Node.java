/* 
   Node.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;

public class Node{
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
    
    //Adds a context to this nodes list of contexts.
    public void addContext(Context context){
	try{
	    contexts.addElement(context);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "N1");
	}
    }
    
    public Vector getContexts(){
	return contexts;}

    public Context getContext(int id){
	Context context = null;
	try{
	    context = (Context) contexts.elementAt(id);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "N2");
	}
	return context;
    }

    public int getNumberOfContexts(){
	return contexts.size();}
    
    //Instance data.
    int nodeID = -1;
    Vector contexts;   
}
