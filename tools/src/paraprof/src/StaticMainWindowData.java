/*  
    StaticMainWindowData.java


    Title:      ParaProf
    Author:     Robert Bell
    Description:   
*/

package paraprof;

import java.util.*;
import java.lang.*;

public class StaticMainWindowData{

    public StaticMainWindowData(ParaProfTrial trial, boolean debug){
	this.trial = trial;
	this.debug = debug;
    }
    
    public Vector getSMWGeneralData(int sortType){   
	Node node;
	Context context;
	Thread thread;
	GlobalThreadDataElement globalThreadDataElement;
 	SMWServer sMWServer;
	SMWContext sMWContext;
	SMWThread sMWThread;
	SMWThreadDataElement sMWThreadDataElement;
	GlobalMapping globalMapping = trial.getGlobalMapping();
	
	Vector newList = new Vector();
    
	for(Enumeration e1 = trial.getNCT().getNodes().elements(); e1.hasMoreElements() ;){
	    node = (Node) e1.nextElement();
	    //Create a new sMWServer object and set the name properly.
	    sMWServer = new SMWServer(node.getNodeID());
	    //Add the server.
	    newList.addElement(sMWServer);
 	    for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		context = (Context) e2.nextElement();
		//Create a new context object and set the name properly.
		sMWContext = new SMWContext(sMWServer, context.getContextID());
		sMWServer.addContext(sMWContext);
		for(Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements() ;){
		    //Counts the number of SMWThreadDataElements that are actually added.
		    //It is possible (because of selection criteria - groups for example) to filter
		    //out all mappings on a particular thread.  The default at present is not to add.
		    int counter = 0; //Counts the number of SMWThreadDataElements that are actually added.
		    thread = (Thread) e3.nextElement();
		    //Create a new thread object.
		    sMWThread = new SMWThread(sMWContext, thread.getThreadID());
		    //Do not add thread to the context until we have verified counter is not zero (done after next loop).
		    //Now enter the thread data loops for this thread.
		    for(Enumeration e4 = thread.getFunctionList().elements(); e4.hasMoreElements() ;){
			globalThreadDataElement = (GlobalThreadDataElement) e4.nextElement();
			//Only want to add an element if this mapping existed on this thread.
			//Check for this.
			if((globalThreadDataElement != null) && (globalMapping.displayMapping(globalThreadDataElement.getMappingID()))){
			    //Create a new thread data object.
			    sMWThreadDataElement = new SMWThreadDataElement(trial, node.getNodeID(), context.getContextID(), thread.getThreadID(), globalThreadDataElement);
			    sMWThreadDataElement.setSortType(sortType);
			    //Add to the thread data object.
			    sMWThread.addFunction(sMWThreadDataElement);
			    counter++;
			}
		    }
		    //Sort thread and add to context if required (see above for an explanation).
		    if(counter!=0){
			Collections.sort(sMWThread.getFunctionList());
			sMWContext.addThread(sMWThread);
		    }
		}
	    }
	}
	return newList;
    }

    public Vector getMappingData(int mappingID, int listType, int sortType){
	Vector newList = new Vector();
	
	try{
	    Node node;
	    Context context;
	    Thread thread;
	    GlobalThreadDataElement globalThreadDataElement;

	    SMWThreadDataElement sMWThreadDataElement;
	    SMWThreadDataElement sMWUserThreadDataElement;
 
	    for(Enumeration e1 = trial.getNCT().getNodes().elements(); e1.hasMoreElements() ;){
		node = (Node) e1.nextElement();
		for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		    context = (Context) e2.nextElement();
		    for(Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements() ;){
			thread = (Thread) e3.nextElement();
			//Only want to add an the element with the correct mapping id.
			if(listType==1)
			    globalThreadDataElement = thread.getFunction(mappingID);
			else
			    globalThreadDataElement = thread.getUserevent(mappingID);
			if(globalThreadDataElement != null){
			    //Create a new thread data object.
			    sMWThreadDataElement = new SMWThreadDataElement(trial, node.getNodeID(), context.getContextID(), thread.getThreadID(), globalThreadDataElement);
			    sMWThreadDataElement.setSortType(sortType);
			    
			    newList.add(sMWThreadDataElement);
			}
		    }
		}
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWD01");
	}
	Collections.sort(newList);
	return newList;
    }

    //Returns a list of SMWThreadDataElement elements.  List does not include
    //GlobalMappingElements that have not had their mean values set.
    //Function also takes into account the current group selection.
    public Vector getMeanData(int sortType){
	GlobalMapping globalMapping = trial.getGlobalMapping();
	Vector list = null;
	Vector newList = null;
	SMWThreadDataElement sMWThreadDataElement = null;

	list = globalMapping.getMapping(0);
	newList = new Vector();

	for(Enumeration e = list.elements(); e.hasMoreElements() ;){
	    GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
	    if(globalMappingElement.getMeanValuesSet()){
		if(globalMapping.displayMapping(globalMappingElement.getMappingID())){
		    sMWThreadDataElement = new SMWThreadDataElement(trial, -1, -1, -1, globalMappingElement);
		    sMWThreadDataElement.setSortType(sortType);
		    newList.addElement(sMWThreadDataElement);
		}
	    }
	}
	Collections.sort(newList);
	return newList;
    }
    
    //Returns a list of SMWThreadDataElement elements.  List does not include
    //GlobalThreadDataElements that are not present on this Thread (indicated by
    //a null value on the Thread).
    //Function also takes into account the current group selection.
    public Vector getThreadData(int nodeID, int contextID, int threadID, int listType, int sortType){
	GlobalMapping globalMapping = trial.getGlobalMapping();
	Vector list = null;
	Vector newList = null;
	GlobalThreadDataElement globalThreadDataElement = null;
	SMWThreadDataElement sMWThreadDataElement = null;
    
	System.out.println("listType: " + listType);

	switch(listType){
	case 1:
	    list = ((Thread)trial.getNCT().getThread(nodeID,contextID,threadID)).getFunctionList();
	    break;
	case 2:
	    list = ((Thread)trial.getNCT().getThread(nodeID,contextID,threadID)).getUsereventList();
	    break;
	default:
	    UtilFncs.systemError(null, null, "Unexpected list type - SMWD value: " + listType);
	}

	newList = new Vector();

	for(Enumeration e1 = list.elements(); e1.hasMoreElements() ;){
	    globalThreadDataElement = (GlobalThreadDataElement) e1.nextElement();
	    System.out.println("userevent: "+globalThreadDataElement.userevent);
	    System.out.println("uename: " + globalThreadDataElement.getMappingName());
	    if(globalThreadDataElement!=null){
		if(globalMapping.displayMapping(globalThreadDataElement.getMappingID())){
		    sMWThreadDataElement = new SMWThreadDataElement(trial, nodeID, contextID, threadID, globalThreadDataElement);
		    sMWThreadDataElement.setSortType(sortType);
		    newList.addElement(sMWThreadDataElement);
		}
	    }
	}
	Collections.sort(newList);
	return newList;
    }

    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}
    //####################################
    //Instance Data.
    //####################################
    private ParaProfTrial trial = null;
    private Vector sMWGeneralData = new Vector();

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance Data.
    //####################################

}
