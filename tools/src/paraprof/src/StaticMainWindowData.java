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

    public StaticMainWindowData(ParaProfTrial trial){
	this.trial = trial;
    }
    
    public void buildSMWGeneralData(){   
	Node node;
	Context context;
	Thread thread;
	GlobalThreadDataElement globalThreadDataElement;
 	SMWServer sMWServer;
	SMWContext sMWContext;
	SMWThread sMWThread;
	SMWThreadDataElement sMWThreadDataElement;

	//Clear the sMWGeneralData list for safety.
	sMWGeneralData.removeAllElements();
    
	for(Enumeration e1 = trial.getNCT().getNodes().elements(); e1.hasMoreElements() ;){
	    node = (Node) e1.nextElement();
	    //Create a new sMWServer object and set the name properly.
	    sMWServer = new SMWServer(node.getNodeID());
	    //Add the server.
	    sMWGeneralData.addElement(sMWServer);
 	    for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		context = (Context) e2.nextElement();
		//Create a new context object and set the name properly.
		sMWContext = new SMWContext(sMWServer, context.getContextID());
		sMWServer.addContext(sMWContext);
		for(Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements() ;){
		    thread = (Thread) e3.nextElement();
		    //Create a new thread object.
		    sMWThread = new SMWThread(sMWContext, thread.getThreadID());
		    //Add to the context.
		    sMWContext.addThread(sMWThread);
		    //Now enter the thread data loops for this thread.
		    for(Enumeration e4 = thread.getFunctionList().elements(); e4.hasMoreElements() ;){
			globalThreadDataElement = (GlobalThreadDataElement) e4.nextElement();
			//Only want to add an element if this mapping existed on this thread.
			//Check for this.
			if(globalThreadDataElement != null){
			    //Create a new thread data object.
			    sMWThreadDataElement = new SMWThreadDataElement(trial, node.getNodeID(), context.getContextID(), thread.getThreadID(), globalThreadDataElement);
			    //Add to the thread data object.
			    sMWThread.addFunction(sMWThreadDataElement);
			}
		    }
		}
	    }
	}
    }

    public Vector getSMWGeneralData(int sortType){
	SMWServer sMWServer;
	SMWContext sMWContext;
	SMWThread sMWThread;
	SMWThreadDataElement sMWThreadDataElement;
    
	for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;){
	    sMWServer = (SMWServer) e1.nextElement();
	    for(Enumeration e2 = sMWServer.getContextList().elements(); e2.hasMoreElements() ;){
		sMWContext = (SMWContext) e2.nextElement();
		for(Enumeration e3 = sMWContext.getThreadList().elements(); e3.hasMoreElements() ;){
		    sMWThread = (SMWThread) e3.nextElement();
		    for(Enumeration e4 = sMWThread.getFunctionList().elements(); e4.hasMoreElements() ;){
			sMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
			sMWThreadDataElement.setSortType(sortType);
		    }
		    Collections.sort(sMWThread.getFunctionList());
		}
	    }
	}
	return sMWGeneralData;
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
			    globalThreadDataElement = (GlobalThreadDataElement) thread.getFunctionList().elementAt(mappingID);
			else{//User events might not have fired on this thread.
			    Vector userEventList = thread.getUsereventList();
			    if(userEventList!=null)
				globalThreadDataElement = (GlobalThreadDataElement) userEventList.elementAt(mappingID);
			    else
				globalThreadDataElement = null;
			}
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
	    ParaProf.systemError(e, null, "SMWD01");
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
	boolean isSelectedGroupOn = false;
	int selectedGroupID = 0;
	SMWThreadDataElement sMWThreadDataElement = null;

	list = globalMapping.getMapping(0);
	newList = new Vector();

	if(globalMapping.getIsSelectedGroupOn()){
	    isSelectedGroupOn = true;
	    selectedGroupID = globalMapping.getSelectedGroupID();
	}

	if(isSelectedGroupOn){
	    for(Enumeration e = list.elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		if(globalMappingElement.getMeanValuesSet()){
		    if(globalMappingElement.isGroupMember(selectedGroupID)){
			sMWThreadDataElement = new SMWThreadDataElement(trial, -1, -1, -1, globalMappingElement);
			sMWThreadDataElement.setSortType(sortType);
			newList.addElement(sMWThreadDataElement);
		    }
		}
	    }
	}
	else{
	    for(Enumeration e = list.elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		if(globalMappingElement.getMeanValuesSet()){
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
	boolean isSelectedGroupOn = false;
	int selectedGroupID = 0;
	GlobalThreadDataElement globalThreadDataElement = null;
	SMWThreadDataElement sMWThreadDataElement = null;
    
	switch(listType){
	case 1:
	    list = ((Thread)trial.getNCT().getThread(nodeID,contextID,threadID)).getFunctionList();
	    break;
	case 2:
	    list = ((Thread)trial.getNCT().getThread(nodeID,contextID,threadID)).getUsereventList();
	    break;
	default:
	    ParaProf.systemError(null, null, "Unexpected list type - SMWD value: " + listType);
	}

	newList = new Vector();

	if(globalMapping.getIsSelectedGroupOn()){
	    isSelectedGroupOn = true;
	    selectedGroupID = globalMapping.getSelectedGroupID();
	}

	if(isSelectedGroupOn){
	    for(Enumeration e1 = list.elements(); e1.hasMoreElements() ;){
		globalThreadDataElement = (GlobalThreadDataElement) e1.nextElement();
		if(globalThreadDataElement!=null){
		    if(globalThreadDataElement.isGroupMember(selectedGroupID)){
			sMWThreadDataElement = new SMWThreadDataElement(trial, nodeID, contextID, threadID, globalThreadDataElement);
			sMWThreadDataElement.setSortType(sortType);
			newList.addElement(sMWThreadDataElement);
		    }
		}
	    }
	}
	else{
	    for(Enumeration e2 = list.elements(); e2.hasMoreElements() ;){
		globalThreadDataElement = (GlobalThreadDataElement) e2.nextElement();
		if(globalThreadDataElement!=null){
		    sMWThreadDataElement = new SMWThreadDataElement(trial, nodeID, contextID, threadID, globalThreadDataElement);
		    sMWThreadDataElement.setSortType(sortType);
		    newList.addElement(sMWThreadDataElement);
		}
	    }
	}
	
	Collections.sort(newList);
	return newList;
    }

    //####################################
    //Instance Data.
    //####################################
    private ParaProfTrial trial = null;
    private Vector sMWGeneralData = new Vector();
    //####################################
    //End - Instance Data.
    //####################################

}
