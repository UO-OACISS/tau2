/* 
  CallPathUtilFuncs.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package paraprof;

import java.util.*;

public class CallPathUtilFuncs{
    public CallPathUtilFuncs(){}

    public static boolean isAvailable(ListIterator l){
	GlobalMappingElement gme = null;
	String s = null;
	boolean result = false;
	while(l.hasNext()){
	    gme = (GlobalMappingElement) l.next();
	    s = gme.getMappingName();
	    if(s!=null){
		if(s.indexOf("=>")>0){
		    gme.setCallPathObject(true);
		    result = true;
		}
	    }
	}
	return result;
    }

    public static void buildRelations(GlobalMapping gm){
	try{
	    GlobalMappingElement gme1 = null;
	    GlobalMappingElement gme2 = null;
	    GlobalMappingElement gme3 = null;
	    String s = null;
	    String parent = null;
	    String child = null;
	    int location = -1;
	    ParaProfIterator l = null;
	    
	    l = (ParaProfIterator)gm.getMappingIterator(0);
	    while(l.hasNext()){
		gme1 = (GlobalMappingElement) l.next();
		s = gme1.getMappingName();
		location = s.lastIndexOf("=>");
		if(location>0){
		    child = s.substring(location+3,(s.length()-2));
		    s = s.substring(0,location-1);
		    location = s.lastIndexOf("=>");
		    if(location>0){
			parent = s.substring(location+3); 
		    }
		    else
			parent = s;
		    //Update parent/child relationships.
		    gme2 = gm.getGlobalMappingElement(parent,0); 
		    gme3 = gm.getGlobalMappingElement(child,0);
		    gme2.addChild(gme3.getGlobalID(),gme1.getGlobalID());
		    gme3.addParent(gme2.getGlobalID(),gme1.getGlobalID());
		}
	    }
	}
	catch(Exception e){
	    System.out.println(e.toString());
	    e.printStackTrace();
	}
    }

    public static void trimCallPathData(Trial trial,
				 int node,
				 int context,
				 int thread){

	ListIterator l1 = null;
	ListIterator l2 = null;
	ListIterator l3 = null;
	GlobalMapping gm = trial.getGlobalMapping();
	GlobalMappingElement gme1 = null;
	GlobalMappingElement gme2 = null;
	Integer listValue = null;
	String s = null;
	Vector staticServerList = null;
	GlobalServer globalServer = null;
	GlobalContext globalContext = null;
	GlobalThread globalThread = null;
	Vector threadDataList = null;
	GlobalThreadDataElement gtde = null;
	SMWThreadDataElement smwtde = null;
	
	//Create a pruned list from the global list.
	//Want to grab a reference to the global list as
	//this list contains null references for mappings
	//which do not exist. Makes lookup much faster.
	
	//Find the correct global thread data element.
	staticServerList = trial.getNodes();
	globalServer = (GlobalServer) staticServerList.elementAt(node);
	Vector tmpRef = globalServer.getContextList();
	globalContext = (GlobalContext) tmpRef.elementAt(context);
	tmpRef = globalContext.getThreadList();
	globalThread = (GlobalThread) tmpRef.elementAt(thread);
	threadDataList = globalThread.getThreadDataList();

	//Check to make sure that we have not trimmed before.
	if(globalThread.trimmed())
	    return;
	
	l1 = gm.getMappingIterator(0);
	while(l1.hasNext()){
	    gme1 = (GlobalMappingElement) l1.next();
	    gtde = (GlobalThreadDataElement) threadDataList.elementAt(gme1.getGlobalID());
	    if((!(gme1.isCallPathObject())) && (gtde!=null)){
		l2 = gme1.getParentsIterator();
		while(l2.hasNext()){
		    listValue = (Integer)l2.next();
		    if(threadDataList.elementAt(listValue.intValue())!=null){
			int location = gtde.addParent(listValue.intValue());
			l3 = gme1.getCallPathIDParents(listValue.intValue());
			while(l3.hasNext()){
			    int pathID = ((Integer)l3.next()).intValue();
			    if(threadDataList.elementAt(pathID)!=null)
				gtde.addParentCallPathID(location, pathID);
			}
		    }
		}
		l2 = gme1.getChildrenIterator();
		while(l2.hasNext()){
		    listValue = (Integer)l2.next();
		    if(threadDataList.elementAt(listValue.intValue())!=null){
			int location = gtde.addChild(listValue.intValue());
			l3 = gme1.getCallPathIDChildren(listValue.intValue());
			while(l3.hasNext()){
			    int pathID = ((Integer)l3.next()).intValue();
			    if(threadDataList.elementAt(pathID)!=null)
					gtde.addChildCallPathID(location, pathID);
			}
		    }
		}
	    }
	}

	//Set this thread to indicate that it has been trimmed.
	globalThread.setTrimmed(true);
    }
}
