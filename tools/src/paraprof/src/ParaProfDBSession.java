/* 
   ParaProfDBSession.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

/*
  To do: 
  1) Add some sanity checks to make sure that multiple metrics really do belong together.
  For example, wrap the creation of nodes, contexts, threads, global mapping elements, and
  the like so that they do not occur after the first metric has been loaded.  This will
  not of course ensure 100% that the data is consistent, but it will at least prevent the
  worst cases.
*/

package paraprof;



import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

import java.io.*;
import java.util.*;
import dms.dss.*;

public class ParaProfDBSession extends ParaProfDataSession{

    public ParaProfDBSession(){
	super();
    }

    /**
     * Initialize the DataSession object.
     *
     * @param	obj	an implementation-specific object required to initialize the DataSession
     */
    public void initialize(Object obj){
	try{
	    //######
	    //Frequently used items.
	    //######
	    PerfDBSession perfDBSession = (PerfDBSession) obj;

	    int metric = 0;
	    GlobalMappingElement globalMappingElement = null;
	    GlobalThreadDataElement globalThreadDataElement = null;
	    
	    Node node = null;
	    Context context = null;
	    Thread thread = null;
	    int nodeID = -1;
	    int contextID = -1;
	    int threadID = -1;
	    int mappingID = -1;

	    //Vector localMap = new Vector();
	    //######
	    //End - Frequently used items.
	    //######
	    System.out.println("Processing data, please wait ......");
	    long time = System.currentTimeMillis();
	    
	    this.setMetrics(perfDBSession.getMetrics());
	    int numberOfMetrics = this.getNumberOfMetrics();
	    for(int i=0;i<numberOfMetrics;i++)
		this.getGlobalMapping().increaseVectorStorage();

	    //Add the functions.
	    ListIterator l = perfDBSession.getFunctions();
	    while(l.hasNext()){
		    Function f = (Function) l.next();
		    int id = this.getGlobalMapping().addGlobalMapping(f.getName(), 0, numberOfMetrics);
		    
		    //Add element to the localMap for more efficient lookup later in the function.
		    //localMap.add(new FunIndexFunIDPair(f.getIndexID(), id));
		    
		    globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(id, 0);
		    for(int i=0;i<numberOfMetrics;i++){
			FunctionDataObject fdo = f.getMeanSummary(i);
			globalMappingElement.setMeanExclusiveValue(i, fdo.getExclusive());
			globalMappingElement.setMeanExclusivePercentValue(i, fdo.getExclusivePercentage());
			globalMappingElement.setMeanInclusiveValue(i, fdo.getInclusive());
			globalMappingElement.setMeanInclusivePercentValue(i, fdo.getInclusivePercentage());

			if((this.getGlobalMapping().getMaxMeanExclusiveValue(i)) < fdo.getExclusive()){
			    this.getGlobalMapping().setMaxMeanExclusiveValue(i, fdo.getExclusive());}
			if((this.getGlobalMapping().getMaxMeanExclusivePercentValue(i)) < fdo.getExclusivePercentage()){
			    this.getGlobalMapping().setMaxMeanExclusivePercentValue(i, fdo.getExclusivePercentage());}

			if((this.getGlobalMapping().getMaxMeanInclusiveValue(i)) < fdo.getInclusive()){
			    this.getGlobalMapping().setMaxMeanInclusiveValue(i, fdo.getInclusive());}
			if((this.getGlobalMapping().getMaxMeanInclusivePercentValue(i)) < fdo.getInclusivePercentage()){
			    this.getGlobalMapping().setMaxMeanInclusivePercentValue(i, fdo.getInclusivePercentage());}
		    }
		    globalMappingElement.setMeanValuesSet(true);
	    }
	    
	    //Collections.sort(localMap);

	    //Increase storage.
	    for(int i=0;i<numberOfMetrics;i++){
		if(this.debug())
		    System.out.println("Increasing the storage for the new counter.");
		for(Enumeration e3 = this.getNCT().getNodes().elements(); e3.hasMoreElements() ;){
		    node = (Node) e3.nextElement();
		    for(Enumeration e4 = node.getContexts().elements(); e4.hasMoreElements() ;){
			context = (Context) e4.nextElement();
			for(Enumeration e5 = context.getThreads().elements(); e5.hasMoreElements() ;){
			    thread = (Thread) e5.nextElement();
			    thread.incrementStorage();
			    for(Enumeration e6 = thread.getFunctionList().elements(); e6.hasMoreElements() ;){
				GlobalThreadDataElement ref = (GlobalThreadDataElement) e6.nextElement();
				//Only want to add an element if this mapping existed on this thread.
				//Check for this.
				if(ref != null)
				    ref.incrementStorage();
			    }
			}
		    }
		}
		if(this.debug())
		    System.out.println("Done increasing the storage for the new counter.");
	    }
	    
	    l = perfDBSession.getFunctionData();
	    while(l.hasNext()){
		FunctionDataObject fdo = (FunctionDataObject) l.next();
		node = this.getNCT().getNode(fdo.getNode());
		if(node==null)
		    node = this.getNCT().addNode(fdo.getNode());
		context = node.getContext(fdo.getContext());
		if(context==null)
		    context = node.addContext(fdo.getContext());
		thread = context.getThread(fdo.getThread());
		if(thread==null){
		    thread = context.addThread(fdo.getThread());
		    thread.setDebug(this.debug());
		    thread.initializeFunctionList(this.getNumberOfMappings());
		}
		
		//Get GlobalMappingElement and GlobalThreadDataElement.
		
		//Obtain the mapping id from the local map.
		//int pos = Collections.binarySearch(localMap, new FunIndexFunIDPair(fdo.getFunctionIndexID(),0));
		//mappingID = ((FunIndexFunIDPair)localMap.elementAt(pos)).paraProfId;
		
		mappingID = this.getGlobalMapping().getMappingID(perfDBSession.getFunction(fdo.getFunctionIndexID()).getName(),0);
		globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 0);
		globalMappingElement.incrementCounter();
		globalThreadDataElement = thread.getFunction(mappingID);
		if(globalThreadDataElement == null){
		    globalThreadDataElement = 
			new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(mappingID, 0), false, numberOfMetrics);
		    thread.addFunction(globalThreadDataElement,mappingID );
		}
		
		for(int i=0;i<numberOfMetrics;i++){
		    globalThreadDataElement.setExclusiveValue(i, fdo.getExclusive(i));
		    globalThreadDataElement.setInclusiveValue(i, fdo.getInclusive(i));
		    globalThreadDataElement.setExclusivePercentValue(i, fdo.getExclusivePercentage(i));
		    globalThreadDataElement.setInclusivePercentValue(i, fdo.getInclusivePercentage(i));
		    globalThreadDataElement.setUserSecPerCall(i, fdo.getInclusivePerCall(i));
		
		    globalThreadDataElement.setNumberOfCalls(fdo.getNumCalls());
		    globalThreadDataElement.setNumberOfSubRoutines(fdo.getNumSubroutines());
		    
		    //Set the max values.
		    if((globalMappingElement.getMaxExclusiveValue(metric)) < fdo.getExclusive(i))
			globalMappingElement.setMaxExclusiveValue(metric, fdo.getExclusive(i));
		    if((globalMappingElement.getMaxExclusivePercentValue(metric)) < fdo.getExclusivePercentage(i))
			globalMappingElement.setMaxExclusivePercentValue(metric, fdo.getExclusivePercentage(i));
		    if((globalMappingElement.getMaxInclusiveValue(metric)) < fdo.getInclusive(i))
			globalMappingElement.setMaxInclusiveValue(metric, fdo.getInclusive(i));
		    if((globalMappingElement.getMaxInclusivePercentValue(metric)) < fdo.getInclusivePercentage(i))
			globalMappingElement.setMaxInclusivePercentValue(metric, fdo.getInclusivePercentage(i));
		    if(globalMappingElement.getMaxNumberOfCalls() < fdo.getNumCalls())
			globalMappingElement.setMaxNumberOfCalls(fdo.getNumCalls());
		    if(globalMappingElement.getMaxNumberOfSubRoutines() < fdo.getNumSubroutines())
			globalMappingElement.setMaxNumberOfSubRoutines(fdo.getNumSubroutines());
		    if(globalMappingElement.getMaxUserSecPerCall(metric) < fdo.getInclusivePerCall(i))
			globalMappingElement.setMaxUserSecPerCall(metric, fdo.getInclusivePerCall(i));

		    if((thread.getMaxExclusiveValue(metric)) < fdo.getExclusive(i))
			thread.setMaxExclusiveValue(metric, fdo.getExclusive(i));
		    if((thread.getMaxExclusivePercentValue(metric)) < fdo.getExclusivePercentage(i))
			thread.setMaxExclusivePercentValue(metric, fdo.getExclusivePercentage(i));
		    if((thread.getMaxInclusiveValue(metric)) < fdo.getInclusive(i))
			thread.setMaxInclusiveValue(metric, fdo.getInclusive(i));
		    if((thread.getMaxInclusivePercentValue(metric)) < fdo.getInclusivePercentage(i))
			thread.setMaxInclusivePercentValue(metric, fdo.getInclusivePercentage(i));
		    if(thread.getMaxNumberOfCalls() < fdo.getNumCalls())
			thread.setMaxNumberOfCalls(fdo.getNumCalls());
		    if(thread.getMaxNumberOfSubRoutines() < fdo.getNumSubroutines())
			thread.setMaxNumberOfSubRoutines(fdo.getNumSubroutines());
		    if(thread.getMaxUserSecPerCall(metric) < fdo.getInclusivePerCall(i))
			thread.setMaxUserSecPerCall(metric, fdo.getInclusivePerCall(i));
		}
	    }

	    while(l.hasNext()){
		UserEvent ue = (UserEvent) l.next();
		System.out.println(ue.getName());
		this.getGlobalMapping().addGlobalMapping(ue.getName(), 2, 1);
	    }

	    l = perfDBSession.getUserEventData();
	    while(l.hasNext()){
		l.next();
	    }

	    time = (System.currentTimeMillis()) - time;
	    System.out.println("Done processing data file!");
	    System.out.println("Time to process file (in milliseconds): " + time);
	}
        catch(Exception e){
	    ParaProf.systemError(e, null, "SSD01");
	}
    }
    
    //####################################
    //Instance data.
    //####################################
    private LineData functionDataLine = new LineData();
    private LineData  usereventDataLine = new LineData();
    //####################################
    //End - Instance data.
    //####################################
}

/*class FunIndexFunIDPair implements Comparable{
    public FunIndexFunIDPair(int functionIndex, int paraProfId){
	this.functionIndex = functionIndex;
	this.paraProfId = paraProfId;
    }

    public int compareTo(Object obj){
	return functionIndex - ((FunIndexFunIDPair)obj).functionIndex;}

    public int functionIndex;
    public int paraProfId;
    }*/
