/* 
   Name:        ParaProfDBSession.java
   Author:      Robert Bell
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

package dms.dss;



import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

import java.io.*;
import java.util.*;

public class ParaProfDBSession extends ParaProfDataSession{

    public ParaProfDBSession(){
	super();
	this.setMetrics(new Vector());
    }

    public void run(){
	try{
	    //######
	    //Frequently used items.
	    //######
	    PerfDBSession perfDBSession = (PerfDBSession) initializeObject;

	    GlobalMappingElement globalMappingElement = null;
	    GlobalThreadDataElement globalThreadDataElement = null;
	    
	    Node node = null;
	    Context context = null;
	    dms.dss.Thread thread = null;
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
	    
	    int numberOfMetrics = this.getNumberOfMetrics();
		System.out.println("Found " + numberOfMetrics + " metrics.");
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
			FunctionDataObject fdo = f.getMeanSummary();
		    for(int i=0;i<numberOfMetrics;i++){
			globalMappingElement.setMeanExclusiveValue(i, fdo.getExclusive(i));
			globalMappingElement.setMeanExclusivePercentValue(i, fdo.getExclusivePercentage(i));
			globalMappingElement.setMeanInclusiveValue(i, fdo.getInclusive(i));
			globalMappingElement.setMeanInclusivePercentValue(i, fdo.getInclusivePercentage(i));

			if((this.getGlobalMapping().getMaxMeanExclusiveValue(i)) < fdo.getExclusive(i)){
			    this.getGlobalMapping().setMaxMeanExclusiveValue(i, fdo.getExclusive(i));}
			if((this.getGlobalMapping().getMaxMeanExclusivePercentValue(i)) < fdo.getExclusivePercentage(i)){
			    this.getGlobalMapping().setMaxMeanExclusivePercentValue(i, fdo.getExclusivePercentage(i));}

			if((this.getGlobalMapping().getMaxMeanInclusiveValue(i)) < fdo.getInclusive(i)){
			    this.getGlobalMapping().setMaxMeanInclusiveValue(i, fdo.getInclusive(i));}
			if((this.getGlobalMapping().getMaxMeanInclusivePercentValue(i)) < fdo.getInclusivePercentage(i)){
			    this.getGlobalMapping().setMaxMeanInclusivePercentValue(i, fdo.getInclusivePercentage(i));}
		    }
		    globalMappingElement.setMeanValuesSet(true);
	    }
	    
	    //Collections.sort(localMap);


	    System.out.println("About to increase storage.");

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
		    thread = context.addThread(fdo.getThread(), numberOfMetrics);
		    thread.setDebug(this.debug());
		    thread.initializeFunctionList(this.getGlobalMapping().getNumberOfMappings(0));
		}
		
		//Get GlobalMappingElement and GlobalThreadDataElement.
		
		//Obtain the mapping id from the local map.
		//int pos = Collections.binarySearch(localMap, new FunIndexFunIDPair(fdo.getFunctionIndexID(),0));
		//mappingID = ((FunIndexFunIDPair)localMap.elementAt(pos)).paraProfId;
		
		mappingID = this.getGlobalMapping().getMappingID(perfDBSession.getFunction(fdo.getFunctionIndexID()).getName(),0);
		globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 0);
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
		    if((globalMappingElement.getMaxExclusiveValue(i)) < fdo.getExclusive(i))
			globalMappingElement.setMaxExclusiveValue(i, fdo.getExclusive(i));
		    if((globalMappingElement.getMaxExclusivePercentValue(i)) < fdo.getExclusivePercentage(i))
			globalMappingElement.setMaxExclusivePercentValue(i, fdo.getExclusivePercentage(i));
		    if((globalMappingElement.getMaxInclusiveValue(i)) < fdo.getInclusive(i))
			globalMappingElement.setMaxInclusiveValue(i, fdo.getInclusive(i));
		    if((globalMappingElement.getMaxInclusivePercentValue(i)) < fdo.getInclusivePercentage(i))
			globalMappingElement.setMaxInclusivePercentValue(i, fdo.getInclusivePercentage(i));
		    if(globalMappingElement.getMaxNumberOfCalls() < fdo.getNumCalls())
			globalMappingElement.setMaxNumberOfCalls(fdo.getNumCalls());
		    if(globalMappingElement.getMaxNumberOfSubRoutines() < fdo.getNumSubroutines())
			globalMappingElement.setMaxNumberOfSubRoutines(fdo.getNumSubroutines());
		    if(globalMappingElement.getMaxUserSecPerCall(i) < fdo.getInclusivePerCall(i))
			globalMappingElement.setMaxUserSecPerCall(i, fdo.getInclusivePerCall(i));

		    if((thread.getMaxExclusiveValue(i)) < fdo.getExclusive(i))
			thread.setMaxExclusiveValue(i, fdo.getExclusive(i));
		    if((thread.getMaxExclusivePercentValue(i)) < fdo.getExclusivePercentage(i))
			thread.setMaxExclusivePercentValue(i, fdo.getExclusivePercentage(i));
		    if((thread.getMaxInclusiveValue(i)) < fdo.getInclusive(i))
			thread.setMaxInclusiveValue(i, fdo.getInclusive(i));
		    if((thread.getMaxInclusivePercentValue(i)) < fdo.getInclusivePercentage(i))
			thread.setMaxInclusivePercentValue(i, fdo.getInclusivePercentage(i));
		    if(thread.getMaxNumberOfCalls() < fdo.getNumCalls())
			thread.setMaxNumberOfCalls(fdo.getNumCalls());
		    if(thread.getMaxNumberOfSubRoutines() < fdo.getNumSubroutines())
			thread.setMaxNumberOfSubRoutines(fdo.getNumSubroutines());
		    if(thread.getMaxUserSecPerCall(i) < fdo.getInclusivePerCall(i))
			thread.setMaxUserSecPerCall(i, fdo.getInclusivePerCall(i));
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
	    
	    //Need to notify observers that we are done.  Be careful here.
	    //It is likely that they will modify swing elements.  Make sure
	    //to dump request onto the event dispatch thread to ensure
	    //safe update of said swing elements.  Remember, swing is not thread
	    //safe for the most part.
	    EventQueue.invokeLater(new Runnable(){
		    public void run(){
			ParaProfDBSession.this.notifyObservers();
		    }
		});
				   
	}
        catch(Exception e){
	    UtilFncs.systemError(e, null, "SSD01");
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
