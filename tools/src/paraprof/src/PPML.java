/*
  PPML.java
  
  
  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package paraprof;

import java.util.*;

public class PPML{

    public PPML(){}

    public static Metric applyOperation(String tmpString1, String tmpString2, String inOperation){

	
	StringTokenizer st = new StringTokenizer(tmpString1, ":");
	int applicationID = Integer.parseInt(st.nextToken());
	int experimentID = Integer.parseInt(st.nextToken());
	int trialID = Integer.parseInt(st.nextToken());
	
	ParaProfTrial trial = ParaProf.applicationManager.getTrial(applicationID, experimentID, trialID);

	int opA = Integer.parseInt(st.nextToken());
	st = new StringTokenizer(tmpString2, ":");
	st.nextToken();
	st.nextToken();
	st.nextToken();
	int opB = Integer.parseInt(st.nextToken());
	
	String tmpString3 = null;
    
	int operation = -1;
    
	if(inOperation.equals("Add")){
	    operation = 0;
	    tmpString3 = " + ";
	}
	else if(inOperation.equals("Subtract")){
	    operation = 1;
	    tmpString3 = " - ";
	}
	else if(inOperation.equals("Multiply")){
	    operation = 2;
	    tmpString3 = " * ";
	}
	else if(inOperation.equals("Divide")){
	    operation = 3;
	    tmpString3 = " / ";
	}
	else{
	    System.out.println("Wrong operation type");
	}

	tmpString3 = ((Metric)trial.getMetrics().elementAt(opA)).getName() + tmpString3 + ((Metric)trial.getMetrics().elementAt(opA)).getName();
      
	Metric newMetric = trial.addMetric();
	newMetric.setName(tmpString3);
	int metric = newMetric.getID();
	trial.setSelectedMetricID(metric);

	ListIterator l = trial.getGlobalMapping().getMappingIterator(0);
	while(l.hasNext()){
	    GlobalMappingElement globalMappingElement = (GlobalMappingElement) l.next();
	    globalMappingElement.incrementStorage();
	    globalMappingElement.setTotalExclusiveValue(0);
	    globalMappingElement.setTotalInclusiveValue(0);
	    globalMappingElement.setCounter(0);
	}
	l = trial.getGlobalMapping().getMappingIterator(2);
	while(l.hasNext()){
	    GlobalMappingElement globalMappingElement = (GlobalMappingElement) l.next();
	    globalMappingElement.incrementStorage();
	}

	trial.getGlobalMapping().increaseVectorStorage();
    
	//######
	//Calculate the raw values.
	//We only need establish exclusive and inclusive time.
	//The rest of the data can either be computed from these,
	//or is already in the system (number of calls as an example
	//of the latter.
	//######

	Node node;
	Context context;
	Thread thread;
     
	for(Enumeration e1 = trial.getNCT().getNodes().elements(); e1.hasMoreElements() ;){
	    node = (Node) e1.nextElement();
 	    for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		context = (Context) e2.nextElement();
 		for(Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements() ;){
		    thread = (Thread) e3.nextElement();
		    thread.incrementStorage();
		    l = thread.getFunctionListIterator();
		    while(l.hasNext()){
			GlobalThreadDataElement globalThreadDataElement = (GlobalThreadDataElement) l.next();
			if(globalThreadDataElement != null){
			    GlobalMappingElement globalMappingElement =
				trial.getGlobalMapping().getGlobalMappingElement(globalThreadDataElement.getMappingID(), 0);
			    globalMappingElement.incrementCounter();
			    globalThreadDataElement.incrementStorage();
              
			    double d1=0.0;
			    double d2=0.0;
			    double result = 0.0;
                            
			    d1 = globalThreadDataElement.getExclusiveValue(opA);
			    d2 = globalThreadDataElement.getExclusiveValue(opB);
			    result = PPML.apply(operation,d1,d2);
			    
			    globalThreadDataElement.setExclusiveValue(metric, result);
			    //Now do the global mapping element exclusive stuff.
			    if((globalMappingElement.getMaxExclusiveValue(metric)) < result)
				globalMappingElement.setMaxExclusiveValue(metric, result);
			    globalMappingElement.incrementTotalExclusiveValue(result);
                  
			    d1 = globalThreadDataElement.getInclusiveValue(opA);
			    d2 = globalThreadDataElement.getInclusiveValue(opB);
			    result = PPML.apply(operation,d1,d2);
			    
			    globalThreadDataElement.setInclusiveValue(metric, result);			    
			    //Now do the global mapping element inclusive stuff.
			    if((globalMappingElement.getMaxInclusiveValue(metric)) < result)
				globalMappingElement.setMaxInclusiveValue(metric, result);
			    globalMappingElement.incrementTotalInclusiveValue(result);
			}
		    }
		    thread.setThreadData(metric);
		}
	    }
	}
	//Done with this metric, let the global mapping compute the mean values.
	trial.setMeanData(0,metric);
	return newMetric;
    }

    public static double apply(int op, double arg1, double arg2){
	double d = 0.0;
	switch(op){
	case(0):
	    d = arg1+arg2;
	    break;
	case(1):
	    if(arg1>arg2){
		d =  arg1-arg2;
	    }
	    break;
	case(2):
	    d = arg1*arg2;
	    break;
	case(3):
	    if(arg2!=0){
		return arg1/arg2;
	    }
	    break;
	default:
	    UtilFncs.systemError(null, null, "Unexpected opertion - PPML01 value: " + op);
	}
	return d;
    }
}
