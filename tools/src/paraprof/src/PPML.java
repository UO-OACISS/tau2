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

    public static Value applyOperation(Trial trial, String tmpString1, String tmpString2, String inOperation){
	  
	int opA = trial.getValuePosition(tmpString1);
	int opB = trial.getValuePosition(tmpString2);

	String tmpString3 = null;
    
	int operation = -1;
    
	if(inOperation.equals("Add")){
	    operation = 0;
	    tmpString3 = tmpString1 + " + " + tmpString2;
	}
	else if(inOperation.equals("Subtract")){
	    operation = 1;
	    tmpString3 = tmpString1 + " - " + tmpString2;
	}
	else if(inOperation.equals("Multiply")){
	    operation = 2;
	    tmpString3 = tmpString1 + " * " + tmpString2;
	}
	else if(inOperation.equals("Divide")){
	    operation = 3;
	    tmpString3 = tmpString1 + " / " + tmpString2;
	}
	else{
	    System.out.println("Wrong operation type");
	}
      
	Value newValue = trial.addValue();
	newValue.setValueName(tmpString3);
	int currentValueLocation = newValue.getValueID();
	trial.setCurValLoc(currentValueLocation);

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

	trial.addDefaultToVectors();
    
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
     
	for(Enumeration e1 = trial.getNodes().elements(); e1.hasMoreElements() ;){
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
			    
			    globalThreadDataElement.setExclusiveValue(currentValueLocation, result);
			    //Now do the global mapping element exclusive stuff.
			    if((globalMappingElement.getMaxExclusiveValue(currentValueLocation)) < result)
				globalMappingElement.setMaxExclusiveValue(currentValueLocation, result);
			    globalMappingElement.incrementTotalExclusiveValue(result);
                  
			    d1 = globalThreadDataElement.getInclusiveValue(opA);
			    d2 = globalThreadDataElement.getInclusiveValue(opB);
			    result = PPML.apply(operation,d1,d2);
			    
			    globalThreadDataElement.setInclusiveValue(currentValueLocation, result);			    
			    //Now do the global mapping element inclusive stuff.
			    if((globalMappingElement.getMaxInclusiveValue(currentValueLocation)) < result)
				globalMappingElement.setMaxInclusiveValue(currentValueLocation, result);
			    globalMappingElement.incrementTotalInclusiveValue(result);
			}
		    }

		    //The thread object takes care of computing maximums and totals for a given metric.
		    thread.setThreadSummaryData(currentValueLocation);
		    
		    //######
		    //Compute percent values.
		    //######
		    l = thread.getFunctionListIterator();
		    while(l.hasNext()){
			GlobalThreadDataElement globalThreadDataElement = (GlobalThreadDataElement) l.next();
			double exclusiveTotal = thread.getTotalExclusiveValue(currentValueLocation);
			double inclusiveMax = thread.getMaxInclusiveValue(currentValueLocation);
			
			if(globalThreadDataElement != null){
			    GlobalMappingElement globalMappingElement =
				trial.getGlobalMapping().getGlobalMappingElement(globalThreadDataElement.getMappingID(), 0);
			    
			    double d1 = globalThreadDataElement.getExclusiveValue(currentValueLocation);
			    double d2 = globalThreadDataElement.getInclusiveValue(currentValueLocation);
			    
			    if(exclusiveTotal!=0){
				double result = (d1/exclusiveTotal)*100.00;
				globalThreadDataElement.setExclusivePercentValue(currentValueLocation, result);
				//Now do the global mapping element exclusive stuff.
				if((globalMappingElement.getMaxExclusivePercentValue(currentValueLocation)) < result)
				    globalMappingElement.setMaxExclusivePercentValue(currentValueLocation, result);
			    }

			    if(inclusiveMax!=0){
				double result = (d2/inclusiveMax) * 100;
				globalThreadDataElement.setInclusivePercentValue(currentValueLocation, result);
				//Now do the global mapping element exclusive stuff.
				if((globalMappingElement.getMaxInclusivePercentValue(currentValueLocation)) < result)
				    globalMappingElement.setMaxInclusivePercentValue(currentValueLocation, result);
			    }
			}
		    }
		    //######
		    //End - Compute percent values.
		    //######

		    //Call the setThreadSummaryData function again on this thread so that
		    //it can fill in all the summary data.
		    thread.setThreadSummaryData(currentValueLocation);
		}
	    }
	}
	
	l = trial.getGlobalMapping().getMappingIterator(0);
	while(l.hasNext()){
	    GlobalMappingElement globalMappingElement = (GlobalMappingElement) l.next();
	    if((globalMappingElement.getCounter()) != 0){
		double d = (globalMappingElement.getTotalExclusiveValue())/(globalMappingElement.getCounter());
		//Increment the total values.
		trial.setTotalMeanExclusiveValue(trial.getCurValLoc(),(trial.getTotalMeanExclusiveValue(trial.getCurValLoc())) + d);
		globalMappingElement.setMeanExclusiveValue(trial.getCurValLoc(), d);
		if((trial.getMaxMeanExclusiveValue(trial.getCurValLoc()) < d))
		    trial.setMaxMeanExclusiveValue(trial.getCurValLoc(), d);
		
		d = (globalMappingElement.getTotalInclusiveValue())/(globalMappingElement.getCounter());
		//Increment the total values.
		trial.setTotalMeanInclusiveValue(trial.getCurValLoc(),(trial.getTotalMeanInclusiveValue(trial.getCurValLoc())) + d);
		globalMappingElement.setMeanInclusiveValue(trial.getCurValLoc(), d);
		if((trial.getMaxMeanInclusiveValue(trial.getCurValLoc()) < d))
		    trial.setMaxMeanInclusiveValue(trial.getCurValLoc(), d);
	    }
	}
		
	double exclusiveTotal = trial.getTotalMeanExclusiveValue(trial.getCurValLoc());
	double inclusiveMax = trial.getMaxMeanInclusiveValue(trial.getCurValLoc());

	l = trial.getGlobalMapping().getMappingIterator(0);
	while(l.hasNext()){
	    GlobalMappingElement globalMappingElement = (GlobalMappingElement) l.next();
	    
	    if(exclusiveTotal!=0){
		double tmpDouble = ((globalMappingElement.getMeanExclusiveValue(trial.getCurValLoc()))/exclusiveTotal) * 100;
		globalMappingElement.setMeanExclusivePercentValue(currentValueLocation, tmpDouble);
		if((trial.getMaxMeanExclusivePercentValue(trial.getCurValLoc()) < tmpDouble))
		    trial.setMaxMeanExclusivePercentValue(trial.getCurValLoc(), tmpDouble);
	    }
      
	    if(inclusiveMax!=0){
		double tmpDouble = ((globalMappingElement.getMeanInclusiveValue(trial.getCurValLoc()))/inclusiveMax) * 100;
		globalMappingElement.setMeanInclusivePercentValue(trial.getCurValLoc(), tmpDouble);
		if((trial.getMaxMeanInclusivePercentValue(trial.getCurValLoc()) < tmpDouble))
		    trial.setMaxMeanInclusivePercentValue(trial.getCurValLoc(), tmpDouble);
	    }
	    
	}
	return newValue;
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
	    ParaProf.systemError(null, null, "Unexpected opertion - PPML01 value: " + op);
	}
	return d;
    }
}
