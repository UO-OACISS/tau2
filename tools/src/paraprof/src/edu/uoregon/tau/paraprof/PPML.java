/*
  PPML.java
  
  
  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import edu.uoregon.tau.dms.dss.*;

public class PPML{

    public PPML(){}

    public static Metric applyOperation(String tmpString1, String tmpString2, String inOperation){
	try{
	    StringTokenizer st = new StringTokenizer(tmpString1, ":");
	    int applicationID = Integer.parseInt(st.nextToken());
	    int experimentID = Integer.parseInt(st.nextToken());
	    int trialID = Integer.parseInt(st.nextToken());
	    int metricID = Integer.parseInt(st.nextToken());
	    ParaProfTrial trialOpA = ParaProf.applicationManager.getTrial(applicationID, experimentID, trialID);
	    int opA = metricID;
	    
	    st = new StringTokenizer(tmpString2, ":");
	    applicationID = Integer.parseInt(st.nextToken());
	    experimentID = Integer.parseInt(st.nextToken());
	    trialID = Integer.parseInt(st.nextToken());
	    metricID = Integer.parseInt(st.nextToken());
	    ParaProfTrial trialOpB = ParaProf.applicationManager.getTrial(applicationID, experimentID, trialID);
	    int opB = metricID;
	

	    //We do not support metric from different trials yet.  Check for this.
	    if(trialOpA!=trialOpB){
		JOptionPane.showMessageDialog(ParaProf.paraProfManager,
					      "Sorry, please select metrics from the same trial!", "ParaProf Error",
					      JOptionPane.ERROR_MESSAGE);
		return null;
	    }

	    String newMetricName = null;
	    int operation = -1;
	    if(inOperation.equals("Add")){
		operation = 0;
		newMetricName = " + ";
	    }
	    else if(inOperation.equals("Subtract")){
		operation = 1;
		newMetricName = " - ";
	    }
	    else if(inOperation.equals("Multiply")){
		operation = 2;
		newMetricName = " * ";
	    }
	    else if(inOperation.equals("Divide")){
		operation = 3;
		newMetricName = " / ";
	    }
	    else{
		System.out.println("Wrong operation type");
	    }

	    newMetricName = ((Metric)trialOpA.getMetrics().elementAt(opA)).getName() + newMetricName + ((Metric)trialOpA.getMetrics().elementAt(opB)).getName();
      
	    Metric newMetric = trialOpA.addMetric();
	    newMetric.setTrial(trialOpA);
	    newMetric.setName(newMetricName);
	    newMetric.setDerivedMetric(true);
	    int metric = newMetric.getID();
	    trialOpA.setSelectedMetricID(metric);

	    ListIterator l = trialOpA.getGlobalMapping().getMappingIterator(0);
	    while(l.hasNext()){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) l.next();
		globalMappingElement.incrementStorage();}
	    l = trialOpA.getGlobalMapping().getMappingIterator(2);
	    while(l.hasNext()){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) l.next();
		globalMappingElement.incrementStorage();}

	    trialOpA.getGlobalMapping().increaseVectorStorage();
    
	    //######
	    //Calculate the raw values.
	    //We only need establish exclusive and inclusive time.
	    //The rest of the data can either be computed from these,
	    //or is already in the system (number of calls as an example
	    //of the latter.
	    //######

	    Node node;
	    Context context;
	    edu.uoregon.tau.dms.dss.Thread thread;
     
	    for(Enumeration e1 = trialOpA.getNCT().getNodes().elements(); e1.hasMoreElements() ;){
		node = (Node) e1.nextElement();
		for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		    context = (Context) e2.nextElement();
		    for(Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements() ;){
			thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
			thread.incrementStorage();
			l = thread.getFunctionListIterator();
			while(l.hasNext()){
			    GlobalThreadDataElement globalThreadDataElement = (GlobalThreadDataElement) l.next();
			    if(globalThreadDataElement != null){
				GlobalMappingElement globalMappingElement =
				    trialOpA.getGlobalMapping().getGlobalMappingElement(globalThreadDataElement.getMappingID(), 0);
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
                  
				d1 = globalThreadDataElement.getInclusiveValue(opA);
				d2 = globalThreadDataElement.getInclusiveValue(opB);
				result = PPML.apply(operation,d1,d2);
			    
				globalThreadDataElement.setInclusiveValue(metric, result);			    
				//Now do the global mapping element inclusive stuff.
				if((globalMappingElement.getMaxInclusiveValue(metric)) < result)
				    globalMappingElement.setMaxInclusiveValue(metric, result);
			    }
			}
			thread.setThreadData(metric);
		    }
		}
	    }
	    //Done with this metric, let the global mapping compute the mean values.
	    trialOpA.setMeanData(0,metric);
	    return newMetric;
	}
	catch(Exception e){
	    if(e instanceof NumberFormatException){
		//Display an error
		JOptionPane.showMessageDialog(ParaProf.paraProfManager, "Did not recognize arguments! Note: DB apply not supported.", "Argument Error!"
					      ,JOptionPane.ERROR_MESSAGE);
	    }
	    return null;
	}
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
