/*
  ParaProfTrial.java
  
  
  Title:      ParaProf
  Author:     Robert Bell
  Description: The manner in which this class behaves is slightly different from its parent.
               It behaves more as a container for its DataSession, than a setting for it. So,
	       in a sense the roll is almost reverse (but not quite). This is a result of
	       the fact that ParaProf must maintain the majority of its data itself, and as
	       such, ParaProfTrial serves as the reference through which data is accessed.
*/

package paraprof;

import java.io.*;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.*;
import dms.dss.*;

public class ParaProfTrial extends Trial{
    public ParaProfTrial(){
	super();}

    public void initialize(Object obj){
	dataSession = new TauPprofOutputSession();
	dataSession.initialize(obj);
    }

    public void setExperiment(ParaProfExperiment experiment){
	this.experiment = experiment;}
  
    public ParaProfExperiment getExperiment(){
	return experiment;}
  
    public void setDMTN(DefaultMutableTreeNode defaultMutableTreeNode){
	this.defaultMutableTreeNode = defaultMutableTreeNode;}
  
    public DefaultMutableTreeNode getDMTN(){
	return defaultMutableTreeNode;}
  
    public String getIDString(){
	if(experiment!=null)
	    return (experiment.getIDString()) + ":" + (super.getID());
	else
	    return  ":" + (super.getID());
    }
  
    public Metric addMetric(){
	Metric metric = new Metric();
	metric.setTrial(this);
	metric.setID((metrics.size()));
	metrics.add(metric);
	return metric;
    }
  
    public ColorChooser getColorChooser(){
	return clrChooser;}
  
    public Preferences getPreferences(){
	return preferences;}
  
    public void setProfilePathName(String profilePathName){
	this.profilePathName = profilePathName;}
  
    public void setProfilePathNameReverse(String profilePathNameReverse){
	this.profilePathNameReverse = profilePathNameReverse;}
  
    public String getProfilePathName(){
	return profilePathName;}
  
    public String getProfilePathNameReverse(){
	return profilePathNameReverse;}
  
    public Vector getMetrics(){
	return metrics;}
  
    public int getMetricPosition(String string){
	int counter = 0;
	for(Enumeration e = metrics.elements(); e.hasMoreElements() ;){
	    Metric metric = (Metric) e.nextElement();
	    if((metric.getName()).equals(string))
		return counter;
	    counter++;
	}
	return -1;
    }

    public String toString(){ 
	return super.getName();}
  
    public int getNumberOfNodes(){
	return nodes.size();}

    public Vector getNodes(){
	return nodes;}

    public Node getNode(int nodeID){
	return (Node) nodes.elementAt(nodeID);}

    //Returns the total number of contexts in this trial.
    public int getTotalNumberOfContexts(){
	if(totalNumberOfContexts==-1){
	    for(Enumeration e = this.getNodes().elements(); e.hasMoreElements() ;){
	     Node node = (Node) e.nextElement();
	     totalNumberOfContexts+=(node.getNumberOfContexts());
	    }
	}
	return totalNumberOfContexts;
    }

    //Returns the number of contexts on the specified node.
    public int getNumberOfContexts(int nodeID){
	return ((Node) nodes.elementAt(nodeID)).getNumberOfContexts();}

    public Vector getContexts(int nodeID){
	return (this.getNode(nodeID)).getContexts();}

    public Context getContext(int nodeID, int contextID){
	return (this.getNode(nodeID)).getContext(contextID);}

    //Returns the total number of threads in this trial.
    public int getTotalNumberOfThreads(){
	if(totalNumberOfThreads==-1){
	    for(Enumeration e1 = this.getNodes().elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    totalNumberOfThreads+=(context.getNumberOfThreads());
		}
	    }
	}
	return totalNumberOfThreads;
    }

    //Returns the number of threads on the specified node,context.
    public int getNumberOfThreads(int nodeID, int contextID){
	return (this.getContext(nodeID,contextID)).getNumberOfThreads();}

    public Vector getThreads(int nodeID, int contextID){
	return (this.getContext(nodeID,contextID)).getThreads();}

    public Thread getThread(int nodeID, int contextID, int threadID){
	return (this.getContext(nodeID,contextID)).getThread(threadID);}

    //Returns an array of length 3 which contains the maximum number
    //of nodes, contexts and threads reached. This is not a total,
    //except in the case of the number of nodes.
    //This method is useful for determining an upper bound on the
    //string sizes being displayed in ParaProf's different windows.
    //Computation will only occur during the first call to this method.
    //Subsequent calls will return the results obtained from the first
    //call.
    public int[] getMaxNCTNumbers(){
	if(nct==null){
	    nct = new int[3];
	    for(int i=0;i<3;i++){
		nct[i]=0;}
	    nct[0] = this.getNumberOfNodes();
	    for(Enumeration e1 = (this.getNodes()).elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		if(node.getNumberOfContexts()>nct[1])
		    nct[1]=node.getNumberOfContexts();
		for(Enumeration e2 = (node.getContexts()).elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    if(context.getNumberOfThreads()>nct[2])
			nct[2]=context.getNumberOfThreads();
		}
	    }
	}
	return nct;
    }
    

    //####################################
    //Functions that control the obtaining and the openning
    //and closing of the static main window for
    //this trial.
    //####################################
    public StaticMainWindow getStaticMainWindow(){
	return sMW;}
  
    public void showMainWindow(){
  
	if(sMW == null){
	    sMW = new StaticMainWindow(this);
	    sMW.setVisible(true);
	    this.getSystemEvents().addObserver(sMW);
	}
	else{
	    this.getSystemEvents().addObserver(sMW);
	    sMW.show();
	}
    }
  
    public void closeStaticMainWindow(){
	if(sMW != null){
	    this.getSystemEvents().deleteObserver(sMW);
	    sMW.setVisible(false);
	}
    }
    
    //####################################
    //Functions that control the opening and closing of the static main window for
    //this trial.
    //####################################
  
    public SystemEvents getSystemEvents(){
	return systemEvents;}
  
    public void setCurValLoc(int inValueLocation){
	currentValueLocation = inValueLocation;}
  
    public int getCurValLoc(){
	return currentValueLocation;}
  
  
    public void addDefaultToVectors(){
	maxMeanInclusiveValueList.add(new Double(0));
	maxMeanExclusiveValueList.add(new Double(0));
	maxMeanInclusivePercentValueList.add(new Double(0));
	maxMeanExclusivePercentValueList.add(new Double(0));
	maxMeanUserSecPerCallList.add(new Double(0));
    }
  
    //The following funtion initializes the GlobalMapping object.
    //Since we are in the static mode, the number of mappings is known,
    //therefore, the appropriate number of GlobalMappingElements are created.
    void initializeGlobalMapping(int inNumberOfMappings, int mappingSelection){
	for(int i=0; i<inNumberOfMappings; i++){
	    //globalMapping.addGlobalMapping("Error ... the mapping name has not been set!");
	    globalMapping.addGlobalMapping(null, mappingSelection);
	}
    }
  
    GlobalMapping getGlobalMapping(){
	return globalMapping;}
  
    public String getCounterName(){
	Metric metric = (Metric) metrics.elementAt(currentValueLocation);
	return metric.getName();
    }

    public boolean isTimeMetric(){
	String trialName = this.getCounterName();
	trialName = trialName.toUpperCase();
	if(trialName.indexOf("TIME") == -1)
	    return false;
	else
	    return true;
    }
  
    //####################################
    //Useful functions to help the drawing windows.
    //
    //For the most part, these functions just return data
    //items that are easier to calculate whilst building the global
    //lists
    //####################################
    private void setNumberOfMappings(int numberOfMappings){
	this.numberOfMappings = numberOfMappings;}

    public int getNumberOfMappings(){
	return numberOfMappings;}

    private void setNumberOfUserEvents(int numberOfUserEvents){
	this.numberOfUserEvents = numberOfUserEvents;}

    public int getNumberOfUserEvents(){
	return numberOfUserEvents;}

    private void setGroupNamesPresent(boolean groupNamesPresent){
	this.groupNamesPresent = groupNamesPresent;}
    
    public boolean groupNamesPresent(){
	return groupNamesPresent;}
  
    private void setUserEventsPresent(boolean userEventsPresent){
	this.userEventsPresent = userEventsPresent;}
  
    public boolean userEventsPresent(){
	return userEventsPresent;}

    private void setCallPathDataPresent(boolean callPathDataPresent){
	this.callPathDataPresent = callPathDataPresent;}
  
    public boolean callPathDataPresent(){
	return callPathDataPresent;}

    public void setMaxMeanInclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanInclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    public double getMaxMeanInclusiveValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanInclusiveValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
  
    public void setMaxMeanExclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanExclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    public double getMaxMeanExclusiveValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanExclusiveValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
  
    public void setMaxMeanInclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanInclusivePercentValueList.add(dataValueLocation, tmpDouble);}
  
    public double getMaxMeanInclusivePercentValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanInclusivePercentValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
  
    public void setMaxMeanExclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanExclusivePercentValueList.add(dataValueLocation, tmpDouble);}
  
    public double getMaxMeanExclusivePercentValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanExclusivePercentValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
  
    public void setMaxMeanNumberOfCalls(double inDouble){
	maxMeanNumberOfCalls = inDouble;}
  
    public double getMaxMeanNumberOfCalls(){
	return maxMeanNumberOfCalls;}
  
    public void setMaxMeanNumberOfSubRoutines(double inDouble){
	maxMeanNumberOfSubRoutines = inDouble;}
  
    public double getMaxMeanNumberOfSubRoutines(){
	return maxMeanNumberOfSubRoutines;}
  
    public void setMaxMeanUserSecPerCall(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanUserSecPerCallList.add(dataValueLocation, tmpDouble);}
  
    public double getMaxMeanUserSecPerCall(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanUserSecPerCallList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
    //####################################
    //End - Useful functions to help the drawing windows.
    //####################################
  
    //####################################
    //Instance data.
    //####################################
    TauPprofOutputSession dataSession = null;
    ParaProfExperiment experiment = null;
    DefaultMutableTreeNode defaultMutableTreeNode = null;
    private Vector metrics = new Vector();
  
    private SystemEvents systemEvents = new SystemEvents();
    private StaticMainWindow sMW = null;
    private ColorChooser clrChooser = new ColorChooser(this, null);
    private Preferences  preferences = new  Preferences(this, null);
  
    private String profilePathName = null;
    private String profilePathNameReverse = null;
    private int currentValueLocation = 0;
  
    private GlobalMapping globalMapping;
    private Vector nodes = new Vector();
    boolean groupNamesPresent = false;
    boolean userEventsPresent = false;
    boolean callPathDataPresent = false;

    private int numberOfMappings = -1;
    private int numberOfUserEvents = -1;
    private int totalNumberOfContexts = -1;
    private int totalNumberOfThreads = -1;
    private int[] nct = null;
  
    //Max mean values.
    private Vector maxMeanInclusiveValueList = new Vector();
    private Vector maxMeanExclusiveValueList = new Vector();
    private Vector maxMeanInclusivePercentValueList = new Vector();
    private Vector maxMeanExclusivePercentValueList = new Vector();
    private double maxMeanNumberOfCalls = 0;
    private double maxMeanNumberOfSubRoutines = 0;
    private Vector maxMeanUserSecPerCallList = new Vector();
    //####################################
    //Instance data.
    //####################################
}
