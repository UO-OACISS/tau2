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
    public ParaProfTrial(int type){
	super();
	this.type = type;
    }

    public void initialize(Object obj){
	switch(type){
	case 0:
	    dataSession = new TauPprofOutputSession();
	    dataSession.setDebug(ParaProf.debugIsOn);
	    break;
	case 1:
	    dataSession = new TauOutputSession();
	    dataSession.setDebug(ParaProf.debugIsOn);
	    break;
	case 2:
	    dataSession = new DynaprofOutputSession();
	    dataSession.setDebug(ParaProf.debugIsOn);
	    break;
	case 3:
	    break;
	default:
	    break;
	}

	dataSession.initialize(obj);

	//Now grab all the data.
	metrics = dataSession.getMetrics();
	globalMapping = dataSession.getGlobalMapping();
	nct = dataSession.getNCT();
	groupNamesPresent = dataSession.groupNamesPresent();
	userEventsPresent = dataSession.userEventsPresent();
	callPathDataPresent = dataSession.callPathDataPresent();

	numberOfMappings = dataSession.getNumberOfMappings();
	numberOfUserEvents = dataSession.getNumberOfUserEvents();
	
	globalMapping.setColors(clrChooser, -1);

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

    public NCT getNCT(){
	return nct;}
  
    //Returns the total number of threads in this trial.
    public int getTotalNumberOfThreads(){
	if(totalNumberOfThreads==-1){
	    for(Enumeration e1 = nct.getNodes().elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    totalNumberOfThreads+=(context.getNumberOfThreads());
		}
	    }
	}
	return totalNumberOfThreads;
    }

    //of nodes, contexts and threads reached. This is not a total,
    //except in the case of the number of nodes.
    //This method is useful for determining an upper bound on the
    //string sizes being displayed in ParaProf's different windows.
    //Computation will only occur during the first call to this method.
    //Subsequent calls will return the results obtained from the first
    //call.
    public int[] getMaxNCTNumbers(){
	if(maxNCT==null){
	    maxNCT = new int[3];
	    for(int i=0;i<3;i++){
		maxNCT[i]=0;}
	    maxNCT[0] = nct.getNumberOfNodes();
	    for(Enumeration e1 = (nct.getNodes()).elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		if(node.getNumberOfContexts()>maxNCT[1])
		    maxNCT[1]=node.getNumberOfContexts();
		for(Enumeration e2 = (node.getContexts()).elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    if(context.getNumberOfThreads()>maxNCT[2])
			maxNCT[2]=context.getNumberOfThreads();
		}
	    }
	}
	return maxNCT;
    }

    //####################################
    //Functions that control the obtaining and the opening
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
    //####################################
    //End - Useful functions to help the drawing windows.
    //####################################
  
    //####################################
    //Instance data.
    //####################################
    int type = -1;
    ParaProfDataSession dataSession = null;
    ParaProfExperiment experiment = null;
    DefaultMutableTreeNode defaultMutableTreeNode = null;
    private Vector metrics = null;
  
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
    private NCT nct = null;
    private int totalNumberOfContexts = -1;
    private int totalNumberOfThreads = -1;
    private int[] maxNCT = null;
    //####################################
    //Instance data.
    //####################################
}
