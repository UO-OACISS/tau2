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
    public ParaProfTrial(Trial trial, int type){
	super();
	if(trial!=null){
	    this.setID(trial.getID());
	    this.setExperimentID(trial.getExperimentID());
	    this.setApplicationID(trial.getApplicationID());
	    if((trial.getName()==null)||(trial.getName().equals("")))
		this.setName("Trial "+ this.getID());
	    else
		this.setName(trial.getName());
	    this.setTime(trial.getTime());
	    this.setProblemDefinition(trial.getProblemDefinition());
	    this.setNodeCount(trial.getNodeCount());
	    this.setNumContextsPerNode(trial.getNumContextsPerNode());
	    this.setNumThreadsPerContext(trial.getNumThreadsPerContext());
	    this.setUserData(trial.getUserData());
	    this.dataSession = new ParaProfDBSession();
	    this.dataSession.setDebug(ParaProf.debugIsOn);
	    int numberOfMetrics = trial.getMetricCount();
	    for(int i=0;i<numberOfMetrics;i++){
		Metric metric = dataSession.addMetric();
		metric.setName(trial.getMetric(i));
		metric.setTrial(this);
	    }
	}
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
	dataSession.getGlobalMapping().setColors(clrChooser, -1);
    }

    public void setExperiment(ParaProfExperiment experiment){
	this.experiment = experiment;}
  
    public ParaProfExperiment getExperiment(){
	return experiment;}
  
    public void setDMTN(DefaultMutableTreeNode defaultMutableTreeNode){
	this.defaultMutableTreeNode = defaultMutableTreeNode;}
  
    public DefaultMutableTreeNode getDMTN(){
	return defaultMutableTreeNode;}

    public void setDBTrial(boolean dBTrial){
	this.dBTrial = dBTrial;}
  
    public boolean dBTrial(){
	return dBTrial;}
  
    public String getIDString(){
	if(experiment!=null)
	    return (experiment.getIDString()) + ":" + (super.getID());
	else
	    return  ":" + (super.getID());
    }

    public ColorChooser getColorChooser(){
	return clrChooser;}
  
    public Preferences getPreferences(){
	return preferences;}

    //Used in many ParaProf windows for the title of the window.
    public String getTrialIdentifier(boolean reverse){
	if(path!=null){
	    if(reverse)
		return pathReverse;
	    else
		return path;
	}
	else
	    return "Application " + this.getApplicationID() +
		", Experiment " + this.getExperimentID() +
		", Trial " + this.getID() + ".";
    }
  
    //Sets both the path and the path reverse.
    public void setPaths(String path){
	this.path = path;
	this.pathReverse = FileList.getPathReverse(path);
    }
    
    public String getPath(){
	return path;}
  
    public String getPathReverse(){
	return pathReverse;}

    public String toString(){ 
	return super.getName();}

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
    //End - Functions that control the opening
    //and closing of the static main window for
    //this trial.
    //####################################
    
    public SystemEvents getSystemEvents(){
	return systemEvents;}
  
    public void setSelectedMetricID(int selectedMetricID){
	this.selectedMetricID = selectedMetricID;}
    
    public int getSelectedMetricID(){
	return selectedMetricID;}
  
    public boolean isTimeMetric(){
	String trialName = this.getMetricName(this.getSelectedMetricID());
	trialName = trialName.toUpperCase();
	if(trialName.indexOf("TIME") == -1)
	    return false;
	else
	    return true;
    }
  
    //####################################
    //Pass-though methods to the data session for this instance.
    //####################################
    GlobalMapping getGlobalMapping(){
	return dataSession.getGlobalMapping();}

    public int getNumberOfMappings(){
	return dataSession.getNumberOfMappings();}

    public int getNumberOfUserEvents(){
	return dataSession.getNumberOfUserEvents();}

    public boolean groupNamesPresent(){
	return dataSession.groupNamesPresent();}
  
    public boolean userEventsPresent(){
	return dataSession.userEventsPresent();}

    public boolean callPathDataPresent(){
	return dataSession.callPathDataPresent();}

    public int getTotalNumberOfThreads(){
	return dataSession.getTotalNumberOfThreads();}

    public Vector getMetrics(){
	return dataSession.getMetrics();}

    public int getNumberOfMetrics(){
	return dataSession.getNumberOfMetrics();}

    public int getMetricID(String string){
	return dataSession.getMetricID(string);}

    public String getMetricName(int metricID){
	return dataSession.getMetricName(metricID);}

    public Metric addMetric(){
	return dataSession.addMetric();}

    public NCT getNCT(){
	return dataSession.getNCT();}

    public int[] getMaxNCTNumbers(){
	return dataSession.getMaxNCTNumbers();}

    public void setMeanData(int mappingSelection, int metricID){
	dataSession.setMeanData(mappingSelection, metricID);}
    
    public void setMeanDataAllMetrics(int mappingSelection, int numberOfMetrics){
	dataSession.setMeanData(mappingSelection, numberOfMetrics);}
    //####################################
    //end - Pass-though methods to the data session for this instance.
    //####################################
  
    //####################################
    //Instance data.
    //####################################
    int type = -1;
    ParaProfDataSession dataSession = null;
    ParaProfExperiment experiment = null;
    DefaultMutableTreeNode defaultMutableTreeNode = null;
    private boolean dBTrial = false;
     
    private SystemEvents systemEvents = new SystemEvents();
    private StaticMainWindow sMW = null;
    private ColorChooser clrChooser = new ColorChooser(this, null);
    private Preferences  preferences = new  Preferences(this, null);
  
    private String path = null;
    private String pathReverse = null;
    private int selectedMetricID = 0;

    private GlobalMapping globalMapping;
    private Vector nodes = new Vector();
    //####################################
    //Instance data.
    //####################################
}
