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

package edu.uoregon.tau.paraprof;

import java.io.*;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.*;
import edu.uoregon.tau.dms.dss.*;

public class ParaProfTrial extends Trial implements ParaProfObserver, ParaProfTreeNodeUserObject{

     public ParaProfTrial(int type){
	super();
	this.debug = UtilFncs.debug;

	this.setID(-1);
	this.setExperimentID(-1);
	this.setApplicationID(-1);
	this.setName("");
	this.setTime("");
	this.setProblemDefinition("");
	this.setNodeCount(-1);
	this.setNumContextsPerNode(-1);
	this.setNumThreadsPerContext(-1);
	this.setUserData("");
	((ParaProfDataSession)this.dataSession).setDebug(UtilFncs.debug);
	this.type = type;
     }

    public ParaProfTrial(Trial trial, int type){
	super();
	this.debug = UtilFncs.debug;

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
	    ((ParaProfDataSession)this.dataSession).setDebug(UtilFncs.debug);
	    this.dataSession.setMetrics(trial.getMetrics());
	}
	this.type = type;
    }

    public void initialize(Object obj){
	switch(type){
	case 0:
	    dataSession = new TauPprofOutputSession();
	    ((ParaProfDataSession)dataSession).setDebug(UtilFncs.debug);
	    break;
	case 1:
	    dataSession = new TauOutputSession();
	    ((ParaProfDataSession)dataSession).setDebug(UtilFncs.debug);
	    break;
	case 2:
	    dataSession = new DynaprofOutputSession();
	    ((ParaProfDataSession)dataSession).setDebug(UtilFncs.debug);
	    break;
	case 3:
	    break;
	case 5:
	    dataSession = new GprofOutputSession();
	    ((ParaProfDataSession)dataSession).setDebug(UtilFncs.debug);
	    break;
	default:
	    break;
	}
	
	this.setLoading(true);
	((ParaProfDataSession)dataSession).addObserver(this);
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

    public void setTreePath(TreePath treePath){
	this.treePath = treePath;}

    public TreePath getTreePath(){
	return treePath;}

    public void setDBTrial(boolean dBTrial){
	this.dBTrial = dBTrial;}
  
    public boolean dBTrial(){
	return dBTrial;}

    public void setDefaultTrial(boolean defaultTrial){
	this.defaultTrial = defaultTrial;}

    public boolean defaultTrial(){
	return defaultTrial;}

    public void setLoading(boolean loading){
	this.loading = loading;}

    public boolean loading(){
	return loading;}
  
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

    public ParaProfDataSession getParaProfDataSession(){
	return (ParaProfDataSession)dataSession;}

    //####################################
    //Interface code.
    //####################################
    
    //######
    //ParaProfTreeUserObject
    //######
    public void clearDefaultMutableTreeNodes(){
	this.setDMTN(null);}
    //######
    //End - ParaProfTreeUserObject
    //######

    //####################################
    //End - Interface code.
    //####################################

    //####################################
    //Functions that control the obtaining and the opening
    //and closing of the static main window for
    //this trial.
    //####################################
    public StaticMainWindow getStaticMainWindow(){
	return sMW;}
  
    public void showMainWindow(){
  
	if(sMW == null){
	    sMW = new StaticMainWindow(this, UtilFncs.debug);
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
  
    //####################################
    //Interface for ParaProfMetrics. 
    //####################################
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

    public boolean isDerivedMetric(){
	return this.getMetric(this.getSelectedMetricID()).getDerivedMetric();}
	
    //Overide this function.
    public Vector getMetrics(){
	return dataSession.getMetrics();}

    public DataSessionIterator getMetricList(){
	return new DataSessionIterator(this.getMetrics());}

    public int getNumberOfMetrics(){
	return dataSession.getNumberOfMetrics();}

    public int getMetricID(String string){
	return dataSession.getMetricID(string);}

    public Metric getMetric(int metricID){
	return (Metric) dataSession.getMetric(metricID);}

    public String getMetricName(int metricID){
	return dataSession.getMetricName(metricID);}

    public Metric addMetric(){
	Metric metric = new Metric();
	dataSession.addMetric(metric);
	return metric;
    }

    //####################################
    //End - Interface for ParaProfMetrics. 
    //####################################
  
    //####################################
    //Pass-though methods to the data session for this instance.
    //####################################
    public GlobalMapping getGlobalMapping(){
	return dataSession.getGlobalMapping();}

    public NCT getNCT(){
	return dataSession.getNCT();}

    public boolean groupNamesPresent(){
	return ((ParaProfDataSession)dataSession).groupNamesPresent();}
  
    public boolean userEventsPresent(){
	return ((ParaProfDataSession)dataSession).userEventsPresent();}

    public boolean callPathDataPresent(){
	return ((ParaProfDataSession)dataSession).callPathDataPresent();}

    //Overides the parent getMaxNCTNumbers.
    public int[] getMaxNCTNumbers(){
	return ((ParaProfDataSession)dataSession).getMaxNCTNumbers();}

    public void setMeanData(int mappingSelection, int metricID){
	((ParaProfDataSession)dataSession).setMeanData(mappingSelection, metricID);}
    
    public void setMeanDataAllMetrics(int mappingSelection, int numberOfMetrics){
	((ParaProfDataSession)dataSession).setMeanData(mappingSelection, numberOfMetrics);}
    //####################################
    //end - Pass-though methods to the data session for this instance.
    //####################################

    //######
    //ParaProfObserver interface.
    //######
    public void update(Object obj){
	dataSession.terminate();
	this.setLoading(false);

	//Set the colours.
	clrChooser.setColors(dataSession.getGlobalMapping(), -1);
	
	//The dataSession has accumulated edu.uoregon.tau.dms.dss.Metrics. Inside ParaProf,
	//these need to be paraprof.Metrics.
	int numberOfMetrics = dataSession.getNumberOfMetrics();
	Vector metrics = new Vector();
	for(int i=0;i<numberOfMetrics;i++){
	    Metric metric = new Metric();
	    metric.setName(dataSession.getMetricName(i));
	    metric.setID(i);
	    metric.setTrial(this);
	    metrics.add(metric);
	}
	//Now set the data session metrics.
	dataSession.setMetrics(metrics);
	ParaProf.paraProfManager.populateTrialMetrics(this, defaultTrial);
	
	//Notify any observers of this trial that the Data Session is done.
	this.notifyObservers();
    }
    public void update(){}
    //######
    //End - Observer.
    //######

    //######
    //Methods that manage the ParaProfObservers.
    //######
    public void addObserver(ParaProfObserver observer){
	observers.add(observer);}

    public void removeObserver(ParaProfObserver observer){
	observers.remove(observer);}

    public void notifyObservers(){
	if(this.debug()){
	    System.out.println("######");
	    System.out.println("ParaProfTrial.notifyObservers()");
	    System.out.println("Listening classes ...");
	    for(Enumeration e = observers.elements(); e.hasMoreElements() ;)
		System.out.println(e.nextElement().getClass());
	    System.out.println("######");
	}
	for(Enumeration e = observers.elements(); e.hasMoreElements() ;)
	    ((ParaProfObserver) e.nextElement()).update(this);
    }
    //######
    //End - Methods that manage the ParaProfObservers.
    //######

    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}
    //####################################
    //Instance data.
    //####################################
    int type = -1;
    boolean defaultTrial = false;
    ParaProfExperiment experiment = null;
    DefaultMutableTreeNode defaultMutableTreeNode = null;
    private TreePath treePath = null;
    private boolean dBTrial = false;
    private boolean loading = false;
     
    private SystemEvents systemEvents = new SystemEvents();
    private StaticMainWindow sMW = null;
    private ColorChooser clrChooser = new ColorChooser(this, null);
    private Preferences  preferences = new  Preferences(this, null);
  
    private String path = null;
    private String pathReverse = null;
    private int selectedMetricID = 0;
    private Vector observers = new Vector();
    private boolean debug = false;
    //####################################
    //Instance data.
    //####################################
}
