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

public class ParaProfTrial extends Trial implements ParaProfObserver, ParaProfTreeNodeUserObject{

     public ParaProfTrial(int type){
	super();
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
	this.dataSession.setDebug(UtilFncs.debug);
	this.type = type;
    }

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
	    this.dataSession.setDebug(UtilFncs.debug);
	    int numberOfMetrics = trial.getMetricCount();
	    for(int i=0;i<numberOfMetrics;i++)
		dataSession.addMetric(trial.getMetricName(i));
	}
	this.type = type;
    }

    public void initialize(Object obj){
	switch(type){
	case 0:
	    dataSession = new TauPprofOutputSession();
	    dataSession.setDebug(UtilFncs.debug);
	    break;
	case 1:
	    dataSession = new TauOutputSession();
	    dataSession.setDebug(UtilFncs.debug);
	    break;
	case 2:
	    dataSession = new DynaprofOutputSession();
	    dataSession.setDebug(UtilFncs.debug);
	    break;
	case 3:
	    break;
	default:
	    break;
	}
	
	this.setLoading(true);
	dataSession.addObserver(this);
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

    public Vector getMetrics(){
	return metrics;}

    public ParaProfIterator getMetricList(){
	return new ParaProfIterator(metrics);}

    public int getNumberOfMetrics(){
	return metrics.size();}

    public int getMetricID(String string){
	for(Enumeration e = metrics.elements(); e.hasMoreElements() ;){
	    Metric metric = (Metric) e.nextElement();
	    if((metric.getName()).equals(string))
		return metric.getID();
	}
	return -1;
    }

    public Metric getMetric(int metricID){
	return (Metric) metrics.elementAt(metricID);}

    public String getMetricName(int metricID){
	return this.getMetric(metricID).getName();}

    public Metric addMetric(){
	Metric metric = new Metric();
	metric.setID((metrics.size()));
	metrics.add(metric);
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

    //######
    //ParaProfObserver interface.
    //######
    public void update(Object obj){
	dataSession.terminate();
	this.setLoading(false);

	//Set the colours.
	clrChooser.setColors(dataSession.getGlobalMapping(), -1);
	
	//Set the metrics.
	int numberOfMetrics = dataSession.getNumberOfMetrics();
	for(int i=0;i<numberOfMetrics;i++){
	    Metric metric = this.addMetric();
	    metric.setName(dataSession.getMetricName(i));
	    metric.setTrial(this);
	}

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
	for(Enumeration e = observers.elements(); e.hasMoreElements() ;)
	    ((ParaProfObserver) e.nextElement()).update(this);
    }
    //######
    //End - Methods that manage the ParaProfObservers.
    //######
  
    //####################################
    //Instance data.
    //####################################
    int type = -1;
    boolean defaultTrial = false;
    ParaProfDataSession dataSession = null;
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

    private Vector metrics = new Vector();

    private Vector observers = new Vector();
    //####################################
    //Instance data.
    //####################################
}
