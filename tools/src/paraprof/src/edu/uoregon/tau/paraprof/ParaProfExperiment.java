/* 
   ParaProfExperiment.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package edu.uoregon.tau.paraprof;

import java.util.*;
import javax.swing.tree.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.database.*;

public class ParaProfExperiment extends Experiment implements ParaProfTreeNodeUserObject{

    public ParaProfExperiment(){
	super(0);
	this.setID(-1);
	this.setApplicationID(-1);
	this.setName("");
    }

    public ParaProfExperiment(DB db){
	super(db);
	this.setID(-1);
	this.setApplicationID(-1);
	this.setName("");
    }

    public ParaProfExperiment(Experiment experiment){
	super(experiment);
    }

    public void setApplication(ParaProfApplication application){
	this.application = application;}

    public ParaProfApplication getApplication(){
	return application;}
  
    public void setDMTN(DefaultMutableTreeNode defaultMutableTreeNode){
	this.defaultMutableTreeNode = defaultMutableTreeNode;}
  
    public DefaultMutableTreeNode getDMTN(){
	return defaultMutableTreeNode;}

    public void setTreePath(TreePath treePath){
	this.treePath = treePath;}

    public TreePath getTreePath(){
	return treePath;}
  
    public void setDBExperiment(boolean dBExperiment){
	this.dBExperiment = dBExperiment;}
  
    public boolean dBExperiment(){
	return dBExperiment;}

    public Vector getTrials(){
	return trials;}

    public DataSessionIterator getTrialList(){
	return new DataSessionIterator(trials);}

    public ParaProfTrial getTrial(int trialID){
	return (ParaProfTrial) trials.elementAt(trialID);}
  
    public void addTrial(ParaProfTrial trial){
	trial.setExperiment(this);
	trial.setID((trials.size()));
	trials.add(trial);
    }

    public void removeTrial(ParaProfTrial trial){
	trials.remove(trial);}
  
    public boolean isTrialPresent(String name){
 	for(Enumeration e = trials.elements(); e.hasMoreElements() ;){
		ParaProfTrial trial = (ParaProfTrial) e.nextElement();
		if(name.equals(trial.toString()))
		    return true;
	}
    	return false;
    }

    public String getIDString(){
	if(application!=null)
	    return (application.getIDString()) + ":" + (super.getID());
	else
	    return  ":" + (super.getID());
    }
  
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
    //Instance data.
    //####################################
    private ParaProfApplication application = null;
    private DefaultMutableTreeNode defaultMutableTreeNode = null;
    private TreePath treePath = null;
    private boolean dBExperiment = false;
    private Vector trials = new Vector();
    //####################################
    //End - Instance data.
    //####################################
}
