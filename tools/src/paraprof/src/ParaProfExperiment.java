/* 
   ParaProfExperiment.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import javax.swing.tree.*;
import dms.dss.*;

public class ParaProfExperiment extends Experiment{

    public ParaProfExperiment(){
	super();}

    public void setApplication(ParaProfApplication application){
	this.application = application;}

    public ParaProfApplication getApplication(){
	return application;}
  
    public void setDMTN(DefaultMutableTreeNode defaultMutableTreeNode){
	this.defaultMutableTreeNode = defaultMutableTreeNode;}
  
    public DefaultMutableTreeNode getDMTN(){
	return defaultMutableTreeNode;}
  
    public Vector getTrials(){
	return trials;}
  
    public void addTrial(ParaProfTrial trial){
	trial.setExperiment(this);
	trial.setID((trials.size()));
	trials.add(trial);
    }
  
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
    //Instance data.
    //####################################
    ParaProfApplication application = null;
    DefaultMutableTreeNode defaultMutableTreeNode = null;
    Vector trials = new Vector();
    //####################################
    //End - Instance data.
    //####################################
}
