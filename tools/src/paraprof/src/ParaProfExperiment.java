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

    public void setParentParaProfApplication(){
	super();}

    public void setParentApplication(ParaProfApplication paraProfApplication){
	this.paraProfApplication = paraProfApplication;}

    public ParaProfApplication getParentApplication(){
	return parentApplication;}
  
    public void setDMTN(DefaultMutableTreeNode defaultMutableTreeNode){
	this.defaultMutableTreeNode = defaultMutableTreeNode;}
  
    public DefaultMutableTreeNode getDMTN(){
	return defaultMutableTreeNode;}
  
    public Vector getTrials(){
	return trials;}
  
    public ParaProfTrial addTrial(){
	ParaProfTrial trial = new ParaProfTrial(this);
	trial.setID((trials.size()));
	trials.add(trial);
	return trial;
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
	return (parentApplication.getIDString()) + (super.getID());}
  
    public String toString(){ 
	return super.getName();}
    
    //####################################
    //Instance data.
    //####################################
    ParaProfApplication parentApplication = null;
    DefaultMutableTreeNode defaultMutableTreeNode = null;
    Vector trials = new Vector();
    //####################################
    //End - Instance data.
    //####################################
}
