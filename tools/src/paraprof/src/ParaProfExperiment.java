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
	super();
	this.setID(-1);
	this.setApplicationID(-1);
	this.setName("");
	this.setUserData("");
	this.setSystemName("");
	this.setSystemMachineType("");
	this.setSystemArch("");
	this.setSystemOS("");
	this.setSystemMemorySize("");
	this.setSystemProcessorAmount("");
	this.setSystemL1CacheSize("");
	this.setSystemL2CacheSize("");
	this.setSystemUserData("");
	this.setConfigurationPrefix("");
	this.setConfigurationArchitecture("");
	this.setConfigurationCpp("");
	this.setConfigurationCc("");
	this.setConfigurationJdk("");
	this.setConfigurationProfile("");
	this.setConfigurationUserData("");
	this.setCompilerCppName("");
	this.setCompilerCppVersion("");
	this.setCompilerCcName("");
	this.setCompilerCcVersion("");
	this.setCompilerJavaDirpath("");
	this.setCompilerJavaVersion("");
	this.setCompilerUserData("");
    }

    public ParaProfExperiment(Experiment experiment){
	super();
	this.setID(experiment.getID());
	this.setApplicationID(experiment.getApplicationID());
	this.setName(experiment.getName());
	this.setUserData(experiment.getUserData());
	this.setSystemName(experiment.getSystemName());
	this.setSystemMachineType(experiment.getSystemMachineType());
	this.setSystemArch(experiment.getSystemArch());
	this.setSystemOS(experiment.getSystemOS());
	this.setSystemMemorySize(experiment.getSystemMemorySize());
	this.setSystemProcessorAmount(experiment.getSystemProcessorAmount());
	this.setSystemL1CacheSize(experiment.getSystemL1CacheSize());
	this.setSystemL2CacheSize(experiment.getSystemL2CacheSize());
	this.setSystemUserData(experiment.getSystemUserData());
	this.setConfigurationPrefix(experiment.getConfigPrefix());
	this.setConfigurationArchitecture(experiment.getConfigArchitecture());
	this.setConfigurationCpp(experiment.getConfigCpp());
	this.setConfigurationCc(experiment.getConfigCc());
	this.setConfigurationJdk(experiment.getConfigJdk());
	this.setConfigurationProfile(experiment.getConfigProfile());
	this.setConfigurationUserData(experiment.getConfigUserData());
	this.setCompilerCppName(experiment.getCompilerCppName());
	this.setCompilerCppVersion(experiment.getCompilerCppVersion());
	this.setCompilerCcName(experiment.getCompilerCcName());
	this.setCompilerCcVersion(experiment.getCompilerCcVersion());
	this.setCompilerJavaDirpath(experiment.getCompilerJavaDirpath());
	this.setCompilerJavaVersion(experiment.getCompilerJavaVersion());
	this.setCompilerUserData(experiment.getCompilerUserData());
    }

    public void setApplication(ParaProfApplication application){
	this.application = application;}

    public ParaProfApplication getApplication(){
	return application;}
  
    public void setDMTN(DefaultMutableTreeNode defaultMutableTreeNode){
	this.defaultMutableTreeNode = defaultMutableTreeNode;}
  
    public DefaultMutableTreeNode getDMTN(){
	return defaultMutableTreeNode;}
  
    public void setDBExperiment(boolean dBExperiment){
	this.dBExperiment = dBExperiment;}
  
    public boolean dBExperiment(){
	return dBExperiment;}

    public Vector getTrials(){
	return trials;}

    public ListIterator getTrialList(){
	return new ParaProfIterator(trials);}

    public ParaProfTrial getTrial(int trialID){
	return (ParaProfTrial) trials.elementAt(trialID);}
  
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
    private ParaProfApplication application = null;
    private DefaultMutableTreeNode defaultMutableTreeNode = null;
    private boolean dBExperiment = false;
    private Vector trials = new Vector();
    //####################################
    //End - Instance data.
    //####################################
}
