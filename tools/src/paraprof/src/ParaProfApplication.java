/* 
   ParaProfApplication.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import javax.swing.tree.*;
import dms.dss.*;

public class ParaProfApplication extends Application{

    public ParaProfApplication(){
	super();}

    public ParaProfApplication(Application application){
	this.setID(application.getID());
	this.setName(application.getName());
	this.setVersion(application.getVersion());
	this.setDescription(application.getDescription());
	this.setLanguage(application.getLanguage());
	this.setParaDiag(application.getParaDiag());
	this.setUsage(application.getUsage());
	this.setExecutableOptions(application.getExecutableOptions());
	this.setUserData(application.getUserData());
    }
  
    public void setDMTN(DefaultMutableTreeNode defaultMutableTreeNode){
	this.defaultMutableTreeNode = defaultMutableTreeNode;}
  
    public DefaultMutableTreeNode getDMTN(){
	return defaultMutableTreeNode;}
  
    public void setDBParaProfApplication(boolean dbParaProfApplication){
	this.dbParaProfApplication = dbParaProfApplication;}
  
    public boolean isDBParaProfApplication(){
	return dbParaProfApplication;}
  
    public Vector getExperiments(){
	return experiments;}
  
    public ParaProfExperiment addExperiment(){
	ParaProfExperiment experiment = new ParaProfExperiment();
	experiment.setApplication(this);
	experiment.setID((experiments.size()));
	experiments.add(experiment);
	return experiment;
    }
  
    public void removeParaProfExperiment(ParaProfExperiment experiment){
	experiments.remove(experiment);}
  
    public boolean isExperimentPresent(String name){
	for(Enumeration e = experiments.elements(); e.hasMoreElements() ;){
	    ParaProfExperiment exp = (ParaProfExperiment) e.nextElement();
	    if(name.equals(exp.getName()))
		return true;
	}
	//If we make it here, the experiment run name is not present.  Return false.
	return false;
    }

    public String getIDString(){
	return Integer.toString(this.getID());}
  
    public String toString(){ 
	return super.getName();}

    //####################################
    //Instance data.
    //####################################
    private DefaultMutableTreeNode defaultMutableTreeNode = null;
    private boolean dbParaProfApplication = false;
    private Vector experiments = new Vector();
    //####################################
    //End - Instance data.
    //####################################
}
