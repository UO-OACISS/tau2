/* 
   ApplicationManager.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  This controls the adding of aaplications to the system.
*/

package paraprof;

import java.util.*;
import dms.dss.*;

public class ApplicationManager extends Observable{
    public ApplicationManager(){}
  
    //Public methods.
    public ParaProfApplication addApplication(){
	ParaProfApplication application = new ParaProfApplication();
	application.setID((applications.size()));
	applications.add(application);
	return application;
    }
  
    public void removeApplication(Object obj){
	applications.remove(obj);}
  
    public Vector getApplications(){
	return applications;}

    public DataSessionIterator getApplicationList(){
	return new DataSessionIterator(applications);}
  
    public boolean isEmpty(){
	if((applications.size()) == 0)
	    return true;
	else
	    return false;
    }

    public ParaProfApplication getApplication(int applicationID){
	return (ParaProfApplication) applications.elementAt(applicationID);}

    public ParaProfExperiment getExperiment(int applicationID, int experimentID){
	return (this.getApplication(applicationID)).getExperiment(experimentID);}

    public ParaProfTrial getTrial(int applicationID, int experimentID, int trialID){
	return ((this.getApplication(applicationID)).getExperiment(experimentID)).getTrial(trialID);}

  
    public boolean isApplicationPresent(String name){
	for(Enumeration e = applications.elements(); e.hasMoreElements() ;){
		ParaProfApplication application = (ParaProfApplication) e.nextElement();
		if(name.equals(application.getName()))
		    return true;
	    }
  	return false;
    }

    //Instance data.
    Vector applications = new Vector();
}
