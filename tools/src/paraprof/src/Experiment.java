/* 
  Experiment.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package paraprof;

import java.util.*;
import javax.swing.tree.*;

public class Experiment
{
  //Constructors. 
  public Experiment(Application inParentApp){
    parentApplication = inParentApp;
    trials = new Vector();
  }
  
  public Application getParentApplication(){
    return parentApplication;
  }
  
  public void setDMTN(DefaultMutableTreeNode inNode){
    nodeRef = inNode;
  }
  
  public DefaultMutableTreeNode getDMTN(){
    return nodeRef;
  }
  
  public void setExperimentName(String inExperimentName)
  {
    experimentName = inExperimentName;
  }
  
  public String getExperimentName()
  {
    return experimentName;
  }
  
  public Vector getTrials(){
    return trials;
  }
  
  public void setExperimentID(int inExperimentID){
    experimentID = inExperimentID;
    //Since the parentExperiment is set in the constructor,
    //it is not null.  Therefore we can safely set the experimentIDString.
    experimentIDString = parentApplication.getApplicationIDString() + experimentID;
  }
  
  public int getExperimentID(){
    return experimentID;
  }
  
  public String getExperimentIDString(){
    return experimentIDString;
  }
  
  public Trial addTrial(){
    Trial newTrial = new Trial(this);
    newTrial.setTrialID((trials.size()));
    trials.add(newTrial);
    return newTrial;
  }
  
  public boolean isTrialNamePresent(String inString){
    
    for(Enumeration e = trials.elements(); e.hasMoreElements() ;)
    {
      Trial trial = (Trial) e.nextElement();
      if(inString.equals(trial.toString()))
        return true;
    }
    
    //If we make it here, the experiment run name is not present.  Return false.
    return false;
  }
  
  public String toString()
  { 
    return experimentName;  
  }
  
  
  //Data section.
  Application parentApplication = null;
  DefaultMutableTreeNode nodeRef = null;
  private String experimentName = null;
  private int experimentID = -1;
  private String experimentIDString = null;
  Vector trials = null;
}
