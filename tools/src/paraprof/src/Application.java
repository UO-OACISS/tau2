/* 
  Application.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package ParaProf;

import java.util.*;
import javax.swing.tree.*;

public class Application
{
  //Constructors. 
  public Application(){
    experiments = new Vector();
  }
  
  public Application(String inApplicationName)
  {
    applicationName = inApplicationName;
    experiments = new Vector();
  }
  
  public void setDMTN(DefaultMutableTreeNode inNode){
    nodeRef = inNode;
  }
  
  public DefaultMutableTreeNode getDMTN(){
    return nodeRef;
  }
  
  public void setDBApplication(boolean inBoolean){
    dbApplication = inBoolean;
  }
  
  public boolean isDBApplication(){
    return dbApplication;
  }
  
  
  public void setApplicationName(String inApplicationName)
  {
    applicationName = inApplicationName;
  }
  
  public String getApplicationName()
  {
    return applicationName;
  }
  
  public Vector getExperimentsList(){
    return experiments;
  }
  
  public Experiment addExperiment()
  {
    Experiment newExperiment = new Experiment(this);
    newExperiment.setExperimentID((experiments.size()));
    experiments.add(newExperiment);
    return newExperiment;
  }
  
  public void removeExperiment(Experiment inExperiment){
    experiments.remove(inExperiment);
  }
  
  public boolean isExperimentNamePresent(String inString){
    
    for(Enumeration e = experiments.elements(); e.hasMoreElements() ;)
    {
      Experiment exp = (Experiment) e.nextElement();
      if(inString.equals(exp.toString()))
        return true;
    }
    
    //If we make it here, the experiment run name is not present.  Return false.
    return false;
  }
  
  public String toString()
  { 
    return applicationName; 
  }
  
  public void setApplicationID(int inApplicationID){
    applicationID = inApplicationID;
    //Set the id string.
    applicationIDString = ""+applicationID;
  }
  
  public int getApplicationID(){
    return applicationID;
  }
  
  public String getApplicationIDString(){
    return applicationIDString;
  }
  
  public void setVersion(String inVersion){
    version = inVersion;
  }
  
  public String getVersion(){
    return version;
  }
  
  public void setDescription(String inDescription){
    description = inDescription;
  }
  
  public String getDescription(){
    return description;
  }
  
  public void setLanguage(String inLanguage){
    language = inLanguage;
  }
  
  public String getLanguage(){
    return language;
  }
  
  public void setPara_diag(String inPara_diag){
    Para_diag = inPara_diag;
  }
  
  public String getPara_diag(){
    return Para_diag;
  }
  
  public void setUsage(String inUsage){
    usage = inUsage;
  }
  
  public String getUsage(){
    return usage;
  }
  
  public void setExe_opt(String inExe_opt){
    Exe_opt = inExe_opt;
  }
  
  public String getExe_opt(){
    return Exe_opt;
  }
  
  
  //Data section.
  DefaultMutableTreeNode nodeRef = null;
  private boolean dbApplication = false;
  private int applicationID = -1;
  private String applicationIDString = null;
  private String applicationName = "Not Set";
  private String version = "Not Set";
  private String description = "Not Set";
  private String language = "Not Set";
  private String Para_diag = "Not Set";
  private String usage = "Not Set";
  private String Exe_opt = "Not Set";
  Vector experiments = null;
}
