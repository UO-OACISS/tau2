/* 
  ApplicationManager.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  This controls the adding of aaplications to the system.
*/

package ParaProf;

import java.util.*;

public class ApplicationManager extends Observable 
{
  //Constructors.
  public ApplicationManager(){
  }
  
  //Public methods.
  public Application addApplication(){
    Application newApplication = new Application();
    newApplication.setApplicationID((applications.size()));
    applications.add(newApplication);
    return newApplication;
  }
  
  
  public void removeApplication(Object inObject){
    applications.remove(inObject);
  }
  
  public Vector getApplicationList()
  {
    return applications;
  }
  
  public boolean isEmpty(){
    if((applications.size()) == 0)
      return true;
    else
      return false;
  }

  
  public boolean isApplicationNamePresent(String inString){
    
    for(Enumeration e = applications.elements(); e.hasMoreElements() ;)
    {
      Application exp = (Application) e.nextElement();
      if(inString.equals(exp.getApplicationName()))
        return true;
    }
    
    //If we make it here, the applications name is not present.  Return false.
    return false;
  }
  
  public String getPathReverse(String inString){
    
    //Now set the reverse.
    String tmpString1 = inString;
    String tmpString2 = "";
    String tmpString3 = "";
    
    boolean isForwardSlash = false; //Just to make the reverse string look nicer on
                    //Unix based systems.
    
    int length = tmpString1.length();
                    
    for(int i=(length-1); i>=0; i--){
      char tmpChar = tmpString1.charAt(i);
      
      if(tmpChar == '/'){
      
        //This does not really need to get done more than once but ...
        isForwardSlash = true;
      
        if(tmpString3.equals(""))
          tmpString3 = tmpString2;
        else
          tmpString3 = tmpString3 + tmpChar + tmpString2;
        tmpString2 = "";
      }
      else if(tmpChar == '\\'){
        if(tmpString3.equals(""))
          tmpString3 = tmpString2;
        else
          tmpString3 = tmpString3 + tmpChar + tmpString2;
        tmpString2 = "";
      }
      else{
        tmpString2 = tmpChar + tmpString2;
      }
    }
      
    if(isForwardSlash)
      tmpString3 = tmpString3 + "/";
    
    return tmpString3;
  }

  //Instance data.
  Vector applications = new Vector();
}
