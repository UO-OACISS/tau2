/* 
  SystemEvents.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package edu.uoregon.tau.paraprof;

import java.util.*;

public class SystemEvents extends Observable
{
  public void updateRegisteredObjects(String inString)
  {
    //Set this object as changed.
    this.setChanged();
    
    //Now notify observers.
    this.notifyObservers(inString);
  }
  
}
