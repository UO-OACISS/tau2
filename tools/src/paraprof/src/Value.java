/* 
  Value.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package ParaProf;

import javax.swing.tree.*;

public class Value{
  public Value(Trial inParentTrial){
    parentTrial = inParentTrial;
  }
    
  public Trial getParentTrial(){
    return parentTrial;
  }
  
  public void setDMTN(DefaultMutableTreeNode inNode){
    nodeRef = inNode;
  }
  
  public DefaultMutableTreeNode getDMTN(){
    return nodeRef;
  }
  
  public void setValueName(String inValueName)
  {
    valueName = inValueName;
  }
  
  public String getValueName()
  {
    return valueName;
  }
  
  public void setValueID(int inValueID){
    valueID = inValueID;
    //Since the parentTrial is set in the constructor,
    //it is not null.  Therefore we can safely set the experimentIDString.
    valueIDString = parentTrial.getTrialIDString() + valueID;
  }
  
  public int getValueID(){
    return valueID;
  }
  
  public String getValueIDString(){
    return valueIDString;
  }
  
  
  
  public String toString(){
    return valueIDString + " - " + valueName;
  }
  
  Trial parentTrial = null;
  DefaultMutableTreeNode nodeRef = null;
  private String valueName = null;
  private int valueID = -1;
  private String valueIDString = null;
}
