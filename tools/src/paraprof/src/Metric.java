/* 
  Metric.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package paraprof;

import javax.swing.tree.*;

public class Metric{
    public Metric(){}
    
    public void setParentParaProfTrial(ParaProfTrial parentParaProfTrial){
	this.parentParaProfTrial = parentParaProfTrial;}

    public ParaProfTrial getParentParaProfTrial(){
	return parentParaProfTrial;}
    
    public void setDMTN(DefaultMutableTreeNode defaultMutableTreeNode){
	this.defaultMutableTreeNode = defaultMutableTreeNode;}
    
    public DefaultMutableTreeNode getDMTN(){
	return defaultMutableTreeNode;}
    
    public void setName(String name){
	this.name = name;}
    
    public String getName(){
	return name;}
    
    public void setID(int id){
	this.id = id;}
    
    public int getID(){
	return id;}
    
    public String getIDString(){
	if(parentParaProfTrial!=null)
	    return parentParaProfTrial.getParaProfTrialIDString() + id + " - " + name;
	else
	    return id + " - " + name;
    }
    
    public String toString(){
	if(parentParaProfTrial!=null)
	    return parentParaProfTrial.getParaProfTrialIDString() + id + " - " + name;
	else
	    return id + " - " + name;
    }
    
    ParaProfTrial parentParaProfTrial = null;
    DefaultMutableTreeNode defaultMutableTreeNode = null;
    private String name = null;
    private int id = -1;
}
