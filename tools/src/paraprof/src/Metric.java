/* 
  Metric.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package paraprof;

import javax.swing.tree.*;

public class Metric{
  public Metric(Trial inParentTrial){
    parentTrial = inParentTrial;}
    
  public Trial getParentTrial(){
    return parentTrial;}
  
  public void setDMTN(DefaultMutableTreeNode inNode){
    nodeRef = inNode;}
  
  public DefaultMutableTreeNode getDMTN(){
    return nodeRef;}
  
  public void setMetricName(String inMetricName){
    metricName = inMetricName;}
  
  public String getMetricName(){
    return metricName;}
  
  public void setMetricID(int inMetricID){
    metricID = inMetricID;
    //Since the parentTrial is set in the constructor,
    //it is not null.  Therefore we can safely set the experimentIDString.
    metricIDString = parentTrial.getTrialIDString() + metricID;
  }
  
  public int getMetricID(){
    return metricID;}
  
  public String getMetricIDString(){
    return metricIDString;}
    
  public String toString(){
    return metricIDString + " - " + metricName;}
  
  Trial parentTrial = null;
  DefaultMutableTreeNode nodeRef = null;
  private String metricName = null;
  private int metricID = -1;
  private String metricIDString = null;
}
