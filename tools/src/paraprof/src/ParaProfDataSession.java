
/* 
   ParaProfDataSession.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import dms.dss.*;

public abstract class ParaProfDataSession  extends DataSession{
    public ParaProfDataSession () {
	super();
    }

    //####################################
    //Public Section.
    //####################################

    public NCT getNCT(){
	return nct;}

    public Vector getMetrics(){
	return metrics;}

    public GlobalMapping getGlobalMapping(){
	return globalMapping;}

    public Vector getMaxMeanExclusiveList(){
	return maxMeanExclusiveValueList;}

    public Vector getMaxMeanInclusiveList(){
	return maxMeanInclusiveValueList;}

    public Vector getMaxMeanInclusivePercentList(){
	return maxMeanInclusivePercentValueList;}

    public Vector getMaxMeanExclusivePercentList(){
	return maxMeanExclusivePercentValueList;}
  
    public Vector getMaxMeanUserSecPerCallList(){
	return maxMeanUserSecPerCallList;}

    public double getMaxMeanExclusiveValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanExclusiveValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}

    public double getMaxMeanInclusiveValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanInclusiveValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}

    public double getMaxMeanInclusivePercentValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanInclusivePercentValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}

    public double getMaxMeanExclusivePercentValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanExclusivePercentValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}

    public double getMaxMeanNumberOfCalls(){
	return maxMeanNumberOfCalls;}
  
    public double getMaxMeanNumberOfSubRoutines(){
	return maxMeanNumberOfSubRoutines;}
  
    public double getMaxMeanUserSecPerCall(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanUserSecPerCallList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}

    public int getNumberOfMappings(){
	return globalMapping.getNumberOfMappings(0);}

    public int getNumberOfUserEvents(){
	return numberOfUserEvents;}
    
    public boolean groupNamesPresent(){
	return groupNamesPresent;}

    public boolean userEventsPresent(){
	return userEventsPresent;}
  
    public boolean callPathDataPresent(){
	return callPathDataPresent;}
    
    //####################################
    //End - Public Section.
    //####################################
    
    //####################################
    //Protected Section.
    //####################################
    protected void increaseVectorStorage(){
	maxMeanInclusiveValueList.add(new Double(0));
	maxMeanExclusiveValueList.add(new Double(0));
	maxMeanInclusivePercentValueList.add(new Double(0));
	maxMeanExclusivePercentValueList.add(new Double(0));
	maxMeanUserSecPerCallList.add(new Double(0));
    }

    protected Metric addMetric(){
	Metric newMetric = new Metric();
	newMetric.setID((metrics.size()));
	metrics.add(newMetric);
	return newMetric;
    }

    protected void setMaxMeanInclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanInclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    protected void setMaxMeanExclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanExclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    protected void setMaxMeanInclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanInclusivePercentValueList.add(dataValueLocation, tmpDouble);}
  
    protected void setMaxMeanExclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanExclusivePercentValueList.add(dataValueLocation, tmpDouble);}

    protected void setMaxMeanNumberOfCalls(double inDouble){
	maxMeanNumberOfCalls = inDouble;}
  
    protected void setMaxMeanNumberOfSubRoutines(double inDouble){
	maxMeanNumberOfSubRoutines = inDouble;}

    protected void setMaxMeanUserSecPerCall(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanUserSecPerCallList.add(dataValueLocation, tmpDouble);}
  
    protected void setNumberOfUserEvents(int numberOfUserEvents){
	this.numberOfUserEvents = numberOfUserEvents;}

    protected void setGroupNamesPresent(boolean groupNamesPresent){
	this.groupNamesPresent = groupNamesPresent;}
  
    protected void setUserEventsPresent(boolean userEventsPresent){
	this.userEventsPresent = userEventsPresent;}

    protected void setCallPathDataPresent(boolean callPathDataPresent){
	this.callPathDataPresent = callPathDataPresent;}

    protected void setFirstMetric(boolean firstMetric){
	this.firstMetric = firstMetric;}

    protected boolean firstMetric(){
	return firstMetric;}

    protected void setGroupCheck(boolean groupCheck){
	this.groupCheck = groupCheck;}

    protected boolean groupCheck(){
	return groupCheck;}
    //####################################
    //End - Protected Section.
    //####################################

    //####################################
    //Instance data.
    //####################################

    //######
    //Private Section.
    //######
    private boolean firstMetric = true;
    private boolean groupCheck = false;
    
    private int numberOfUserEvents = 0;
    private int totalNumberOfContexts = -1;
    private int totalNumberOfThreads = -1;
    private boolean groupNamesPresent = false;
    private boolean userEventsPresent = false;
    private boolean callPathDataPresent = false;

    private GlobalMapping globalMapping = new GlobalMapping();
    private NCT nct = new NCT();
    private Vector metrics = new Vector();

    private Vector maxMeanInclusiveValueList = new Vector();
    private Vector maxMeanExclusiveValueList = new Vector();
    private Vector maxMeanInclusivePercentValueList = new Vector();
    private Vector maxMeanExclusivePercentValueList = new Vector();
    private double maxMeanNumberOfCalls = 0;
    private double maxMeanNumberOfSubRoutines = 0;
    private Vector maxMeanUserSecPerCallList = new Vector();
    //######
    //End - Private Section.
    //######

    //####################################
    //End - Instance data.
    //####################################


}
