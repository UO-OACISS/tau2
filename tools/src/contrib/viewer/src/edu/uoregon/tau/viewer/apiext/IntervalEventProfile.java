/*
 * Created on Jun 4, 2003
 * This class capsulate gerneral fuction information and function performance profile data.
 */

package edu.uoregon.tau.viewer.apiext;
import edu.uoregon.tau.dms.dss.*;

/**
 * @author lili
 *
 */
public class IntervalEventProfile {
    private IntervalLocationProfile myIntervalLocationProfile;
    private IntervalEvent myIntervalEvent; 
    private int timeMetricIndex; // identify execution time metric's index  
    		
    public IntervalEventProfile(DataSession session, IntervalLocationProfile funcData){
	myIntervalEvent = session.getIntervalEvent(funcData.getIntervalEventID());
	myIntervalLocationProfile = funcData;			
    }

    public void setTimeMetricIndex(int metricIndex){
	this.timeMetricIndex = metricIndex;
    }

    public int getTimeMetricIndex(){ return timeMetricIndex; }
	
    public String getName(){
	return myIntervalEvent.getName();
    }
	
    public int getIntervalEventID(){
	return myIntervalEvent.getID();
    }
	
    public String getGroup(){
	return myIntervalEvent.getGroup();
    }
	
    public int getTrialID(){
	return myIntervalEvent.getTrialID();
    }
	
    public int getNode () {
	return myIntervalLocationProfile.getNode();
    }

    public int getContext () {
	return myIntervalLocationProfile.getContext();
    }

    public int getThread () {
	return myIntervalLocationProfile.getThread();
    }

    public double getInclusivePercentage (int metricIndex) {
	return myIntervalLocationProfile.getInclusivePercentage(metricIndex);
    }

    public double getInclusivePercentage () {
	return myIntervalLocationProfile.getInclusivePercentage();
    }

    public double getInclusive (int metricIndex) {
	return myIntervalLocationProfile.getInclusive(metricIndex);
    }

    public double getInclusive () {
	return myIntervalLocationProfile.getInclusive();
    }

    public double getExclusivePercentage (int metricIndex) {
	return myIntervalLocationProfile.getExclusivePercentage(metricIndex);
    }

    public double getExclusivePercentage () {
	return myIntervalLocationProfile.getExclusivePercentage();
    }

    public double getExclusive (int metricIndex) {
	return myIntervalLocationProfile.getExclusive(metricIndex);
    }

    public double getExclusive () {
	return myIntervalLocationProfile.getExclusive();
    }

    public int getNumCalls () {
	return myIntervalLocationProfile.getNumCalls();
    }

    public int getNumSubroutines () {
	return myIntervalLocationProfile.getNumSubroutines();
    }

    public double getInclusivePerCall (int metricIndex) {
	return myIntervalLocationProfile.getInclusivePerCall(metricIndex);
    }

    public double getInclusivePerCall () {
	return myIntervalLocationProfile.getInclusivePerCall();
    }
}
