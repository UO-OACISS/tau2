package edu.uoregon.tau.viewer.perfcomparison;

import java.awt.*;
import edu.uoregon.tau.dms.dss.*;

/**
 * @author lili
 * this class capsulates function info, performance data, display shapes of a function. 
 *
 */
public class ComparisonWindowIntervalEvent{
    
		private Shape myShape; 
        private IntervalLocationProfile funcData;
        private int numCalls;
        private int numSubroutines;
     
        private String groupName;
        private String trialName;
        private int trialID;

	public ComparisonWindowIntervalEvent(IntervalEvent function, IntervalLocationProfile aFuncData){
	    this.funcData = aFuncData;
	    this.numCalls = aFuncData.getNumCalls();
	    this.numSubroutines = aFuncData.getNumSubroutines();	    
	    this.groupName = function.getGroup();
	    this.trialName = "Trial "+function.getTrialID();
	    this.trialID = function.getTrialID();
	}

	public void setShape(Shape shape){ myShape = shape;}
	public Shape getShape(){ return myShape;}

        public String getGroup(){ return groupName;}

        public String getTrialName(){ return trialName;}

        public int getTrialID(){return trialID;}

	public double getInclusivePercentage (int metricIndex) {
                return funcData.getInclusivePercentage(metricIndex);
        }

        public double getInclusive (int metricIndex) {
                return funcData.getInclusive(metricIndex);
        }

		public double getExclusivePercentage (int metricIndex) {
                return funcData.getExclusivePercentage(metricIndex);
        }

        public double getExclusive (int metricIndex) {
                return funcData.getExclusive(metricIndex);
        }

        public int getNumCalls () {
                return this.numCalls;
        }

        public int getNumSubroutines () {
                return this.numSubroutines;
        }

        public double getInclusivePerCall (int metricIndex) {
                return funcData.getInclusivePerCall(metricIndex);
        }
    
        public double getInclusivePercentage () {
                return funcData.getInclusivePercentage();
        }

        public double getInclusive () {
                return funcData.getInclusive();
        }

		public double getExclusivePercentage () {
                return funcData.getExclusivePercentage();
        }

        public double getExclusive () {
                return funcData.getExclusive();
        }

        public double getInclusivePerCall () {
                return funcData.getInclusivePerCall();
        }
}
