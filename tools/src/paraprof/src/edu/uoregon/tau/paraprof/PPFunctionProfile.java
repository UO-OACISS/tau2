/*
 * PPFunctionProfile.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description: The primary function of this class are: 
 * 1) Pass data calls onto the objects which contain function userevent, mean, and other data. 
 * 2) Implement the Comparable interface to allow it to be sorted. 
 * 3) Hold drawing information.
 * 
 * Thus, it can be considered loosly as representing a particular object that
 * will be drawn onto the screen at some point. It is not set up to represent
 * MULTIPLE occurrences of drawing or sorting information. That is, it can hold
 * only one set of drawing and sorting data at a time. Different windows must
 * create their own instances of this object to avoid conflicts.
 *  
 */

package edu.uoregon.tau.paraprof;

import java.text.DecimalFormat;
import java.util.*;
import java.awt.*;
import edu.uoregon.tau.dms.dss.*;

public class PPFunctionProfile implements Comparable {

    private ParaProfTrial trial = null;
    private edu.uoregon.tau.dms.dss.Thread thread;

    private FunctionProfile functionProfile;
    private FunctionProfile meanProfile;
    private int sortType;

    // drawing coordinates for this object
    private int xBeg = 0;
    private int xEnd = 0;
    private int yBeg = 0;
    private int yEnd = 0;


    public PPFunctionProfile(ParaProfTrial trial, edu.uoregon.tau.dms.dss.Thread thread, FunctionProfile fp) {
        this.trial = trial;
        this.thread = thread;
        this.functionProfile = fp;

        // prefetch this
        this.meanProfile = functionProfile.getFunction().getMeanProfile();
    }

    public int getNodeID() {
        return thread.getNodeID();
    }

    public int getContextID() {
        return thread.getContextID();
    }

    public int getThreadID() {
        return thread.getThreadID();
    }

    public FunctionProfile getFunctionProfile() {
        return functionProfile;
    }

    public Function getFunction() {
        return functionProfile.getFunction();
    }

    public String getFunctionName() {
        return functionProfile.getFunction().getName();
    }

    public Color getColor() {
        return functionProfile.getFunction().getColor();
    }

    public boolean isGroupMember(Group group) {
        return functionProfile.getFunction().isGroupMember(group);
    }

    public boolean isCallPathObject() {
        return functionProfile.getFunction().isCallPathFunction();
    }

    public double getInclusiveValue() {
        return functionProfile.getInclusive(trial.getSelectedMetricID());
    }

    public double getExclusiveValue() {
        return functionProfile.getExclusive(trial.getSelectedMetricID());
    }

    public double getInclusivePercentValue() {
        return functionProfile.getInclusivePercent(trial.getSelectedMetricID());
    }

    public double getExclusivePercentValue() {
        return functionProfile.getExclusivePercent(trial.getSelectedMetricID());
    }

    public double getNumberOfCalls() {
        return functionProfile.getNumCalls();
    }

    public double getNumberOfSubRoutines() {
        return functionProfile.getNumSubr();
    }

    public double getInclusivePerCall() {
        return functionProfile.getInclusivePerCall(trial.getSelectedMetricID());
    }

    //Parent/child interface.
       
    public Iterator getChildProfiles() {
        return functionProfile.getChildProfiles();
    }
    
    public Iterator getParentProfiles() {
        return functionProfile.getParentProfiles();
    }
    
    /*
     * (0) name 
     * (2) exclusive 
     * (4) inclusive 
     * (6) number of calls 
     * (8) number of subroutines 
     * (10) per call value 
     * (12) userevent number value 
     * (14) userevent min value 
     * (16) userevent max value 
     * (18) userevent mean value
     * (20) mean exclusive 
     * (22) mean inclusive 
     * (24) mean number of calls 
     * (26) mean number of subroutines 
     * (28) mean per call value 
     * (30) n,c,t.
     * 
     * The even values represent these items sorted in descending order, the odd
     * values in ascending order. Thus (0) is name descending, and (1) is name
     * ascending. Set sortType to the integer value required.
     */

    public int compareTo(Object inObject) {
        switch (sortType) {
        case 0:
            return (((PPFunctionProfile) inObject).getFunctionName()).compareTo(this.getFunctionName());
        case 1:
            return (this.getFunctionName()).compareTo(((PPFunctionProfile) inObject).getFunctionName());
        case 2:
            return compareToHelper(((PPFunctionProfile) inObject).getExclusiveValue(), this.getExclusiveValue());
        case 3:
            return compareToHelper(this.getExclusiveValue(), ((PPFunctionProfile) inObject).getExclusiveValue());
        case 4:
            return compareToHelper(((PPFunctionProfile) inObject).getInclusiveValue(), this.getInclusiveValue());
        case 5:
            return compareToHelper(this.getInclusiveValue(), ((PPFunctionProfile) inObject).getInclusiveValue());
        case 6:
            return compareToHelper(((PPFunctionProfile) inObject).getNumberOfCalls(), this.getNumberOfCalls());
        case 7:
            return compareToHelper(this.getNumberOfCalls(), ((PPFunctionProfile) inObject).getNumberOfCalls());
        case 8:
            return compareToHelper(((PPFunctionProfile) inObject).getNumberOfSubRoutines(),
                    this.getNumberOfSubRoutines());
        case 9:
            return compareToHelper(this.getNumberOfSubRoutines(),
                    ((PPFunctionProfile) inObject).getNumberOfSubRoutines());
        case 10:
            return compareToHelper(((PPFunctionProfile) inObject).getInclusivePerCall(),
                    this.getInclusivePerCall());
        case 11:
            return compareToHelper(this.getInclusivePerCall(),
                    ((PPFunctionProfile) inObject).getInclusivePerCall());
        case 20:
            return compareToHelper(
                    ((PPFunctionProfile) inObject).meanProfile.getExclusive(trial.getSelectedMetricID()),
                    this.meanProfile.getExclusive(trial.getSelectedMetricID()), this,
                    (PPFunctionProfile) inObject);
        case 21:
            return compareToHelper(this.meanProfile.getExclusive(trial.getSelectedMetricID()),
                    ((PPFunctionProfile) inObject).meanProfile.getExclusive(trial.getSelectedMetricID()), this,
                    (PPFunctionProfile) inObject);
        case 22:
            return compareToHelper(
                    ((PPFunctionProfile) inObject).meanProfile.getInclusive(trial.getSelectedMetricID()),
                    this.meanProfile.getInclusive(trial.getSelectedMetricID()));
        case 23:
            return compareToHelper(this.meanProfile.getInclusive(trial.getSelectedMetricID()),
                    ((PPFunctionProfile) inObject).meanProfile.getInclusive(trial.getSelectedMetricID()));
        case 24:
            return compareToHelper(((PPFunctionProfile) inObject).meanProfile.getNumCalls(),
                    this.meanProfile.getNumCalls());
        case 25:
            return compareToHelper(this.meanProfile.getNumCalls(),
                    ((PPFunctionProfile) inObject).meanProfile.getNumCalls());
        case 26:
            return compareToHelper(((PPFunctionProfile) inObject).meanProfile.getNumSubr(),
                    this.meanProfile.getNumSubr());
        case 27:
            return compareToHelper(this.meanProfile.getNumSubr(),
                    ((PPFunctionProfile) inObject).meanProfile.getNumSubr());
        case 28:
            return compareToHelper(
                    ((PPFunctionProfile) inObject).meanProfile.getInclusivePerCall(trial.getSelectedMetricID()),
                    this.meanProfile.getInclusivePerCall(trial.getSelectedMetricID()));
        case 29:
            return compareToHelper(this.meanProfile.getInclusivePerCall(trial.getSelectedMetricID()),
                    ((PPFunctionProfile) inObject).meanProfile.getInclusivePerCall(trial.getSelectedMetricID()));
        case 30:
            PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) inObject;
            if (ppFunctionProfile.getNodeID() != this.getNodeID())
                return ppFunctionProfile.getNodeID() - this.getNodeID();
            else if (ppFunctionProfile.getContextID() != this.getContextID())
                return ppFunctionProfile.getContextID() - this.getContextID();
            else
                return ppFunctionProfile.getThreadID() - this.getThreadID();
        case 31:
            ppFunctionProfile = (PPFunctionProfile) inObject;
            if (ppFunctionProfile.getNodeID() != this.getNodeID())
                return this.getNodeID() - ppFunctionProfile.getNodeID();
            else if (ppFunctionProfile.getContextID() != this.getContextID())
                return this.getContextID() - ppFunctionProfile.getContextID();
            else
                return this.getThreadID() - ppFunctionProfile.getThreadID();
        default:
            throw new ParaProfException("Unexpected sort type: " + sortType);
        }
    }

    private int compareToHelper(double d1, double d2) {
        double result = d1 - d2;
        if (result < 0.00)
            return -1;
        else if (result == 0.00)
            return 0;
        else
            return 1;
    }

    private int compareToHelper(double d1, double d2, PPFunctionProfile f1, PPFunctionProfile f2) {
        double result = d1 - d2;
        if (result < 0.00)
            return -1;
        else if (result == 0.00) {
            // this is here to make sure that things get sorted the same for mean and other threads
            // in the case of callpath profiles, multiple functionProfiles may have the same values
            // we need them in the same order for everyone
            return f1.functionProfile.getFunction().compareTo(f2.functionProfile.getFunction());
        } else
            return 1;
    }

    public void setDrawCoords(int xBeg, int xEnd, int yBeg, int yEnd) {
        this.xBeg = xBeg;
        this.xEnd = xEnd;
        this.yBeg = yBeg;
        this.yEnd = yEnd;
    }

    public int getXBeg() {
        return xBeg;
    }

    public int getXEnd() {
        return xEnd;
    }

    public int getYBeg() {
        return yBeg;
    }

    public int getYEnd() {
        return yEnd;
    }

    public void setSortType(int sortType) {
        this.sortType = sortType;
    }

    public ParaProfTrial getTrial() {
        return trial;
    }

    public String getStatString(int type) {

        int metric = trial.getSelectedMetricID();
        String tmpString;

        DecimalFormat dF = new DecimalFormat("##0.0");
        tmpString = UtilFncs.lpad(dF.format(functionProfile.getInclusivePercent(metric)), 13);

        tmpString = tmpString + "  " + UtilFncs.getOutputString(type, functionProfile.getExclusive(metric), 14);
        tmpString = tmpString + "  " + UtilFncs.getOutputString(type, functionProfile.getInclusive(metric), 16);
        tmpString = tmpString + "  " + UtilFncs.formatDouble(functionProfile.getNumCalls(), 12);
        tmpString = tmpString + "  " + UtilFncs.formatDouble(functionProfile.getNumSubr(), 12);
        tmpString = tmpString + "  "
                + UtilFncs.getOutputString(type, functionProfile.getInclusivePerCall(metric), 19);

        //Everything should be added now except the function name.
        return tmpString;
    }

    // Static Functions

    public static String getStatStringHeading(String metricType) {
        return UtilFncs.lpad("%Total " + metricType, 13) + UtilFncs.lpad(metricType, 16)
                + UtilFncs.lpad("Total " + metricType, 18) + UtilFncs.lpad("#Calls", 14)
                + UtilFncs.lpad("#Subrs", 14) + UtilFncs.lpad("Total " + metricType + "/Call", 21) + "   ";
    }

    
}