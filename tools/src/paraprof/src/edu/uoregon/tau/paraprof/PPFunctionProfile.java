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
import edu.uoregon.tau.paraprof.enums.*;

public class PPFunctionProfile implements Comparable {

    private DataSorter dataSorter;
    private edu.uoregon.tau.dms.dss.Thread thread;

    private FunctionProfile functionProfile;
    private FunctionProfile meanProfile;

    // drawing coordinates for this object
    private int xBeg = 0;
    private int xEnd = 0;
    private int yBeg = 0;
    private int yEnd = 0;

    public PPFunctionProfile(DataSorter dataSorter, edu.uoregon.tau.dms.dss.Thread thread, FunctionProfile fp) {
        this.dataSorter = dataSorter;
        this.thread = thread;
        this.functionProfile = fp;

        // prefetch this
        this.meanProfile = functionProfile.getFunction().getMeanProfile();
    }

    public FunctionProfile getMeanProfile() {
        return meanProfile;
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
        if (ParaProf.preferences.getReversedCallPaths()) {
            return functionProfile.getFunction().getReversedName();
        } else {
            return functionProfile.getFunction().getName();
        }
    }

    public String getFunctionNameReversedCallPath() {
        return functionProfile.getFunction().getReversedName();
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
        return functionProfile.getInclusive(dataSorter.getSelectedMetricID());
    }

    public double getExclusiveValue() {
        return functionProfile.getExclusive(dataSorter.getSelectedMetricID());
    }

    public double getInclusivePercentValue() {
        return functionProfile.getInclusivePercent(dataSorter.getSelectedMetricID());
    }

    public double getExclusivePercentValue() {
        return functionProfile.getExclusivePercent(dataSorter.getSelectedMetricID());
    }

    public double getNumberOfCalls() {
        return functionProfile.getNumCalls();
    }

    public double getNumberOfSubRoutines() {
        return functionProfile.getNumSubr();
    }

    public double getInclusivePerCall() {
        return functionProfile.getInclusivePerCall(dataSorter.getSelectedMetricID());
    }

    //Parent/child interface.

    public Iterator getChildProfiles() {
        return functionProfile.getChildProfiles();
    }

    public Iterator getParentProfiles() {
        return functionProfile.getParentProfiles();
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    public double getValue() {
        return dataSorter.getValueType().getValue(this.getFunctionProfile(), dataSorter.getSelectedMetricID());
    }

    //   

    private int checkDescending(int value) {
        if (dataSorter.getDescendingOrder())
            return -value;
        return value;
    }

    public int compareTo(Object inObject) {
        ValueType valueType = dataSorter.getValueType();

        PPFunctionProfile other = (PPFunctionProfile) inObject;

        if (dataSorter.getSortType() == SortType.NAME) {
            return checkDescending(other.getFunctionName().compareTo(this.getFunctionName()));
        } else if (dataSorter.getSortType() == SortType.NCT) {
            if (other.getNodeID() != this.getNodeID())
                return checkDescending(this.getNodeID() - other.getNodeID());
            else if (other.getContextID() != this.getContextID())
                return checkDescending(this.getContextID() - other.getContextID());
            else
                return checkDescending(this.getThreadID() - other.getThreadID());
        } else if (dataSorter.getSortType() == SortType.MEAN_VALUE) {

            return checkDescending(compareToHelper(valueType.getValue(this.meanProfile,
                    dataSorter.getSelectedMetricID()), valueType.getValue(other.meanProfile,
                    dataSorter.getSelectedMetricID()), this.meanProfile, other.meanProfile));

        } else if (dataSorter.getSortType() == SortType.VALUE) {
            return checkDescending(compareToHelper(valueType.getValue(this.getFunctionProfile(),
                    dataSorter.getSelectedMetricID()), valueType.getValue(other.getFunctionProfile(),
                    dataSorter.getSelectedMetricID())));
        } else {
            throw new ParaProfException("Unexpected sort type: " + dataSorter.getSortType());
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

    private int compareToHelper(double d1, double d2, FunctionProfile f1, FunctionProfile f2) {
        double result = d1 - d2;
        if (result < 0.00)
            return -1;
        else if (result == 0.00) {
            // this is here to make sure that things get sorted the same for mean and other threads
            // in the case of callpath profiles, multiple functionProfiles may have the same values
            // we need them in the same order for everyone
            return f1.getFunction().compareTo(f2.getFunction());
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

    public String getStatString(int type) {

        int metric = dataSorter.getSelectedMetricID();
        String tmpString;

        DecimalFormat dF = new DecimalFormat("##0.0");
        tmpString = UtilFncs.lpad(dF.format(functionProfile.getInclusivePercent(metric)), 13);

        tmpString = tmpString + "  " + UtilFncs.getOutputString(type, functionProfile.getExclusive(metric), 14);
        tmpString = tmpString + "  " + UtilFncs.getOutputString(type, functionProfile.getInclusive(metric), 16);
        tmpString = tmpString + "  " + UtilFncs.formatDouble(functionProfile.getNumCalls(), 12, true);
        tmpString = tmpString + "  " + UtilFncs.formatDouble(functionProfile.getNumSubr(), 12, true);
        tmpString = tmpString + "  " + UtilFncs.getOutputString(type, functionProfile.getInclusivePerCall(metric), 19);

        //Everything should be added now except the function name.
        return tmpString;
    }

    // Static Functions

    public static String getStatStringHeading(String metricType) {
        //        return UtilFncs.lpad("%Total " + metricType, 13) + UtilFncs.lpad(metricType, 16)
        //        + UtilFncs.lpad("Total " + metricType, 18) + UtilFncs.lpad("#Calls", 14)
        //        + UtilFncs.lpad("#Child Calls", 14) + UtilFncs.lpad("Total " + metricType + "/Call", 21) + "   ";
        return UtilFncs.lpad("%Total " + metricType, 13) + UtilFncs.lpad("Exclusive", 16)
                + UtilFncs.lpad("Inclusive", 18) + UtilFncs.lpad("#Calls", 14) + UtilFncs.lpad("#Child Calls", 14)
                + UtilFncs.lpad("Inclusive/Call", 21) + "   ";
    }

    public String toString() {
        return functionProfile.toString();
    }

    public edu.uoregon.tau.dms.dss.Thread getThread() {
        return thread;
    }

}