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
 *  
 *  NOTE: This class will soon be obsolete and be removed as it is an unnecessary waste of memory
 *  
 */

package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.text.DecimalFormat;
import java.util.Iterator;

import edu.uoregon.tau.common.AlphanumComparator;
import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Group;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UtilFncs;

public class PPFunctionProfile implements Comparable<PPFunctionProfile> {

    private DataSorter dataSorter;
    private Thread thread;
    private FunctionProfile functionProfile;
    private static AlphanumComparator alphanum = new AlphanumComparator();

    public PPFunctionProfile(DataSorter dataSorter, Thread thread, FunctionProfile fp) {
        this.dataSorter = dataSorter;
        this.thread = thread;
        this.functionProfile = fp;
    }

    public FunctionProfile getMeanProfile() {
        return functionProfile.getFunction().getMeanProfile();
    }

    public int getID() {
        return functionProfile.getFunction().getID();
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

    public ParaProfTrial getPPTrial() {
        return dataSorter.getPpTrial();
    }

    // handles reversed callpaths
    public String getDisplayName() {
        return ParaProfUtils.getDisplayName(functionProfile.getFunction());
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
        return functionProfile.getInclusive(dataSorter.getSelectedMetric().getID());
    }

    public double getExclusiveValue() {
        if (functionProfile.getFunction().isPhase() && functionProfile.getFunction().isCallPathFunction()) {
            return functionProfile.getInclusive(dataSorter.getSelectedMetric().getID());
        } else {
            return functionProfile.getExclusive(dataSorter.getSelectedMetric().getID());
        }
    }

    public double getInclusivePercentValue() {
        return functionProfile.getInclusivePercent(dataSorter.getSelectedMetric().getID());
    }

    public double getExclusivePercentValue() {
        return functionProfile.getExclusivePercent(dataSorter.getSelectedMetric().getID());
    }

    public double getNumberOfCalls() {
        return functionProfile.getNumCalls();
    }

    public double getNumberOfSubRoutines() {
        return functionProfile.getNumSubr();
    }

    public double getInclusivePerCall() {
        return functionProfile.getInclusivePerCall(dataSorter.getSelectedSnapshot(), dataSorter.getSelectedMetric().getID());
    }

    public Iterator<FunctionProfile> getChildProfiles() {
        return functionProfile.getChildProfiles();
    }

    public Iterator<FunctionProfile> getParentProfiles() {
        return functionProfile.getParentProfiles();
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    public double getValue() {
        int differential = 0;
        int snap = dataSorter.getSelectedSnapshot();
        if (differential == 1 && snap > 0) {
            double value1 = dataSorter.getValueType().getValue(this.getFunctionProfile(), dataSorter.getSelectedMetric().getID(),
                    snap - 1);
            double value2 = dataSorter.getValueType().getValue(this.getFunctionProfile(), dataSorter.getSelectedMetric().getID(), snap);
            return value2 - value1;
        } else {
            return dataSorter.getValueType().getValue(this.getFunctionProfile(), dataSorter.getSelectedMetric().getID(),
                    dataSorter.getSelectedSnapshot());
        }

    }

    public double getSortValue() {
        int differential = 0;
        int snap = dataSorter.getSelectedSnapshot();
        if (differential == 1 && snap > 0) {
            double value1 = dataSorter.getSortValueType().getValue(this.getFunctionProfile(), dataSorter.getSortMetric().getID(),
                    snap - 1);
            double value2 = dataSorter.getSortValueType().getValue(this.getFunctionProfile(), dataSorter.getSortMetric().getID(),
                    snap);
            return value2 - value1;
        } else { 
            return dataSorter.getSortValueType().getValue(this.getFunctionProfile(), dataSorter.getSortMetric(),
                    dataSorter.getSelectedSnapshot());
        }
    }

    private int checkDescending(int value) {
        if (dataSorter.getDescendingOrder()) {
            return -value;
        }
        return value;
    }

    private int compareNCT(PPFunctionProfile other) {
        if (other.getNodeID() != this.getNodeID()) {
            return checkDescending(this.getNodeID() - other.getNodeID());
        } else if (other.getContextID() != this.getContextID()) {
            return checkDescending(this.getContextID() - other.getContextID());
        } else {
            return checkDescending(this.getThreadID() - other.getThreadID());
        }
    }

    public int compareTo(PPFunctionProfile inObject) {
        ValueType valueType = dataSorter.getSortValueType();

        PPFunctionProfile other = (PPFunctionProfile) inObject;

        //TODO: I think valueType might need to be sortValueType???
        if (dataSorter.getSortType() == SortType.NAME) {
            //return checkDescending(other.getDisplayName().compareTo(this.getDisplayName()));
            return checkDescending(alphanum.compare(other.getDisplayName(), this.getDisplayName()));

        } else if (dataSorter.getSortType() == SortType.NCT) {
            return compareNCT(other);

        } else if (dataSorter.getSortType() == SortType.MEAN_VALUE) {
            return checkDescending(compareToHelper(valueType.getValue(this.getMeanProfile(), dataSorter.getSortMetric(),
                    dataSorter.getSelectedSnapshot()), valueType.getValue(other.getMeanProfile(), dataSorter.getSortMetric(),
                    dataSorter.getSelectedSnapshot()), this.getMeanProfile(), other.getMeanProfile()));

        } else if (dataSorter.getSortType() == SortType.VALUE) {
            return checkDescending(compareToHelper(getSortValue(), other.getSortValue()));
            //            return checkDescending(compareToHelper(valueType.getValue(this.getFunctionProfile(), dataSorter.getSortMetric(), dataSorter.getSelectedSnapshot()),
            //                    valueType.getValue(other.getFunctionProfile(), dataSorter.getSortMetric(), dataSorter.getSelectedSnapshot())));
        } else {
            throw new ParaProfException("Unexpected sort type: " + dataSorter.getSortType());
        }
    }

    private int compareToHelper(double d1, double d2) {
        double result = d1 - d2;
        if (result < 0.00) {
            return -1;
        } else if (result == 0.00) {
            return 0;
        } else {
            return 1;
        }
    }

    private int compareToHelper(double d1, double d2, FunctionProfile f1, FunctionProfile f2) {
        double result = d1 - d2;
        if (result < 0.00) {
            return -1;
        } else if (result == 0.00) {
            // this is here to make sure that things get sorted the same for mean and other threads
            // in the case of callpath profiles, multiple functionProfiles may have the same values
            // we need them in the same order for everyone
            return f1.getFunction().compareTo(f2.getFunction());
        } else {
            return 1;
        }
    }

    public String getStatString(int type) {
        int metric = dataSorter.getSelectedMetric().getID();
        String tmpString;
        
        boolean timeDenominator = dataSorter.getSelectedMetric().isTimeDenominator();

        DecimalFormat dF = new DecimalFormat("##0.0");
        tmpString = UtilFncs.lpad(dF.format(functionProfile.getInclusivePercent(dataSorter.getSelectedSnapshot(), metric)), 13);

        if (functionProfile.getFunction().isPhase() && functionProfile.getFunction().isCallPathFunction()) {
            tmpString = tmpString + "  "
                    + UtilFncs.getOutputString(type, functionProfile.getInclusive(dataSorter.getSelectedSnapshot(), metric), 14, timeDenominator);
        } else {
            tmpString = tmpString + "  "
                    + UtilFncs.getOutputString(type, functionProfile.getExclusive(dataSorter.getSelectedSnapshot(), metric), 14, timeDenominator);
        }
        tmpString = tmpString + "  "
                + UtilFncs.getOutputString(type, functionProfile.getInclusive(dataSorter.getSelectedSnapshot(), metric), 16, timeDenominator);
        tmpString = tmpString + "  "
                + UtilFncs.formatDouble(functionProfile.getNumCalls(dataSorter.getSelectedSnapshot()), 12, true);
        tmpString = tmpString + "  "
                + UtilFncs.formatDouble(functionProfile.getNumSubr(dataSorter.getSelectedSnapshot()), 12, true);
        tmpString = tmpString
                + "  "
                + UtilFncs.getOutputString(type, functionProfile.getInclusivePerCall(dataSorter.getSelectedSnapshot(), metric),
                        19, timeDenominator);

        //Everything should be added now except the function name.
        return tmpString;
    }

    public String toString() {
        return functionProfile.toString();
    }

    public Thread getThread() {
        return thread;
    }

    public static String getStatStringHeading(String metricType) {
        return UtilFncs.lpad("%Total " + metricType, 13) + UtilFncs.lpad("Exclusive", 16) + UtilFncs.lpad("Inclusive", 18)
                + UtilFncs.lpad("#Calls", 14) + UtilFncs.lpad("#Child Calls", 14) + UtilFncs.lpad("Inclusive/Call", 21) + "   ";
    }

}