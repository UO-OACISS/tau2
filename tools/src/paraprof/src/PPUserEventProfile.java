/*
 * PPUserEventProfile.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description: The primary functions of
 * this class are: 1)Pass data calls onto the objects which contain function
 * userevent, mean, and other data. 2)Implement the Comparable interface to
 * allow it to be sorted. 3)Hold drawing information.
 * 
 * Thus, it can be considered loosly as representing a particular object that
 * will be drawn onto the screen at some point. It is not set up to represent
 * MULTIPLE occurrences of drawing or sorting information. That is, it can hold
 * only one set of drawing and sorting data at a time. Different windows must
 * create their own instances of this object to avoid conflicts.
 *  
 */

package edu.uoregon.tau.paraprof;

import java.awt.Color;

import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.UserEventValueType;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.perfdmf.Thread;


public class PPUserEventProfile implements Comparable {

    //Instance data.

    private UserEventProfile userEventProfile;
    private DataSorter dataSorter;
    private UserEvent userEvent;


    //Boolean indicating whether or not this object is highlighted.
    private boolean highlighted = false;

    private Thread thread;

    public PPUserEventProfile(DataSorter dataSorter, Thread thread,
            UserEventProfile userEventProfile) {
        this.thread = thread;

        this.dataSorter = dataSorter;

        this.userEventProfile = userEventProfile;
        this.userEvent = userEventProfile.getUserEvent();
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

    public UserEvent getUserEvent() {
        return userEvent;
    }

    public String getUserEventName() {
        return userEvent.getName();
    }

    public Color getColor() {
        return userEvent.getColor();
    }

    public UserEventProfile getUserEventProfile() {
        return userEventProfile;
    }

    public double getNumSamples() {
        return userEventProfile.getNumSamples(dataSorter.getSelectedSnapshot());
    }

    public double getMinValue() {
        return userEventProfile.getMinValue(dataSorter.getSelectedSnapshot());
    }

    public double getMaxValue() {
        return userEventProfile.getMaxValue(dataSorter.getSelectedSnapshot());
    }

    public double getMeanValue() {
        return userEventProfile.getMeanValue(dataSorter.getSelectedSnapshot());
    }

    public double getStdDev() {
        return userEventProfile.getStdDev(dataSorter.getSelectedSnapshot());
    }

    public String getUserEventStatString(int precision) {
        int initialBufferLength = 90;
        int position = 0;
        char[] statStringArray = new char[initialBufferLength];
        char[] tmpArray;
        String tmpString;

        PPUserEventProfile.insertSpaces(statStringArray, 0, 90);

        tmpString = UtilFncs.getOutputString(0, this.getNumSamples(), precision, false);
        tmpArray = tmpString.toCharArray();
        for (int i = 0; i < tmpArray.length; i++) {
            statStringArray[position] = tmpArray[i];
            position++;
        }

        position = 18;
        tmpString = UtilFncs.getOutputString(0, this.getMaxValue(), precision, false);
        tmpArray = tmpString.toCharArray();
        for (int i = 0; i < tmpArray.length; i++) {
            statStringArray[position] = tmpArray[i];
            position++;
        }

        position = 36;
        tmpString = UtilFncs.getOutputString(0, this.getMinValue(), precision, false);
        tmpArray = tmpString.toCharArray();
        for (int i = 0; i < tmpArray.length; i++) {
            statStringArray[position] = tmpArray[i];
            position++;
        }

        position = 54;
        tmpString = UtilFncs.getOutputString(0, this.getMeanValue(), precision, false);
        tmpArray = tmpString.toCharArray();
        for (int i = 0; i < tmpArray.length; i++) {
            statStringArray[position] = tmpArray[i];
            position++;
        }

        position = 72;
        tmpString = UtilFncs.getOutputString(0, this.getStdDev(), precision, false);
        tmpArray = tmpString.toCharArray();
        for (int i = 0; i < tmpArray.length; i++) {
            statStringArray[position] = tmpArray[i];
            position++;
        }

        //Everything should be added now except the function name.
        return new String(statStringArray);
    }

    private static int insertSpaces(char[] inArray, int position, int number) {
        for (int i = 0; i < number; i++) {
            inArray[position] = '\u0020';
            position++;
        }
        return position;
    }

    private int checkDescending(int value) {
        if (dataSorter.getDescendingOrder())
            return -value;
        return value;
    }

    public int compareTo(Object inObject) {
        UserEventValueType valueType = dataSorter.getUserEventValueType();

        PPUserEventProfile other = (PPUserEventProfile) inObject;

        if (dataSorter.getSortType() == SortType.NAME) {
            return checkDescending(other.getUserEventName().compareTo(this.getUserEventName()));

        } else if (dataSorter.getSortType() == SortType.NCT) {
            if (other.getNodeID() != this.getNodeID())
                return checkDescending(this.getNodeID() - other.getNodeID());
            else if (other.getContextID() != this.getContextID())
                return checkDescending(this.getContextID() - other.getContextID());
            else
                return checkDescending(this.getThreadID() - other.getThreadID());
            //        } else if (dataSorter.getSortType() == SortType.MEAN_VALUE) {
            //
            //            return checkDescending(Double.compare(valueType.getValue(other.meanProfile,
            //                    dataSorter.getSelectedMetricID()), valueType.getValue(this.meanProfile,
            //                    dataSorter.getSelectedMetricID())));
            //
        } else if (dataSorter.getSortType() == SortType.VALUE || dataSorter.getSortType() == SortType.MEAN_VALUE) {
            return checkDescending(compareToHelper(valueType.getValue(this.getUserEventProfile()),
                    valueType.getValue(other.getUserEventProfile())));

        } else {
            throw new ParaProfException("Unexpected sort type: " + dataSorter.getSortType());
        }
    }

    /*
     * (0) name (2) exclusive (4) inclusive (6) number of calls (8) number of
     * subroutines (10) per call value (12) userevent number value (14)
     * userevent min value (16) userevent max value (18) userevent mean value
     * (20) mean exclusive (22) mean inclusive (24) mean number of calls (26)
     * mean number of subroutines (28) mean per call value (30) n,c,t.
     * 
     * The even values represent these items sorted in descending order, the odd
     * values in ascending order. Thus (0) is name descending, and (1) is name
     * ascending. Set sortType to the integer value required.
     */

    //    public int compareTo(Object inObject) {
    //        switch (sortType) {
    //        case 0:
    //            return (((PPUserEventProfile) inObject).getUserEventName()).compareTo(this.getUserEventName());
    //        case 1:
    //            return (this.getUserEventName()).compareTo(((PPUserEventProfile) inObject).getUserEventName());
    //        case 12:
    //            return compareToHelper(((PPUserEventProfile) inObject).getUserEventNumberValue(),
    //                    this.getUserEventNumberValue());
    //        case 13:
    //            return compareToHelper(this.getUserEventNumberValue(),
    //                    ((PPUserEventProfile) inObject).getUserEventNumberValue());
    //        case 14:
    //            return compareToHelper(((PPUserEventProfile) inObject).getUserEventMinValue(),
    //                    this.getUserEventMinValue());
    //        case 15:
    //            return compareToHelper(this.getUserEventMinValue(),
    //                    ((PPUserEventProfile) inObject).getUserEventMinValue());
    //        case 16:
    //            return compareToHelper(((PPUserEventProfile) inObject).getUserEventMaxValue(),
    //                    this.getUserEventMaxValue());
    //        case 17:
    //            return compareToHelper(this.getUserEventMaxValue(),
    //                    ((PPUserEventProfile) inObject).getUserEventMaxValue());
    //        case 18:
    //            return compareToHelper(((PPUserEventProfile) inObject).getUserEventMeanValue(),
    //                    this.getUserEventMeanValue());
    //        case 19:
    //            return compareToHelper(this.getUserEventMeanValue(),
    //                    ((PPUserEventProfile) inObject).getUserEventMeanValue());
    //        case 20:
    //            return compareToHelper(this.getStdDev(), ((PPUserEventProfile) inObject).getStdDev());
    //        case 21:
    //            return compareToHelper(this.getUserEventMeanValue(),
    //                    ((PPUserEventProfile) inObject).getUserEventMeanValue());
    //
    //        case 30:
    //            PPUserEventProfile ppUserEventProfile = (PPUserEventProfile) inObject;
    //            if (ppUserEventProfile.getNodeID() != this.getNodeID())
    //                return ppUserEventProfile.getNodeID() - this.getNodeID();
    //            else if (ppUserEventProfile.getContextID() != this.getContextID())
    //                return ppUserEventProfile.getContextID() - this.getContextID();
    //            else
    //                return ppUserEventProfile.getThreadID() - this.getThreadID();
    //        case 31:
    //            ppUserEventProfile = (PPUserEventProfile) inObject;
    //            if (ppUserEventProfile.getNodeID() != this.getNodeID())
    //                return this.getNodeID() - ppUserEventProfile.getNodeID();
    //            else if (ppUserEventProfile.getContextID() != this.getContextID())
    //                return this.getContextID() - ppUserEventProfile.getContextID();
    //            else
    //                return this.getThreadID() - ppUserEventProfile.getThreadID();
    //        default:
    //            throw new ParaProfException("Unexpected sort type: " + sortType);
    //        }
    //    }
    private int compareToHelper(double d1, double d2) {
        double result = d1 - d2;
        if (result < 0.00)
            return -1;
        else if (result == 0.00)
            return 0;
        else
            return 1;
    }


    public void setHighlighted(boolean highlighted) {
        this.highlighted = highlighted;
    }

    public boolean isHighlighted() {
        return highlighted;
    }

    

    public Thread getThread() {
        return thread;
    }

}