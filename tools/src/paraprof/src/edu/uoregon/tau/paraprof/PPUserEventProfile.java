/*
 * PPFunctionProfile.java
 * 
 * Title: ParaProf Author: Robert Bell Description: The primary functionProfiles of
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

import java.awt.*;
import edu.uoregon.tau.dms.dss.*;

public class PPUserEventProfile implements Comparable {
    public PPUserEventProfile(ParaProfTrial trial, int nodeID, int contextID, int threadID,
            UserEventProfile userEventProfile) {

        this.nodeID = nodeID;
        this.contextID = contextID;
        this.threadID = threadID;

        this.userEventProfile = userEventProfile;
        this.userEvent = userEventProfile.getUserEvent();
    }

    public int getNodeID() {
        return nodeID;
    }

    public int getContextID() {
        return contextID;
    }

    public int getThreadID() {
        return threadID;
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

    public int getUserEventNumberValue() {
        return userEventProfile.getUserEventNumberValue();
    }

    public double getUserEventMinValue() {
        return userEventProfile.getUserEventMinValue();
    }

    public double getUserEventMaxValue() {
        return userEventProfile.getUserEventMaxValue();
    }

    public double getUserEventMeanValue() {
        return userEventProfile.getUserEventMeanValue();
    }

    public double getStdDev() {
        return userEventProfile.getUserEventStdDev();
    }

    public String getUserEventStatString(int precision) {
            int initialBufferLength = 90;
            int position = 0;
            char[] statStringArray = new char[initialBufferLength];
            char[] tmpArray;
            String tmpString;

            PPUserEventProfile.insertSpaces(statStringArray, 0, 90);

            tmpArray = (Integer.toString(this.getUserEventNumberValue()).toCharArray());
            for (int i = 0; i < tmpArray.length; i++) {
                statStringArray[position] = tmpArray[i];
                position++;
            }


            position = 18;
            tmpString = UtilFncs.getOutputString(0, this.getUserEventMaxValue(), precision);
            tmpArray = tmpString.toCharArray();
            for (int i = 0; i < tmpArray.length; i++) {
                statStringArray[position] = tmpArray[i];
                position++;
            }

            position = 36;
            tmpString = UtilFncs.getOutputString(0, this.getUserEventMinValue(), precision);
            tmpArray = tmpString.toCharArray();
            for (int i = 0; i < tmpArray.length; i++) {
                statStringArray[position] = tmpArray[i];
                position++;
            }

            position = 54;
            tmpString = UtilFncs.getOutputString(0, this.getUserEventMeanValue(), precision);
            tmpArray = tmpString.toCharArray();
            for (int i = 0; i < tmpArray.length; i++) {
                statStringArray[position] = tmpArray[i];
                position++;
            }

            position = 72;
            tmpString = UtilFncs.getOutputString(0, this.getStdDev(), precision);
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

    //####################################
    //End - Userevent interface.
    //####################################

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

    public int compareTo(Object inObject) {
        switch (sortType) {
        case 0:
            return (((PPUserEventProfile) inObject).getUserEventName()).compareTo(this.getUserEventName());
        case 1:
            return (this.getUserEventName()).compareTo(((PPUserEventProfile) inObject).getUserEventName());
        case 12:
            return compareToHelper(((PPUserEventProfile) inObject).getUserEventNumberValue(),
                    this.getUserEventNumberValue());
        case 13:
            return compareToHelper(this.getUserEventNumberValue(),
                    ((PPUserEventProfile) inObject).getUserEventNumberValue());
        case 14:
            return compareToHelper(((PPUserEventProfile) inObject).getUserEventMinValue(),
                    this.getUserEventMinValue());
        case 15:
            return compareToHelper(this.getUserEventMinValue(),
                    ((PPUserEventProfile) inObject).getUserEventMinValue());
        case 16:
            return compareToHelper(((PPUserEventProfile) inObject).getUserEventMaxValue(),
                    this.getUserEventMaxValue());
        case 17:
            return compareToHelper(this.getUserEventMaxValue(),
                    ((PPUserEventProfile) inObject).getUserEventMaxValue());
        case 18:
            return compareToHelper(((PPUserEventProfile) inObject).getUserEventMeanValue(),
                    this.getUserEventMeanValue());
        case 19:
            return compareToHelper(this.getUserEventMeanValue(),
                    ((PPUserEventProfile) inObject).getUserEventMeanValue());
        case 20:
            return compareToHelper(this.getStdDev(), ((PPUserEventProfile) inObject).getStdDev());
        case 21:
            return compareToHelper(this.getUserEventMeanValue(),
                    ((PPUserEventProfile) inObject).getUserEventMeanValue());

        case 30:
            PPUserEventProfile ppUserEventProfile = (PPUserEventProfile) inObject;
            if (ppUserEventProfile.getNodeID() != this.getNodeID())
                return ppUserEventProfile.getNodeID() - this.getNodeID();
            else if (ppUserEventProfile.getContextID() != this.getContextID())
                return ppUserEventProfile.getContextID() - this.getContextID();
            else
                return ppUserEventProfile.getThreadID() - this.getThreadID();
        case 31:
            ppUserEventProfile = (PPUserEventProfile) inObject;
            if (ppUserEventProfile.getNodeID() != this.getNodeID())
                return this.getNodeID() - ppUserEventProfile.getNodeID();
            else if (ppUserEventProfile.getContextID() != this.getContextID())
                return this.getContextID() - ppUserEventProfile.getContextID();
            else
                return this.getThreadID() - ppUserEventProfile.getThreadID();
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

    public boolean getStatDrawnTo() {
        return statDrawnTo;
    }

    public void setStatDrawnTo(boolean statDrawnTo) {
        this.statDrawnTo = statDrawnTo;
    }

    public void setHighlighted(boolean highlighted) {
        this.highlighted = highlighted;
    }

    public boolean isHighlighted() {
        return highlighted;
    }

    public void setSortType(int sortType) {
        this.sortType = sortType;
    }

    //Instance data.

    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;

    UserEventProfile userEventProfile;

    UserEvent userEvent = null;

    //Drawing coordinates for this thread data object.
    int xBeg = 0;
    int xEnd = 0;
    int yBeg = 0;
    int yEnd = 0;

    boolean statDrawnTo;

    //Boolean indicating whether or not this object is highlighted.
    boolean highlighted = false;

    int sortType;

}