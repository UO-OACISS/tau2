/*
 * Name: FunctionProfile.java Author: Robert Bell Description:
 */

package edu.uoregon.tau.dms.dss;

import java.util.*;

public class UserEventProfile {

    public UserEventProfile(UserEvent userEvent) {
        doubleList = new double[4];
        this.userEvent = userEvent;
    }

    public UserEvent getUserEvent() {
        return userEvent;
    }

    public static int getPositionOfName() {
        return 103;
    }

    public void setUserEventNumberValue(int inInt) {
        userEventNumberValue = inInt;
    }

    public int getUserEventNumberValue() {
        return userEventNumberValue;
    }

    public void setUserEventMinValue(double inDouble) {
        doubleList[0] = inDouble;
    }

    public double getUserEventMinValue() {
        return doubleList[0];
    }

    public void setUserEventMaxValue(double inDouble) {
        doubleList[1] = inDouble;
    }

    public double getUserEventMaxValue() {
        return doubleList[1];
    }

    public void setUserEventMeanValue(double inDouble) {
        doubleList[2] = inDouble;
    }

    public double getUserEventMeanValue() {
        return doubleList[2];
    }

    public void setUserEventSumSquared(double inDouble) {
        doubleList[3] = inDouble;
    }

    public double getUserEventSumSquared() {
        return doubleList[3];
    }

    public static String getUserEventStatStringHeading() {

        int w = 18;
        return UtilFncs.pad("NumSamples", w) + UtilFncs.pad("Max", w) + UtilFncs.pad("Min", w)
                + UtilFncs.pad("Mean", w) + UtilFncs.pad("Std. Dev", w);

        // this is great fun to maintain, what is the point of this stuff?
        /*
         * try{
         * 
         * 
         * int width = 16; int initialBufferLength = 91; int position = 0; char []
         * statStringArray = new char[initialBufferLength]; char [] tmpArray;
         * String tmpString;
         * 
         * insertSpaces(statStringArray , 0, 91);
         * 
         * tmpArray = ("NumSamples").toCharArray(); for(int i=0;i
         * <tmpArray.length;i++){ statStringArray[position] = tmpArray[i];
         * position++; }
         * 
         * position = 18; tmpArray = ("MaxValue").toCharArray(); for(int i=0;i
         * <tmpArray.length;i++){ statStringArray[position] = tmpArray[i];
         * position++; }
         * 
         * position = 36; tmpArray = ("MinValue").toCharArray(); for(int i=0;i
         * <tmpArray.length;i++){ statStringArray[position] = tmpArray[i];
         * position++; }
         * 
         * position = 54; tmpArray = ("MeanValue").toCharArray(); for(int i=0;i
         * <tmpArray.length;i++){ statStringArray[position] = tmpArray[i];
         * position++; }
         * 
         * return new String(statStringArray); } catch(Exception e){
         * UtilFncs.systemError(e, null, "GTDE01"); } return "An error occurred
         * processing this string!";
         */
    }

    public String getUserEventStatString(int precision) {
        try {
            int initialBufferLength = 90;
            int position = 0;
            char[] statStringArray = new char[initialBufferLength];
            char[] tmpArray;
            String tmpString;

            UserEventProfile.insertSpaces(statStringArray, 0, 90);

            tmpArray = (Integer.toString(this.getUserEventNumberValue()).toCharArray());
            for (int i = 0; i < tmpArray.length; i++) {
                statStringArray[position] = tmpArray[i];
                position++;
            }

            position = 18;
            tmpString = new String(Double.toString(UtilFncs.adjustDoublePresision(
                    this.getUserEventMaxValue(), precision)));
            tmpArray = tmpString.toCharArray();
            for (int i = 0; i < tmpArray.length; i++) {
                statStringArray[position] = tmpArray[i];
                position++;
            }

            position = 36;
            tmpString = new String(Double.toString(UtilFncs.adjustDoublePresision(
                    this.getUserEventMinValue(), precision)));
            tmpArray = tmpString.toCharArray();
            for (int i = 0; i < tmpArray.length; i++) {
                statStringArray[position] = tmpArray[i];
                position++;
            }

            position = 54;
            tmpString = new String(Double.toString(UtilFncs.adjustDoublePresision(
                    this.getUserEventMeanValue(), precision)));
            tmpArray = tmpString.toCharArray();
            for (int i = 0; i < tmpArray.length; i++) {
                statStringArray[position] = tmpArray[i];
                position++;
            }

            // compute the standard deviation
            // this should be computed just once, somewhere else
            // but it's here, for now
            double sumsqr = this.getUserEventSumSquared();
            double numEvents = this.getUserEventNumberValue();
            double mean = this.getUserEventMeanValue();

            double stddev = java.lang.Math.sqrt(java.lang.Math.abs((sumsqr / numEvents)
                    - (mean * mean)));

            position = 72;
            tmpString = new String(
                    Double.toString(UtilFncs.adjustDoublePresision(stddev, precision)));
            tmpArray = tmpString.toCharArray();
            for (int i = 0; i < tmpArray.length; i++) {
                statStringArray[position] = tmpArray[i];
                position++;
            }

            //Everything should be added now except the function name.
            return new String(statStringArray);
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "GTDE01");
        }

        return "An error occurred processing this string!";
    }

    public int getStorageSize() {
        return doubleList.length / 4;
    }

    public void incrementStorage() {
        if (userevent)
            UtilFncs.systemError(null, null,
                    "Error: Attempt to increase storage on a user event object!");
        int currentLength = doubleList.length;
        //can use a little space here ... space for speed! :-)
        double[] newArray = new double[currentLength + 5];

        for (int i = 0; i < currentLength; i++) {
            newArray[i] = doubleList[i];
        }
        doubleList = newArray;
    }

    //####################################
    //Private section.
    //####################################
    private void insertDouble(int metric, int offset, double inDouble) {
        int actualLocation = (metric * 5) + offset;
        try {
            doubleList[actualLocation] = inDouble;
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("inDouble: " + inDouble);
            System.out.println("metric: " + metric);
            System.out.println("offset: " + offset);
            System.out.println("actualLocation: " + actualLocation);
            System.out.println("doubleList size: " + doubleList.length);
            UtilFncs.systemError(e, null, "GTDE06");
        }
    }

    private double getDouble(int metric, int offset) {
        int actualLocation = (metric * 5) + offset;
        try {
            return doubleList[actualLocation];
        } catch (Exception e) {
            e.printStackTrace();
            e.printStackTrace();
            System.out.println("metric: " + metric);
            System.out.println("offset: " + offset);
            System.out.println("actualLocation: " + actualLocation);
            System.out.println("doubleList size: " + doubleList.length);
            UtilFncs.systemError(e, null, "GTDE06");
        }
        return -1;
    }

    private static int insertSpaces(char[] inArray, int position, int number) {
        for (int i = 0; i < number; i++) {
            inArray[position] = '\u0020';
            position++;
        }
        return position;
    }

    //####################################
    //End - Private section.
    //####################################

    //####################################
    //Instance data.
    //####################################
    private UserEvent userEvent = null;
    private boolean mappingExists = false;
    private double[] doubleList;
    private int numberOfCalls = 0;
    private int numberOfSubRoutines = 0;
    private int userEventNumberValue = 0;
    private boolean userevent = false;

    private Vector calls = null;

    private Vector parents = null;
    private Vector children = null;
    private Vector callPathIDSParents = null;
    private Vector callPathIDSChildren = null;
    //####################################
    //End - Instance data.
    //####################################
}