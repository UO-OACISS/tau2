/*
 * Name: Thread.java 
 * Author: Robert Bell 
 * Description:
 */

package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.io.*;

public class Thread implements Comparable {

    public Thread() {
        doubleList = new double[Thread.arrayIncrementSize];
        this.numberOfMetrics = 1;
    }

    public Thread(int initialCapacity) {
        doubleList = new double[initialCapacity * Thread.arrayIncrementSize];
        this.numberOfMetrics = initialCapacity;
    }

    public Thread(int nodeID, int contextID, int threadID) {
        this.nodeID = nodeID;
        this.contextID = contextID;
        this.threadID = threadID;
        doubleList = new double[Thread.arrayIncrementSize];
        this.numberOfMetrics = 1;
    }

    public Thread(int nodeID, int contextID, int threadID, int initialCapacity) {
        this.nodeID = nodeID;
        this.contextID = contextID;
        this.threadID = threadID;
        doubleList = new double[initialCapacity * Thread.arrayIncrementSize];
        this.numberOfMetrics = initialCapacity;
    }

    //####################################
    //Public section.
    //####################################
    public void setNodeId(int nodeID) {
        this.nodeID = nodeID;
    }

    public int getNodeID() {
        return nodeID;
    }

    public void setContextID(int contextID) {
        this.contextID = contextID;
    }

    public int getContextID() {
        return contextID;
    }

    public void setThreadID(int threadID) {
        this.threadID = threadID;
    }

    public int getThreadID() {
        return threadID;
    }

    public void initializeFunctionList(int size) {
        functionProfiles = new Vector(size);
        Object ref = null;
        for (int i = 0; i < size; i++) {
            functionProfiles.add(ref);
        }
    }

    public void initializeUsereventList(int size) {
        userevents = new Vector(size);
        Object ref = null;
        for (int i = 0; i < size; i++) {
            userevents.add(ref);
        }
    }

    public void incrementStorage() {
        int currentLength = doubleList.length;
        //can use a little space here ... space for speed! :-)
        double[] newArray = new double[currentLength + Thread.arrayIncrementSize];

        for (int i = 0; i < currentLength; i++) {
            newArray[i] = doubleList[i];
        }
        doubleList = newArray;
        this.numberOfMetrics++;
    }

    public int getNumberOfMetrics() {
        return this.numberOfMetrics;
    }

    //Since name to id lookups do not occur at the thread level, we
    //store only in id order. If an id is added which is greater than
    //the current size of the Vector of functionProfiles, the vector will be
    //increased in size.
    public void addFunctionProfile(FunctionProfile ref, int id) {
        boolean added = false;
        //Increase the function list size if required.
        FunctionProfile placeHolder = null;
        while (id >= (functionProfiles.size())) {
            functionProfiles.add(placeHolder);
        }

        //It is now safe to add (but do not add if there is already
        //an element here.
        if ((functionProfiles.elementAt(id)) == null) {
            functionProfiles.setElementAt(ref, id);
            added = true;
        }

    }

    public void addUserEvent(UserEventProfile ref) {
        userevents.addElement(ref);
    }

    public void addUserEvent(UserEventProfile ref, int id) {
        //There are two paths here.
        //1) This id has not been seen in the system before.
        //   In this case, add to the end of functionProfiles.
        //2) The id has been seen in the system before.
        //   In this case, check to see if its location is
        //   not set to null in functionProfiles, and if it is not
        //   set the location to point to ref.
        boolean added = false;
        if (id >= (userevents.size())) {
            userevents.add(ref);
            added = true;
        } else {
            if ((userevents.elementAt(id)) == null)
                userevents.setElementAt(ref, id);
        }
    }

    public FunctionProfile getFunctionProfile(Function function) {
        FunctionProfile functionProfile = null;
        try {
            if ((functionProfiles != null) && (function.getID() < functionProfiles.size()))
                functionProfile = (FunctionProfile) functionProfiles.elementAt(function.getID());
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "T2");
        }
        return functionProfile;
    }

    public Vector getFunctionList() {
        return functionProfiles;
    }

    public ListIterator getFunctionListIterator() {
        return new DssIterator(functionProfiles);
    }

    public UserEventProfile getUserEvent(int id) {
        UserEventProfile functionProfile = null;
        try {
            if ((userevents != null) && (id < userevents.size()))
                functionProfile = (UserEventProfile) userevents.elementAt(id);
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "T3");
        }
        return functionProfile;
    }

    public Vector getUsereventList() {
        return userevents;
    }

    public ListIterator getUsereventListIterator() {
        return new DssIterator(userevents);
    }

    public void setThreadData(int metric) {
        if (this.debug) {
            this.outputDebugMessage("setThreadData\nmetrics:" + metric);
        }
        this.setThreadDataHelper(metric);
        this.setPercentData(metric);
        this.setThreadDataHelper(metric);
    }

    public void setThreadDataAllMetrics() {
        //This needs to be made more efficient (such as that used for setting
        // the mean values).
        if (this.debug) {
            this.outputDebugMessage("setThreadDataAllMetrics()\nnumberOfMetrics:"
                    + this.getNumberOfMetrics());
        }
        for (int i = 0; i < this.getNumberOfMetrics(); i++) {
            this.setThreadDataHelper(i);
            this.setPercentData(i);
            this.setThreadDataHelper(i);
        }
    }

    public void setMaxInclusive(int metric, double inDouble) {
        this.insertDouble(metric, 0, inDouble);
    }

    public double getMaxInclusive(int metric) {
        return this.getDouble(metric, 0);
    }

    public void setMaxExclusive(int metric, double inDouble) {
        this.insertDouble(metric, 1, inDouble);
    }

    public double getMaxExclusive(int metric) {
        return this.getDouble(metric, 1);
    }

    public void setMaxInclusivePercent(int metric, double inDouble) {
        this.insertDouble(metric, 2, inDouble);
    }

    public double getMaxInclusivePercent(int metric) {
        return this.getDouble(metric, 2);
    }

    public void setMaxExclusivePercent(int metric, double inDouble) {
        this.insertDouble(metric, 3, inDouble);
    }

    public double getMaxExclusivePercent(int metric) {
        return this.getDouble(metric, 3);
    }

    public void setMaxInclusivePerCall(int metric, double inDouble) {
        this.insertDouble(metric, 4, inDouble);
    }

    public double getMaxInclusivePerCall(int metric) {
        return this.getDouble(metric, 4);
    }

    public void setMaxNumCalls(double inDouble) {
        maxNumCalls = inDouble;
    }

    public double getMaxNumCalls() {
        return maxNumCalls;
    }

    public void setMaxNumSubr(double inDouble) {
        maxNumSubr = inDouble;
    }

    public double getMaxNumSubr() {
        return maxNumSubr;
    }

    // Since per thread callpath relations are build on demand, the following four functions tell whether this
    // thread's callpath information has been set yet.  This way, we only compute it once.
    public void setTrimmed(boolean b) {
        trimmed = b;
    }

    public boolean trimmed() {
        return trimmed;
    }

    public void setRelationsBuilt(boolean b) {
        relationsBuilt = b;
    }

    public boolean relationsBuilt() {
        return relationsBuilt;
    }

    public int compareTo(Object obj) {
        if (obj instanceof Integer)
            return threadID - ((Integer) obj).intValue();
        else
            return threadID - ((Thread) obj).getThreadID();
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    public void outputDebugMessage(String debugMessage) {
        UtilFncs.objectDebug.outputToFile(this.toString() + "\n" + debugMessage);
    }

    public String toString() {
        return this.getClass().getName() + ": " + this.getNodeID() + "," + this.getContextID()
                + "," + this.getThreadID();
    }

    //####################################
    //End - Public section.
    //####################################

    //####################################
    //Private section.
    //####################################
    private void insertDouble(int metric, int offset, double inDouble) {
        int actualLocation = (metric * 5) + offset;
        try {
            doubleList[actualLocation] = inDouble;
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "GT01");
        }
    }

    private double getDouble(int metric, int offset) {
        int actualLocation = (metric * 5) + offset;
        try {
            return doubleList[actualLocation];
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "GT01");
        }
        return -1;
    }

    private void setThreadDataHelper(int metric) {
        double maxInclusive = 0.0;
        double maxExclusive = 0.0;
        double maxInclusivePercent = 0.0;
        double maxExclusivePercent = 0.0;
        double maxInclusivePerCall = 0.0;
        double maxNumCalls = 0;
        double maxNumSubr = 0;

        double d = 0.0;
        ListIterator l = this.getFunctionListIterator();

        while (l.hasNext()) {
            FunctionProfile functionProfile = (FunctionProfile) l.next();
            if (functionProfile != null) {
                d = functionProfile.getInclusive(metric);
                if (d > maxInclusive)
                    maxInclusive = d;
                d = functionProfile.getExclusive(metric);
                if (d > maxExclusive)
                    maxExclusive = d;
                d = functionProfile.getInclusivePercent(metric);
                if (d > maxInclusivePercent)
                    maxInclusivePercent = d;
                d = functionProfile.getExclusivePercent(metric);
                if (d > maxExclusivePercent)
                    maxExclusivePercent = d;
                d = functionProfile.getInclusivePerCall(metric);
                if (d > maxInclusivePerCall)
                    maxInclusivePerCall = d;
                d = functionProfile.getNumCalls();
                if (d > maxNumCalls)
                    maxNumCalls = d;
                d = functionProfile.getNumSubr();
                if (d > maxNumSubr)
                    maxNumSubr = d;
            }
        }

        this.setMaxInclusive(metric, maxInclusive);
        this.setMaxExclusive(metric, maxExclusive);
        this.setMaxInclusivePercent(metric, maxInclusivePercent);
        this.setMaxExclusivePercent(metric, maxExclusivePercent);
        this.setMaxInclusivePerCall(metric, maxInclusivePerCall);
        this.setMaxNumCalls(maxNumCalls);
        this.setMaxNumSubr(maxNumSubr);
    }

    private void setPercentData(int metric) {
        ListIterator l = this.getFunctionListIterator();
        while (l.hasNext()) {
            FunctionProfile functionProfile = (FunctionProfile) l.next();

            if (functionProfile != null) {
                Function function = functionProfile.getFunction();

                //Note: Assumption is made that the max inclusive value is the
                // value required to calculate
                //percentage (ie, divide by). Thus, we are assuming that the
                // sum of the exclusive
                //values is equal to the max inclusive value. This is a
                // reasonable assuption. This also gets
                //us out of sticky situations when call path data is present
                // (this skews attempts to calculate
                //the total exclusive value unless checks are made to ensure
                // that we do not include call path
                //objects).

                if (this.getNodeID() == -1) {
                    functionProfile.setExclusivePercent(metric,
                            function.getTotalExclusivePercent(metric));
                    functionProfile.setInclusivePercent(metric,
                            function.getTotalInclusivePercent(metric));
                } else {

                    double inclusiveMax = this.getMaxInclusive(metric);

                    double d1 = functionProfile.getExclusive(metric);
                    double d2 = functionProfile.getInclusive(metric);

                    if (inclusiveMax != 0) {
                        double result = (d1 / inclusiveMax) * 100.00;

                        functionProfile.setExclusivePercent(metric, result);
                        //Now do the function exclusive stuff.
                        if ((function.getMaxExclusivePercent(metric)) < result)
                            function.setMaxExclusivePercent(metric, result);

                        result = (d2 / inclusiveMax) * 100;

                        functionProfile.setInclusivePercent(metric, result);
                        //Now do the global mapping element exclusive stuff.
                        if ((function.getMaxInclusivePercent(metric)) < result)
                            function.setMaxInclusivePercent(metric, result);
                    }
                }
            }
        }
    }

    //####################################
    //End - Private section.
    //####################################

    //####################################
    //Instance data.
    //####################################
    int nodeID = -1;
    int contextID = -1;
    int threadID = -1;
    Vector functionProfiles = null;
    Vector userevents = null;
    private double[] doubleList;
    double totalExclusiveValue = 0;
    double totalInclusiveValue = 0;
    private double maxNumCalls = 0;
    private double maxNumSubr = 0;
    private boolean trimmed = false;
    private boolean relationsBuilt = false;

    private int numberOfMetrics = 0;
    private static int arrayIncrementSize = 7;

    private boolean debug = false;
    //When in debugging mode, this class can print a lot of data.
    //Initialized in this.setDebug(...).
    private PrintWriter out = null;
    //####################################
    //End - Instance data.
    //####################################
}