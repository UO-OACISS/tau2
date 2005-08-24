package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.text.*;

/**
 * This class represents a single function profile on a single thread.
 *
 * <P>CVS $Id: FunctionProfile.java,v 1.14 2005/08/24 01:45:04 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.14 $
 * @see		Function
 */
public class FunctionProfile implements Comparable {
    private static final int METRIC_SIZE = 4;

    private Function function;
    private double[] doubleList;
    private double numCalls;
    private double numSubr;

    // unused profile calls
    //private List calls;

    private Set childProfiles;
    private Set parentProfiles;
    private Map childProfileCallPathSets;
    private Map parentProfileCallPathSets;

    public FunctionProfile(Function function) {
        this(function, 1);
    }

    public FunctionProfile(Function function, int numMetrics) {
        doubleList = new double[numMetrics * METRIC_SIZE];
        this.function = function;
    }

    public Function getFunction() {
        return function;
    }

    public String getName() {
        return function.getName();
    }

    public void setInclusive(int metric, double value) {
        this.insertDouble(metric, 0, value);
    }

    public double getInclusive(int metric) {
        return this.getDouble(metric, 0);
    }

    public void setExclusive(int metric, double value) {
        this.insertDouble(metric, 1, value);
    }

    public double getExclusive(int metric) {
//        if (function.isPhase()) {
//            return this.getDouble(metric, 0);
//        } else {
//            return this.getDouble(metric, 1);
//        }
        return this.getDouble(metric, 1);
    }

    public void setInclusivePercent(int metric, double value) {
        this.insertDouble(metric, 2, value);
    }

    public double getInclusivePercent(int metric) {
        return this.getDouble(metric, 2);
    }

    public void setExclusivePercent(int metric, double value) {
        this.insertDouble(metric, 3, value);
    }

    public double getExclusivePercent(int metric) {
        return this.getDouble(metric, 3);
    }

    public void setNumCalls(double value) {
        numCalls = value;
    }

    public double getNumCalls() {
        return numCalls;
    }

    public void setNumSubr(double value) {
        numSubr = value;
    }

    public double getNumSubr() {
        return numSubr;
    }

    public double getInclusivePerCall(int metric) {
        if (this.getNumCalls() == 0) {
            return 0;
        }
        return this.getInclusive(metric) / this.getNumCalls();
    }

    public double getExclusivePerCall(int metric) {
        if (this.getNumCalls() == 0) {
            return 0;
        }
        return this.getExclusive(metric) / this.getNumCalls();
    }

    // unused profile calls
    //    public void addCall(double exclusive, double inclusive) {
    //        if (calls == null)
    //            calls = new ArrayList();
    //        double[] arr = new double[2];
    //        arr[0] = exclusive;
    //        arr[1] = inclusive;
    //        calls.add(arr);
    //    }

    public int getStorageSize() {
        return doubleList.length / METRIC_SIZE;
    }

    public void incrementStorage() {
        int currentLength = doubleList.length;
        double[] newArray = new double[currentLength + METRIC_SIZE];

        for (int i = 0; i < currentLength; i++) {
            newArray[i] = doubleList[i];
        }
        doubleList = newArray;
    }

    // call path section
    public void addChildProfile(FunctionProfile child, FunctionProfile callpath) {
        // example:
        // callpath: a => b => c => d
        // child: d
        // this: c

        if (childProfiles == null)
            childProfiles = new TreeSet();
        childProfiles.add(child);

        if (childProfileCallPathSets == null)
            childProfileCallPathSets = new TreeMap();

        // we maintain a set of callpaths for each child, retrieve the set for this child
        Set callPathSet = (Set) childProfileCallPathSets.get(child);

        if (callPathSet == null) {
            callPathSet = new TreeSet();
            childProfileCallPathSets.put(child, callPathSet);
        }

        callPathSet.add(callpath);
    }

    public void addParentProfile(FunctionProfile parent, FunctionProfile callpath) {
        // example:
        // callpath: a => b => c => d
        // parent: c
        // this: d

        if (parentProfiles == null)
            parentProfiles = new TreeSet();
        parentProfiles.add(parent);

        if (parentProfileCallPathSets == null)
            parentProfileCallPathSets = new TreeMap();

        // we maintain a set of callpaths for each child, retrieve the set for this child
        Set callPathSet = (Set) parentProfileCallPathSets.get(parent);

        if (callPathSet == null) {
            callPathSet = new TreeSet();
            parentProfileCallPathSets.put(parent, callPathSet);
        }

        callPathSet.add(callpath);
    }

    public Iterator getChildProfiles() {
        if (childProfiles != null)
            return childProfiles.iterator();
        return new UtilFncs.EmptyIterator();
    }

    public Iterator getParentProfiles() {
        if (parentProfiles != null)
            return parentProfiles.iterator();
        return new UtilFncs.EmptyIterator();
    }

    public Iterator getParentProfileCallPathIterator(FunctionProfile parent) {
        if (parentProfileCallPathSets == null)
            return new UtilFncs.EmptyIterator();
        return ((Set) parentProfileCallPathSets.get(parent)).iterator();
    }

    public Iterator getChildProfileCallPathIterator(FunctionProfile child) {
        if (childProfileCallPathSets == null)
            return new UtilFncs.EmptyIterator();
        return ((Set) childProfileCallPathSets.get(child)).iterator();
    }

    /**
     * Passthrough to the actual function's isCallPathFunction
     * 
     * @return		whether or not this function is a callpath (contains '=>')
     */
    public boolean isCallPathFunction() {
        return function.isCallPathFunction();
    }

    private void insertDouble(int metric, int offset, double inDouble) {
        int actualLocation = (metric * METRIC_SIZE) + offset;
        doubleList[actualLocation] = inDouble;
    }

    private double getDouble(int metric, int offset) {
        int actualLocation = (metric * METRIC_SIZE) + offset;
        return doubleList[actualLocation];
    }

    public int compareTo(Object inObject) {
        return this.function.compareTo(((FunctionProfile) inObject).function);
    }

    public String toString() {
        return "A FunctionProfile for " + function.toString();
    }
}