package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.text.*;

public class FunctionProfile implements Comparable {

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

    public void setInclusivePerCall(int metric, double value) {
        this.insertDouble(metric, 4, value);
    }

    public double getInclusivePerCall(int metric) {
        return this.getDouble(metric, 4);
    }

    // unused profile calls
    //    public void addCall(double exclusive, double inclusive) {
    //        if (calls == null)
    //            calls = new Vector();
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

    //Call path section.
    public void addParent(Function parent, Function callPath) {
        // example:
        // fullpath = a => b => c => d
        // child = d
        // we are c

        if (parents == null)
            parents = new TreeSet();

        parents.add(parent);

        if (parentCallPathSets == null)
            parentCallPathSets = new TreeMap();

        // we maintain a set of callpaths for each parent, retrieve the set for this parent
        Set callPathSet = (Set) parentCallPathSets.get(parent);

        if (callPathSet == null) {
            callPathSet = new TreeSet();
            parentCallPathSets.put(parent, callPathSet);
        }

        callPathSet.add(callPath);
    }

    public Iterator getParents() {
        if (parents != null)
            return parents.iterator();
        return new DssIterator();
    }

    public Iterator getChildren() {
        if (children != null)
            return children.iterator();
        return new DssIterator();
    }

    public Iterator getParentCallPathIterator(Function parent) {
        if (parentCallPathSets == null)
            return new DssIterator();
        return ((Set) parentCallPathSets.get(parent)).iterator();
    }

    public Iterator getChildCallPathIterator(Function child) {
        if (childCallPathSets == null)
            return new DssIterator();
        return ((Set) childCallPathSets.get(child)).iterator();
    }

    public void addChild(FunctionProfile child) {
        if (childProfiles == null)
            childProfiles = new TreeSet();
        childProfiles.add(child);
    }

    public void addParent(FunctionProfile parent) {
        if (parentProfiles == null)
            parentProfiles = new TreeSet();
        parentProfiles.add(parent);
    }

    public Iterator getChildProfiles() {
        if (childProfiles != null)
            return childProfiles.iterator();
        return new DssIterator();
    }

    public Iterator getParentProfiles() {
        if (parentProfiles != null)
            return parentProfiles.iterator();
        return new DssIterator();
    }

    public void addChild(Function child, Function callPath) {
        // example:
        // fullpath = a => b => c => d
        // child = d
        // we are c

        if (children == null)
            children = new TreeSet();

        children.add(child);

        if (childCallPathSets == null)
            childCallPathSets = new TreeMap();

        // we maintain a set of callpaths for each child, retrieve the set for this child
        Set callPathSet = (Set) childCallPathSets.get(child);

        if (callPathSet == null) {
            callPathSet = new TreeSet();
            childCallPathSets.put(child, callPathSet);
        }

        callPathSet.add(callPath);
    }

    public boolean isCallPathObject() {
        return function.isCallPathFunction();
    }

    //End - Call path section.

    //Private section.
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

    private Function function;
    private double[] doubleList;
    private double numCalls;
    private double numSubr;

    // unused profile calls
    //private Vector calls;

    private Set children;
    private Set parents;

    private Set childProfiles;
    private Set parentProfiles;

    private Map childCallPathSets;
    private Map parentCallPathSets;

    private static final int METRIC_SIZE = 5;
}