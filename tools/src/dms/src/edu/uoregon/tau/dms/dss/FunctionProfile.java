
package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.text.*;

public class FunctionProfile implements Comparable {

    public FunctionProfile(Function func) {
        doubleList = new double[5];
        this.function = func;
    }

    public FunctionProfile(Function func, int capacity) {
        doubleList = new double[capacity * 5];
        this.function = func;
    }


    public Function getFunction() {
        return function;
    }

    // helpers
    public String getName() {
        return function.getName();
    }

   


    public void setInclusive(int metric, double inDouble) {
        this.insertDouble(metric, 0, inDouble);
    }

    public double getInclusive(int metric) {
        return this.getDouble(metric, 0);
    }

    public void setExclusive(int metric, double inDouble) {
        this.insertDouble(metric, 1, inDouble);
    }

    public double getExclusive(int metric) {
        return this.getDouble(metric, 1);
    }

    public void setInclusivePercent(int metric, double inDouble) {
        this.insertDouble(metric, 2, inDouble);
    }

    public double getInclusivePercent(int metric) {
        return this.getDouble(metric, 2);
    }

    public void setExclusivePercent(int metric, double inDouble) {
        this.insertDouble(metric, 3, inDouble);
    }

    public double getExclusivePercent(int metric) {
        return this.getDouble(metric, 3);
    }

    public void setNumCalls(double inDouble) {
        numCalls = inDouble;
    }

    public double getNumCalls() {
        return numCalls;
    }

    public void setNumSubr(double inDouble) {
        numSubr = inDouble;
    }

    public double getNumSubr() {
        return numSubr;
    }

    public void setInclusivePerCall(int metric, double inDouble) {
        this.insertDouble(metric, 4, inDouble);
    }

    public double getInclusivePerCall(int metric) {
        return this.getDouble(metric, 4);
    }




    public void addCall(double exclusive, double inclusive) {
        if (calls == null)
            calls = new Vector();
        double[] arr = new double[2];
        arr[0] = exclusive;
        arr[1] = inclusive;
        calls.add(arr);
    }

    public int getStorageSize() {
        return doubleList.length / 5;
    }

    public void incrementStorage() {
        int currentLength = doubleList.length;
        //can use a little space here ... space for speed! :-)
        double[] newArray = new double[currentLength + 5];

        for (int i = 0; i < currentLength; i++) {
            newArray[i] = doubleList[i];
        }
        doubleList = newArray;
    }

    //######
    //Call path section.
    //######
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

        // we maintain a set of callpaths for each parent, retrieve the set for
        // this parent
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

        // we maintain a set of callpaths for each child, retrieve the set for
        // this child
        Set callPathSet = (Set) childCallPathSets.get(child);

        if (callPathSet == null) {
            callPathSet = new TreeSet();
            childCallPathSets.put(child, callPathSet);
        }

        callPathSet.add(callPath);
    }


    public boolean isCallPathObject() {
        return function.isCallPathObject();
    }

    //######
    //End - Call path section.
    //######

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


    
    public int compareTo(Object inObject) {
        return this.function.compareTo(((FunctionProfile)inObject).function);
    }

    
    private Function function = null;
    private double[] doubleList;
    private double numCalls = 0;
    private double numSubr = 0;

    private Vector calls = null;

    private Set children;
    private Set parents;

    private Set childProfiles;
    private Set parentProfiles;

    private Map childCallPathSets;
    private Map parentCallPathSets;
    private boolean callPathObject = false;
}