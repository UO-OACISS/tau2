package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.awt.*;
import java.io.*;
import java.text.*;

/**
 * Function
 * This class represents a "function".  A function is defined over all threads
 * in the profile, so per-thread data is not stored here.
 *  
 * <P>CVS $Id: Function.java,v 1.3 2005/01/06 22:46:56 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.3 $
 * @see		FunctionProfile
 */

public class Function implements Serializable, Comparable {

    public Function(String name, int id, int numMetrics) {
        this.name = name;
        this.id = id;
        doubleList = new double[numMetrics * STORAGE_SIZE];
    }

    public int getID() {
        return id;
    }

    public String getName() {
        return name;
    }

    public String toString() {
        return name;
    }

    // Storage control
    public int getStorageSize() {
        return doubleList.length / STORAGE_SIZE;
    }

    public void incrementStorage() {
        int currentLength = doubleList.length;
        double[] newArray = new double[currentLength + STORAGE_SIZE];
        for (int i = 0; i < currentLength; i++) {
            newArray[i] = doubleList[i];
        }
        doubleList = newArray;
    }

    public void incrementStorage(int increase) {
        int currentLength = doubleList.length;
        double[] newArray = new double[currentLength + (increase * STORAGE_SIZE)];
        for (int i = 0; i < currentLength; i++) {
            newArray[i] = doubleList[i];
        }
        doubleList = newArray;
    }
    //End - Storage control.

    // Group section
    public void addGroup(Group group) {
        //Don't add group if already a member.
        if (this.isGroupMember(group))
            return;
        if (groups == null)
            groups = new Vector();

        groups.add(group);
    }

    public boolean isGroupMember(Group group) {
        if (groups == null)
            return false;
        return groups.contains(group);
    }

    public Vector getGroups() {
        return groups;
    }



    //Call path section.
    public void addParent(Function parent, Function callPath) {
        meanProfile.addParent(parent, callPath);
//        // example:
//        // fullpath = a => b => c => d
//        // parent = c
//        // we are d
//
//        if (parents == null)
//            parents = new HashSet();
//
//        parents.add(parent);
//
//        if (parentCallPathSets == null)
//            parentCallPathSets = new HashMap();
//
//        // we maintain a set of callpaths for each parent, retrieve the set for this parent
//        Set callPathSet = (Set) parentCallPathSets.get(parent);
//
//        if (callPathSet == null) {
//            callPathSet = new HashSet();
//            parentCallPathSets.put(parent, callPathSet);
//        }
//
//        callPathSet.add(callPath);
    }

    public Iterator getParents() {
        return meanProfile.getParents();
//        if (parents == null)
//            return new DssIterator();
//        return parents.iterator();
    }

    public Iterator getChildren() {
        return meanProfile.getChildren();
//        if (children == null)
//            return new DssIterator();
//        return children.iterator();
    }

    public Iterator getParentCallPathIterator(Function parent) {
        return meanProfile.getParentCallPathIterator(parent);
//        if (parentCallPathSets != null)
//            if (parentCallPathSets.get(parent) != null)
//                return ((Set) parentCallPathSets.get(parent)).iterator();
//        return new DssIterator();
    }

    public Iterator getChildCallPathIterator(Function child) {
        return meanProfile.getChildCallPathIterator(child);
//        if (childCallPathSets == null)
//            return new DssIterator();
//        return ((Set) childCallPathSets.get(child)).iterator();
    }

    public void addChild(Function child, Function callPath) {
        meanProfile.addChild(child, callPath);
//        
//        // example:
//        // fullpath = a => b => c => d
//        // child = d
//        // we are c
//
//        if (children == null)
//            children = new TreeSet();
//
//        children.add(child);
//
//        if (childCallPathSets == null)
//            childCallPathSets = new TreeMap();
//
//        // we maintain a set of callpaths for each child, retrieve the set for this child
//        Set callPathSet = (Set) childCallPathSets.get(child);
//
//        if (callPathSet == null) {
//            callPathSet = new TreeSet();
//            childCallPathSets.put(child, callPathSet);
//        }
//
//        callPathSet.add(callPath);
    }

    public void setCallPathFunction(boolean b) {
        callPathFunction = b;
    }

    public boolean isCallPathFunction() {
        return callPathFunction;
    }

    //End - Call path section.

    //Begin - Color section.
    public void setColor(Color color) {
        this.color = color;
    }

    public Color getColor() {
        if (colorFlag)
            return specificColor;
        else
            return color;
    }

    public void setColorFlag(boolean colorFlag) {
        this.colorFlag = colorFlag;
    }

    public boolean isColorFlagSet() {
        return colorFlag;
    }

    public void setSpecificColor(Color specificColor) {
        this.specificColor = specificColor;
    }
    //End - Color section.

    //Max values section.
    public void setMaxInclusive(int metric, double d) {
        this.insertDouble(metric, 0, d);
    }

    public double getMaxInclusive(int metric) {
        return this.getDouble(metric, 0);
    }

    public void setMaxExclusive(int metric, double d) {
        this.insertDouble(metric, 1, d);
    }

    public double getMaxExclusive(int metric) {
        return this.getDouble(metric, 1);
    }

    public void setMaxInclusivePercent(int metric, double d) {
        this.insertDouble(metric, 2, d);
    }

    public double getMaxInclusivePercent(int metric) {
        return this.getDouble(metric, 2);
    }

    public void setMaxExclusivePercent(int metric, double d) {
        this.insertDouble(metric, 3, d);
    }

    public double getMaxExclusivePercent(int metric) {
        return this.getDouble(metric, 3);
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

    public void setMaxInclusivePerCall(int metric, double d) {
        this.insertDouble(metric, 4, d);
    }

    public double getMaxInclusivePerCall(int metric) {
        return this.getDouble(metric, 4);
    }

    //End - Max values section.

    //Mean section.
    public void setMeanProfile(FunctionProfile fp) {
        this.meanProfile = fp;
    }

    public FunctionProfile getMeanProfile() {
        return meanProfile;
    }

    public double getMeanInclusive(int metric) {
        return meanProfile.getInclusive(metric);
    }

    public double getMeanExclusive(int metric) {
        return meanProfile.getExclusive(metric);
    }

    public double getMeanInclusivePercent(int metric) {
        return meanProfile.getInclusivePercent(metric);
    }

    public double getMeanExclusivePercent(int metric) {
        return meanProfile.getExclusivePercent(metric);
    }

    public double getMeanNumCalls() {
        return meanProfile.getNumCalls();
    }

    public double getMeanNumSubr() {
        return meanProfile.getNumSubr();
    }

    public double getMeanInclusivePerCall(int metric) {
        return meanProfile.getInclusivePerCall(metric);
    }

    //End - Mean section.

    //Total section.
    public double getTotalInclusive(int metric) {
        return totalProfile.getInclusive(metric);
    }

    public double getTotalExclusive(int metric) {
        return totalProfile.getExclusive(metric);
    }

    public double getTotalInclusivePercent(int metric) {
        return totalProfile.getInclusivePercent(metric);
    }

    public double getTotalExclusivePercent(int metric) {
        return totalProfile.getExclusivePercent(metric);
    }

    public double getTotalNumCalls() {
        return totalProfile.getNumCalls();
    }

    public double getTotalNumSubr() {
        return totalProfile.getNumSubr();
    }

    public double getTotalInclusivePerCall(int metric) {
        return totalProfile.getInclusivePerCall(metric);
    }

    public void setTotalProfile(FunctionProfile fp) {
        this.totalProfile = fp;
    }

    public FunctionProfile getTotalProfile() {
        return totalProfile;
    }

    public int compareTo(Object inObject) {
        Integer thisInt = new Integer(this.id);
        return thisInt.compareTo(new Integer(((Function) inObject).getID()));
        //return name.compareTo(((Function)inObject).getName());
    }

    //Private section.
    private void insertDouble(int metric, int offset, double d) {
        doubleList[(metric * STORAGE_SIZE) + offset] = d;
    }

    private double getDouble(int metric, int offset) {
        return doubleList[(metric * STORAGE_SIZE) + offset];
    }

    private final static int STORAGE_SIZE = 5;

    // instance data
    private String name = null;
    private int id = -1;

    private Vector groups = null;

    private boolean callPathFunction = false;

    //Color Settings.
    private boolean colorFlag = false;
    private Color color = null;
    private Color specificColor = null;

    private double[] doubleList = null;
    private double maxNumCalls = 0;
    private double maxNumSubr = 0;


    private FunctionProfile meanProfile;
    private FunctionProfile totalProfile;

}