package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.awt.*;
import java.io.*;
import java.text.*;

public class Function implements Serializable, Comparable {

    public Function(String name, int id, int numMetrics) {
        this.name = name;
        this.id = id;
        doubleList = new double[numMetrics * 15];
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
    
    //######
    //Storage control.
    //######
    public int getStorageSize() {
        return doubleList.length / 15;
    }

    public void incrementStorage() {
        int currentLength = doubleList.length;
        double[] newArray = new double[currentLength + 15];
        for (int i = 0; i < currentLength; i++) {
            newArray[i] = doubleList[i];
        }
        doubleList = newArray;
    }

    public void incrementStorage(int increase) {
        int currentLength = doubleList.length;
        double[] newArray = new double[currentLength + (increase * 15)];
        for (int i = 0; i < currentLength; i++) {
            newArray[i] = doubleList[i];
        }
        doubleList = newArray;
    }

    //######
    //End - Storage control.
    //######

    //######
    //Group section.
    //######
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

    public void setGroupsSet(boolean groupsSet) {
        this.groupsSet = groupsSet;
    }

    public boolean groupsSet() {
        return groupsSet;
    }


    //######
    //Call path section.
    //######
    public void addParent(Function parent, Function callPath) {
        // example:
        // fullpath = a => b => c => d
        // parent = c
        // we are d

        if (parents == null)
            parents = new HashSet();

        parents.add(parent);

        if (parentCallPathSets == null)
            parentCallPathSets = new HashMap();

        // we maintain a set of callpaths for each parent, retrieve the set for
        // this parent
        Set callPathSet = (Set) parentCallPathSets.get(parent);

        if (callPathSet == null) {
            callPathSet = new HashSet();
            parentCallPathSets.put(parent, callPathSet);
        }

        callPathSet.add(callPath);
    }

    public Iterator getParents() {
        if (parents == null)
            return new DssIterator();
        return parents.iterator();
    }

    public Iterator getChildren() {
        if (children == null)
            return new DssIterator();
        return children.iterator();
    }

    public Iterator getParentCallPathIterator(Function parent) {
        if (parentCallPathSets != null)
            if (parentCallPathSets.get(parent) != null)
                return ((Set) parentCallPathSets.get(parent)).iterator();
        return new DssIterator();
    }

    public Iterator getChildCallPathIterator(Function child) {
        if (childCallPathSets == null)
            return new DssIterator();
        return ((Set) childCallPathSets.get(child)).iterator();
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

   
    
    public void setCallPathObject(boolean b) {
        callPathObject = b;
    }

    public boolean isCallPathObject() {
        return callPathObject;
    }

    //######
    //End - Call path section.
    //######

    //######
    //Begin - Color section.
    //######
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

    //######
    //End - Color section.
    //######

    //######
    //Max values section.
    //######
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

    //######
    //End - Max values section.
    //######

    //######
    //Mean section.
    //######
   
    public void setMeanProfile(FunctionProfile fp) {
        this.meanProfile = fp;
    }
    
    public FunctionProfile getMeanProfile() {
        return meanProfile;
    }
    
////    public void setMeanInclusiveValue(int metric, double d) {
//       // meanProfile.setInclusiveValue(metric, d);
// //   }
//    
//    public void setMeanExclusiveValue(int metric, double d) {
//    }
//
//    public void setMeanInclusivePercentValue(int metric, double d) {}
//    public void setMeanExclusivePercentValue(int metric, double d) {}
//    public void setMeanNumberOfCalls(double d) {}
//    public void setMeanNumberOfSubRoutines(double d) {}
//    public void setMeanUserSecPerCall(int metric, double d) {}
//    
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
//
//    
    public void setMeanValuesSet(boolean meanValuesSet) {
        this.meanValuesSet = meanValuesSet;
    }

    public boolean getMeanValuesSet() {
        return meanValuesSet;
    }

    //######
    //End - Mean section.
    //######

    //######
    //Total section.
    //######
    public void setTotalInclusive(int metric, double d) {
        this.insertDouble(metric, 10, d);
    }

    public double getTotalInclusive(int metric) {
        return this.getDouble(metric, 10);
    }

    public void setTotalExclusive(int metric, double d) {
        this.insertDouble(metric, 11, d);
    }

    public double getTotalExclusive(int metric) {
        return this.getDouble(metric, 11);
    }

    public void setTotalInclusivePercent(int metric, double d) {
        this.insertDouble(metric, 12, d);
    }

    public double getTotalInclusivePercent(int metric) {
        return this.getDouble(metric, 12);
    }

    public void setTotalExclusivePercent(int metric, double d) {
        this.insertDouble(metric, 13, d);
    }

    public double getTotalExclusivePercent(int metric) {
        return this.getDouble(metric, 13);
    }

    public void setTotalNumCalls(double i) {
        totalNumCalls = i;
    }


    public double getTotalNumCalls() {
        return totalNumCalls;
    }

    public void setTotalNumSubr(double i) {
        totalNumSubr = i;
    }


    public double getTotalNumSubr() {
        return totalNumSubr;
    }

    public void setTotalInclusivePerCall(int metric, double d) {
        this.insertDouble(metric, 14, d);
    }

    public double getTotalInclusivePerCall(int metric) {
        return this.getDouble(metric, 14);
    }

    //######
    //End - Total section.
    //######

    //####################################
    //Interface code.
    //####################################

    //######
    //Comparable section.
    //######
    public int compareTo(Object inObject) {
        Integer thisInt = new Integer(this.id);
        return thisInt.compareTo(new Integer(((Function)inObject).getID()));
        
        //return name.compareTo(((Function)inObject).getName());
    }

    //######
    //End - Comparable section.
    //######

    //####################################
    //End - Interface code.
    //####################################

    //######
    //Private section.
    //######
    private void insertDouble(int metric, int offset, double d) {
            doubleList[(metric * 15) + offset] = d;
    }

    private double getDouble(int metric, int offset) {
            return doubleList[(metric * 15) + offset];
    }

    private int insertSpaces(char[] inArray, int position, int number) {
        for (int i = 0; i < number; i++) {
            inArray[position] = '\u0020';
            position++;
        }
        return position;
    }

    //######
    //End - Private section.
    //######

    //####################################
    //Instance data.
    private String name = null;
    private int id = -1;

    private Vector groups = null;

    //    private Vector parents = null;
    //    private Vector children = null;
    //    private Vector callPathIDSParents = null;
    //    private Vector callPathIDSChildren = null;

    private Set children;
    private Set parents;
 
    
    private Map childCallPathSets;
    private Map parentCallPathSets;
    private boolean callPathObject = false;

    //Color Settings.
    private boolean colorFlag = false;
    private Color color = null;
    private Color specificColor = null;

    private double[] doubleList = null;
    private double maxNumCalls = 0;
    private double maxNumSubr = 0;
    private double totalNumCalls = 0;
    private double totalNumSubr = 0;

    private boolean meanValuesSet = false;
    private boolean groupsSet = false;

    //Instance values used to calculate the meanProfile values for derived values
    // (such as flops)
    private int counter = 0;
    private double totalExclusiveValue = 0;
    private double totalInclusiveValue = 0;

    private FunctionProfile meanProfile;
    
    //####################################
    //End - Instance data.
    //####################################
}