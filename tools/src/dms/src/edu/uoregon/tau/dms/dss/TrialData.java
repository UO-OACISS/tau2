
package edu.uoregon.tau.dms.dss;

import java.util.*;

public class TrialData {

    public Function addFunction(String name, int numMetrics) {
        Object obj = functions.get(name);

        if (obj != null)
            return (Function) obj;

        Function func = new Function(name, functions.size(), numMetrics);
        functions.put(name, func);
        return func;
    }

    public Function getFunction(String name) {
        return (Function) functions.get(name);
    }
    
    public Group addGroup(String name) {
        Object obj = groups.get(name);

        if (obj != null)
            return (Group) obj;

        Group group = new Group(name,groups.size()+1);
        groups.put(name, group);
        return group;
    }

    public UserEvent addUserEvent(String name) {
        Object obj = userEvents.get(name);

        if (obj != null)
            return (UserEvent) obj;

        UserEvent userEvent = new UserEvent(name, userEvents.size()+1);
        userEvents.put(name, userEvent);
        return userEvent;
    }
    
    public UserEvent getUserEvent(String name) {
        return (UserEvent) userEvents.get(name);
    }
    
    public int getNumUserEvents() {
        return userEvents.size();
    }
    

    public void increaseVectorStorage() {
        //	System.out.println ("increaseVectorStorage called!\n");
        maxMeanInclusiveValueList.add(new Double(0));
        maxMeanExclusiveValueList.add(new Double(0));
        maxMeanInclusivePercentValueList.add(new Double(0));
        maxMeanExclusivePercentValueList.add(new Double(0));
        maxMeanInclusivePerCallList.add(new Double(0));
    }

    public int getNumFunctions() {
        return functions.size();
    }

    public Iterator getFunctions() {
        return functions.values().iterator();
    }
    
    public Iterator getGroups() {
        return groups.values().iterator();
    }
    
    public Iterator getUserEvents() {
        return userEvents.values().iterator();
    }

    public boolean displayFunction(Function func) {
        switch (groupFilter) {
        case 0:
            //No specific group selection is required.
            return true;
        case 1:
            //Show this group only.
            if (func.isGroupMember(this.selectedGroup))
                return true;
            else
                return false;
        case 2:
            //Show all groups except this one.
            if (func.isGroupMember(this.selectedGroup))
                return false;
            else
                return true;
        default:
            //Default case behaves as case 0.
            return true;
        }
    }

    public void setSelectedGroup(Group group) {
        this.selectedGroup = group;
    }

    public Group getSelectedGroup() {
        return selectedGroup;
    }

    public void setGroupFilter(int groupFilter) {
        this.groupFilter = groupFilter;
    }

    public int getGroupFilter() {
        return groupFilter;
    }

    //######
    //End - Group functionProfiles
    //######



    //######
    //Functions for max mean values.
    //######
    public void setMaxMeanInclusiveValue(int metric, double inDouble) {
        Double tmpDouble = new Double(inDouble);
        //maxMeanInclusiveValueList.add(metric, tmpDouble);
        maxMeanInclusiveValueList.setElementAt(tmpDouble, metric);
    }

    public void setMaxMeanExclusiveValue(int metric, double inDouble) {
        //System.out.println ("setMaxMeanExclusiveValue(" + metric +
        // ", " + inDouble + ")");
        Double tmpDouble = new Double(inDouble);
        //maxMeanExclusiveValueList.add(metric, tmpDouble);
        maxMeanExclusiveValueList.setElementAt(tmpDouble, metric);
    }

    public void setMaxMeanInclusivePercentValue(int metric, double inDouble) {
        Double tmpDouble = new Double(inDouble);
        //maxMeanInclusivePercentValueList.add(metric, tmpDouble);
        maxMeanInclusivePercentValueList.setElementAt(tmpDouble, metric);
    }

    public void setMaxMeanExclusivePercentValue(int metric, double inDouble) {
        Double tmpDouble = new Double(inDouble);
        //maxMeanExclusivePercentValueList.add(metric, tmpDouble);
        maxMeanExclusivePercentValueList.setElementAt(tmpDouble, metric);
    }

    public void setMaxMeanNumberOfCalls(double inDouble) {
        maxMeanNumberOfCalls = inDouble;
    }

    public void setMaxMeanNumberOfSubRoutines(double inDouble) {
        maxMeanNumberOfSubRoutines = inDouble;
    }

    public void setMaxMeanInclusivePerCall(int metric, double inDouble) {
        Double tmpDouble = new Double(inDouble);
        //maxMeanInclusivePerCallList.add(metric, tmpDouble);
        maxMeanInclusivePerCallList.setElementAt(tmpDouble, metric);
    }

    public double getMaxMeanExclusiveValue(int metric) {
        Double tmpDouble = (Double) maxMeanExclusiveValueList.elementAt(metric);
        //System.out.println ("getMaxMeanExclusiveValue(" + metric +
        // ") = " + tmpDouble);
        return tmpDouble.doubleValue();
    }

    public double getMaxMeanInclusiveValue(int metric) {
        Double tmpDouble = (Double) maxMeanInclusiveValueList.elementAt(metric);
        return tmpDouble.doubleValue();
    }

    public double getMaxMeanInclusivePercentValue(int metric) {
        Double tmpDouble = (Double) maxMeanInclusivePercentValueList.elementAt(metric);
        return tmpDouble.doubleValue();
    }

    public double getMaxMeanExclusivePercentValue(int metric) {
        Double tmpDouble = (Double) maxMeanExclusivePercentValueList.elementAt(metric);
        return tmpDouble.doubleValue();
    }

    public double getMaxMeanNumberOfCalls() {
        return maxMeanNumberOfCalls;
    }

    public double getMaxMeanNumberOfSubRoutines() {
        return maxMeanNumberOfSubRoutines;
    }

    public double getMaxMeanInclusivePerCall(int metric) {
        Double tmpDouble = (Double) maxMeanInclusivePerCallList.elementAt(metric);
        return tmpDouble.doubleValue();
    }

    //######
    //End - Function to set max mean values.
    //######


    //####################################
    //Instance data.
    //####################################

    private Map functions = new TreeMap();
    private Map groups = new TreeMap();
    private Map userEvents = new TreeMap();
    
    private Vector maxMeanInclusiveValueList = new Vector();
    private Vector maxMeanExclusiveValueList = new Vector();
    private Vector maxMeanInclusivePercentValueList = new Vector();
    private Vector maxMeanExclusivePercentValueList = new Vector();
    private double maxMeanNumberOfCalls = 0;
    private double maxMeanNumberOfSubRoutines = 0;
    private Vector maxMeanInclusivePerCallList = new Vector();

    private Group selectedGroup;
    private int groupFilter = 0;
    //####################################
    //End - Instance data.
    //####################################

}