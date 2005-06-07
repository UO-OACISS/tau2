package edu.uoregon.tau.dms.dss;

import java.awt.Color;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * This class represents a "function".  A function is defined over all threads
 * in the profile, so per-thread data is not stored here.
 *  
 * <P>CVS $Id: Function.java,v 1.9 2005/06/07 01:25:33 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.9 $
 * @see		FunctionProfile
 */
public class Function implements Serializable, Comparable {

    private String name = null;
    private String reversedName = null;
    private int id = -1;
    private List groups = null;
    private boolean callPathFunction = false;
 
    // we hold on to the mean and total profiles for pass-through functions
    private FunctionProfile meanProfile;
    private FunctionProfile stddevProfile;
    private FunctionProfile totalProfile;

    // color settings
    private boolean colorFlag = false;
    private Color color = null;
    private Color specificColor = null;

    public Function(String name, int id, int numMetrics) {
        this.name = name;
        this.id = id;
    }

    public int getID() {
        return id;
    }

    public String getName() {
        return name;
    }

    /**
     * Retrieve the reversed (callpath) name
     * If the function's name is "A => B => C", this will return "C <= B <= A"
     * 
     * @return      The reversed name
     */
    public String getReversedName() {
        if (reversedName == null) {
            if (callPathFunction == false) {
                reversedName = name;
            } else {

                String s = name;
                int location = s.lastIndexOf("=>");
                reversedName = "";

                while (location != -1) {
                    String childName = s.substring(location + 3, s.length());
                    
                    childName = childName.trim();
                    reversedName = reversedName + childName;
                    s = s.substring(0, location);
                    
                    location = s.lastIndexOf("=>");
                    if (location != -1) {
                        reversedName = reversedName + " <= ";
                    }
                }
                reversedName = reversedName + " <= " + s;
            }
        }
        
        return reversedName;
    }
    
    public String toString() {
        return name;
    }


    // Group section
    public void addGroup(Group group) {
        //Don't add group if already a member.
        if (this.isGroupMember(group))
            return;
        if (groups == null)
            groups = new ArrayList();

        groups.add(group);
    }

    public boolean isGroupMember(Group group) {
        if (groups == null)
            return false;
        return groups.contains(group);
    }

    public List getGroups() {
        return groups;
    }


    /**
     * Set callpath status
     * 
     * @param	b			whether or not this function is a callpath (contains '=>')
     */
    public void setCallPathFunction(boolean b) {
        callPathFunction = b;
    }

    /**
     * Retrieve callpath status
     * 
     * @return				whether or not this function is a callpath (contains '=>')
     */
    public boolean isCallPathFunction() {
        return callPathFunction;
    }

    // color section
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


    public void setStddevProfile(FunctionProfile fp) {
        this.stddevProfile = fp;
    }

    public FunctionProfile getStddevProfile() {
        return stddevProfile;
    }

    
    // mean section
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

    // total section
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

    public int compareTo(Object o) {
        return this.id - ((Function)o).getID();
    }

}