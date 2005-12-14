package edu.uoregon.tau.perfdmf;

import java.awt.Color;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * This class represents a "function".  A function is defined over all threads
 * in the profile, so per-thread data is not stored here.
 *  
 * <P>CVS $Id: Function.java,v 1.2 2005/12/14 01:30:54 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.2 $
 * @see		FunctionProfile
 */
/**
 * @author amorris
 *
 * TODO ...
 */
public class Function implements Serializable, Comparable {

    private String name = null;
    private String reversedName = null;
    private int id = -1;
    private List groups = null;
    private boolean phase = false;
    private Function actualPhase;
    private Function parentPhase;
    //private boolean phaseSet = false;

    boolean callpathFunction = false;
    boolean callpathFunctionSet = false;

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
            if (isCallPathFunction() == false) {
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

    public boolean isPhaseMember(Function phase) {
        if (phase == null) {
            return true;
        }

        if (isCallPathFunction() != true) {
            return false;
        }

        int location = name.indexOf("=>");
        String phaseRoot = name.substring(0, location).trim();

        if (phaseRoot.compareTo(phase.getName()) == 0) {
            return true;
        }

        return false;
    }


    public boolean isCallPathFunction() {
        if (!callpathFunctionSet) {
            if (name.indexOf("=>") > 0) {
                callpathFunction = true;
            }
            callpathFunctionSet = true;
        }
        return callpathFunction;
    }

//    public boolean isPhase() {
//        if (!phaseSet) {
//            for (int i = 0; i < groups.size(); i++) {
//                if (((Group) groups.get(i)).getName().compareTo("TAU_PHASE") == 0) {
//                    phase = true;
//                }
//            }
//            phaseSet = true;
//        }
//        return phase;
//    }

    
//    private String getRightSide() {
//        if (!getCallPathFunction()) {
//            return null;
//        } 
//
//        
//        int location = name.indexOf("=>");
//        String phaseRoot = name.substring(0, location).trim();
//        String phaseChild = name.substring(location).trim();
//        
//        return phaseChild;
//    }
//    
//    public boolean isPhase() {
//        if (!phaseSet) {
//
//            if (name.indexOf("=>") > 0) {
//                callpathFunction = true;
//            }
//
//            for (int i = 0; i < groups.size(); i++) {
//                if (((Group) groups.get(i)).getName().compareTo("TAU_PHASE") == 0) {
//                    phase = true;
//                }
//            }
//            phaseSet = true;
//        }
//        return phase;
//    }
    
    
    
    // color section
    public void setColor(Color color) {
        this.color = color;
    }

    public Color getColor() {
        if (colorFlag) {
            return specificColor;
        } else {
            return color;
        }
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
        return this.id - ((Function) o).getID();
    }

    public boolean isPhase() {
        return phase;
    }

    public void setPhase(boolean phase) {
        this.phase = phase;
    }

    /**
     * Returns the actual phase.  If "A" and "B" are phases, then "A => B" is also a phase.
     * getActualPhase will return "B" for "A => B".  It will return "B" for "B".
     * 
     * @return the actual phase
     */
    public Function getActualPhase() {
        return actualPhase;
    }

    public void setActualPhase(Function actualPhase) {
        this.actualPhase = actualPhase;
    }

    /**
     * Returns the phase that this function exists in.
     * Example:
     *   if this function is "A => B", then parent phase is "A"
     * 
     * @return the parent phase
     */
    public Function getParentPhase() {
        return parentPhase;
    }

    public void setParentPhase(Function parentPhase) {
        this.parentPhase = parentPhase;
    }

}