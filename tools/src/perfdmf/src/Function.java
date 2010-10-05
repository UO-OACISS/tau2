package edu.uoregon.tau.perfdmf;

import java.awt.Color;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * This class represents a "function".  A function is defined over all threads
 * in the profile, so per-thread data is not stored here.
 *  
 * <P>CVS $Id: Function.java,v 1.19 2009/12/18 04:00:06 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.19 $
 * @see		FunctionProfile
 */
/**
 * @author amorris
 *
 * TODO ...
 */
public class Function implements Serializable, Comparable<Function> {

    /**
	 * 
	 */
	private static final long serialVersionUID = 362090098221172924L;
	private String name;
    private String reversedName;
    private int id = -1;
    private List<Group> groups = new ArrayList<Group>();
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

    // source code link
    private SourceRegion sourceLink;

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

    public void setName(String name) {
        this.name = name;
        this.reversedName = null;
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

        return reversedName.trim();
    }

    public String toString() {
        return name;
    }

    public SourceRegion getSourceLink() {
        if (this.sourceLink == null) {
            if (isGroupMember("TAU_CALLPATH_DERIVED")) {
                this.sourceLink = getSourceLink(UtilFncs.getLeftSide(name));
            } else {
                this.sourceLink = getSourceLink(this.name);
            }
        }
        return sourceLink;
    }

    public static SourceRegion getSourceLink(String name) {
        SourceRegion sourceLink = new SourceRegion();

        if (name.indexOf("OpenMP location:") != -1) { // opari instrumentation points
            // parse source location with format:
            // parallel [OpenMP location: file:omp_hello.chk.c <31, 48>]

            int fileIndex = name.indexOf("file:");
            int left = name.indexOf("<");
            int comma = name.indexOf(",");
            int right = name.indexOf(">");

            sourceLink.setFilename(name.substring(fileIndex + 5, left).trim());
            sourceLink.setStartLine(Integer.parseInt(name.substring(left + 1, comma).trim()));
            sourceLink.setEndLine(Integer.parseInt(name.substring(comma + 1, right).trim()));
        }
        if (name.indexOf("file:") != -1 && name.indexOf("line:") != -1) {
            // MpiP source information

            // start with the last section of location information
            // we may have: "main [file:ring.c line:37] =>  func [file:ring.c line:19] => MPI_Recv"
            // and we want the last section (for now)
            name = name.substring(name.lastIndexOf("["));
            int fileIndex = name.indexOf("file:");
            int lineIndex = name.indexOf("line:");
            String filename = name.substring(fileIndex + 5, lineIndex).trim();

            sourceLink.setFilename(filename);
            int lineNumber;
            if (name.indexOf("]", lineIndex) != -1) {
                // new mpiP
                lineNumber = Integer.parseInt(name.substring(lineIndex + 5, name.indexOf("]")));
            } else {
                // old mpiP
                lineNumber = Integer.parseInt(name.substring(lineIndex + 5).trim());
            }
            sourceLink.setStartLine(lineNumber);
            sourceLink.setEndLine(lineNumber);
            return sourceLink;
        }

        // for TAU, look at the leaf location information
        if (isCallPathFunction(name)) {
            name = name.substring(name.lastIndexOf("=>") + 2);
        }

        int filenameStart = name.indexOf("[{");
        if (filenameStart == -1) {
            return sourceLink;
        }
        int filenameEnd = name.indexOf("}", filenameStart);
        if (filenameEnd == -1) {
            // quit, it's not valid
            return sourceLink;
        }

        int openbracket1 = name.indexOf("{", filenameEnd + 1);
        int comma1 = name.indexOf(",", filenameEnd + 1);
        int closebracket1 = name.indexOf("}", filenameEnd + 1);
        int dash = name.indexOf("-", closebracket1 + 1);
        int openbracket2 = name.indexOf("{", openbracket1 + 1);
        int comma2 = name.indexOf(",", comma1 + 1);
        int closebracket2 = name.indexOf("}", closebracket1 + 1);

        String filename = name.substring(filenameStart + 2, filenameEnd);
        filename = filename.substring(filename.lastIndexOf("/") + 1);

        sourceLink.setFilename(filename);

        if (openbracket1 == -1) {
            return sourceLink;
        }

        if (dash == -1) {
            // fortran (e.g. "foo [{foo.cpp} {1,1}]")
            if (comma1 == -1) {
                if (closebracket1 != -1) {
                    int linenumber = Integer.parseInt(name.substring(openbracket1 + 1, closebracket1));
                    sourceLink.setStartLine(linenumber);
                    sourceLink.setEndLine(linenumber);
                    return sourceLink;
                } else {
                    return sourceLink;
                }
            }
            int linenumber = Integer.parseInt(name.substring(openbracket1 + 1, comma1));
            sourceLink.setStartLine(linenumber);
            sourceLink.setEndLine(linenumber);
            return sourceLink;
        } else {
            // loop or c/c++ (e.g. "foo [{foo.cpp} {1,1}-{5,5}]")
            if (openbracket1 == -1 || openbracket2 == -1 || comma1 == -1 || comma2 == -1 || closebracket1 == -1
                    || closebracket2 == -1) {
                return sourceLink;
            }
            int startLine = Integer.parseInt(name.substring(openbracket1 + 1, comma1));
            int startColumn = Integer.parseInt(name.substring(comma1 + 1, closebracket1));
            int endLine = Integer.parseInt(name.substring(openbracket2 + 1, comma2));
            int endColumn = Integer.parseInt(name.substring(comma2 + 1, closebracket2));

            sourceLink.setStartLine(startLine);
            sourceLink.setStartColumn(startColumn);
            sourceLink.setEndLine(endLine);
            sourceLink.setEndColumn(endColumn);
        }
        return sourceLink;
    }

    public static boolean isCallPathFunction(String name) {
        if (name.indexOf("=>") > 0) {
            return true;
        }
        return false;
    }

    // Group section
    public void addGroup(Group group) {
        //Don't add group if already a member.
        if (this.isGroupMember(group)) {
            return;
        }
        groups.add(group);
    }

    public void removeGroup(Group group) {
        if (isGroupMember(group)) {
            groups.remove(group);
        }
    }

    public boolean isGroupMember(Group group) {
        return groups.contains(group);
    }

    public List<Group> getGroups() {
        return groups;
    }

    public String getGroupString() {
        String groupString = "";
        for (int i = 0; i < groups.size(); i++) {
            Group group = groups.get(i);
            if (i == 0) {
                groupString = group.getName();
            } else {
                groupString = groupString + " | " + group.getName();
            }
        }
        return groupString;
    }

    public boolean isPhaseMember(Function phase) {

        if (phase == this) {
            return true;
        }

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

    public boolean isGroupMember(String groupName) {
        for (Iterator<Group> it = this.groups.iterator(); it.hasNext();) {
            Group group = it.next();
            if (group.getName().equals(groupName)) {
                return true;
            }
        }
        return false;
    }

    public boolean isCallPathFunction() {
        if (!callpathFunctionSet) {
            callpathFunction = isCallPathFunction(this.name);
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

    public int compareTo(Function o) {
        return this.id -  o.getID();
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