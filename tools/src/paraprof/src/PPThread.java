/* 
 PPThread.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.paraprof;

import java.util.*;

import edu.uoregon.tau.perfdmf.*;

public class PPThread {
    private int miscXBeg;
    private int miscXEnd;
    private int miscYBeg;
    private int miscYEnd;

    private ParaProfTrial ppTrial;
    private edu.uoregon.tau.perfdmf.Thread thread = null;
    private List functions = new ArrayList();
    private List userevents = new ArrayList();
    //To aid with drawing searches.
    private int yDrawCoord = -1;

    private double maxExclusivePercent;

    public PPThread(edu.uoregon.tau.perfdmf.Thread thread, ParaProfTrial ppTrial) {
        if (thread == null) {
            throw new ParaProfException("PPThread constructor called with null thread");
        }
        this.ppTrial = ppTrial;
        this.thread = thread;
    }

    public edu.uoregon.tau.perfdmf.Thread getThread() {
        return thread;
    }

    public int getNodeID() {
        return this.thread.getNodeID();
    }

    public int getContextID() {
        return this.thread.getContextID();
    }

    public int getThreadID() {
        return this.thread.getThreadID();
    }

    public String getName() {
        if (this.getNodeID() == -1) {
            return "mean";
        } else if (this.getNodeID() == -2) {
            return "total";
        } else if (this.getNodeID() == -3) {
            return "std. dev.";
        } else {
            return "n,c,t " + (this.getNodeID()) + "," + (this.getContextID()) + "," + (this.getThreadID());
        }
    }

    public void addFunction(PPFunctionProfile ppFunctionProfile) {
        functions.add(ppFunctionProfile);
    }

    public void addUserevent(PPFunctionProfile ppFunctionProfile) {
        userevents.add(ppFunctionProfile);
    }

    public List getFunctionList() {
        return functions;
    }

    public ListIterator getFunctionListIterator() {
        return functions.listIterator();
    }

    public List getUsereventList() {
        return userevents;
    }

    public ListIterator getUsereventListIterator() {
        return userevents.listIterator();
    }

    public double getMaxExclusivePercent() {
        return maxExclusivePercent;
    }

    public Vector getSortedFunctionProfiles(DataSorter dataSorter, boolean getAll) {
        Vector newList = null;

        List functionList = thread.getFunctionProfiles();
        newList = new Vector();

        maxExclusivePercent = 0;

        for (Iterator e1 = functionList.iterator(); e1.hasNext();) {
            FunctionProfile functionProfile = (FunctionProfile) e1.next();
            if (functionProfile != null) {
                if (getAll || ppTrial.displayFunction(functionProfile.getFunction()) && functionProfile.getFunction().isPhaseMember(dataSorter.getPhase())) {
                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(dataSorter, thread, functionProfile);
                    newList.addElement(ppFunctionProfile);
                    maxExclusivePercent = Math.max(maxExclusivePercent,
                            functionProfile.getExclusivePercent(ppTrial.getDefaultMetricID()));
                }
            }
        }
        Collections.sort(newList);
        return newList;
    }

    //Rest of the public functions
    public void setYDrawCoord(int yDrawCoord) {
        yDrawCoord = this.yDrawCoord;
    }

    public int getYDrawCoord() {
        return yDrawCoord;
    }

    public void setMiscCoords(int xBeg, int xEnd, int yBeg, int yEnd) {
        this.miscXBeg = xBeg;
        this.miscXEnd = xEnd;
        this.miscYBeg = yBeg;
        this.miscYEnd = yEnd;
    }

    public int getMiscXBeg() {
        return miscXBeg;
    }

    public int getMiscXEnd() {
        return miscXEnd;
    }

    public int getMiscYBeg() {
        return miscYBeg;
    }

    public int getMiscYEnd() {
        return miscYEnd;
    }

}
