/* 
 PPThread.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.paraprof;

import java.util.*;

import edu.uoregon.tau.dms.dss.*;

public class PPThread {


    public PPThread(edu.uoregon.tau.dms.dss.Thread thread, ParaProfTrial ppTrial) {
        this.ppTrial = ppTrial;
        this.thread = thread;
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

    public void addFunction(PPFunctionProfile ppFunctionProfile) {
        functions.addElement(ppFunctionProfile);
    }

    public void addUserevent(PPFunctionProfile ppFunctionProfile) {
        userevents.addElement(ppFunctionProfile);
    }

    public Vector getFunctionList() {
        return functions;
    }

    public ListIterator getFunctionListIterator() {
        return new DssIterator(functions);
    }

    public Vector getUsereventList() {
        return userevents;
    }

    public ListIterator getUsereventListIterator() {
        return new DssIterator(userevents);
    }

    
    
    
    
    
    
    private double maxExclusive;
    private double maxExclusivePercent;
    private double maxInclusive;
    private double maxInclusivePercent;
    private double maxNumCalls;
    private double maxNumSubr;
    private double maxInclusivePerCall;

    public double getMaxExclusive() {
        return maxExclusive;
    }

    
    public double getMaxExclusivePercent() {
        return maxExclusivePercent;
    }


    public double getMaxInclusive() {
        return maxInclusive;
    }
    public double getMaxInclusivePercent() {
        return maxInclusivePercent;
    }
    public double getMaxNumCalls() {
        return maxNumCalls;
    }
    public double getMaxNumSubr() {
        return maxNumSubr;
    }
    public double getMaxInclusivePerCall() {
        return maxInclusivePerCall;
    }

    
    
    public double getMaxValue(int valueType, boolean percent) {
        double maxValue = 0;
        switch (valueType) {
        case 2:
            if (percent)
                maxValue = maxExclusivePercent;
            else
                maxValue = maxExclusive;
            break;
        case 4:
            if (percent)
                maxValue = maxInclusivePercent;
            else
                maxValue = maxInclusive;
            break;
        case 6:
            maxValue = maxNumCalls;
            break;
        case 8:
            maxValue = maxNumSubr;
            break;
        case 10:
            maxValue = maxInclusivePerCall;
            break;
        default:
            throw new RuntimeException("Invalid Value Type: " + valueType);
        }
        return maxValue;   
        
    }
    
    public Vector getSortedFunctionProfiles(int sortType, boolean getAll) {
        Vector newList = null;

        Vector functionList = thread.getFunctionProfiles();
        newList = new Vector();

        maxExclusive = 0;
        maxExclusivePercent = 0;
        maxInclusive = 0;
        maxInclusivePercent = 0;
        maxNumCalls = 0;
        maxNumSubr = 0;
        maxInclusivePerCall = 0;
        
        for (Enumeration e1 = functionList.elements(); e1.hasMoreElements();) {
            FunctionProfile functionProfile = (FunctionProfile) e1.nextElement();
            if (functionProfile != null) {
                if (getAll || ppTrial.displayFunction(functionProfile.getFunction())) {
                    PPFunctionProfile ppFunctionProfile = new PPFunctionProfile(ppTrial, thread, functionProfile);
                    ppFunctionProfile.setSortType(sortType);
                    newList.addElement(ppFunctionProfile);

                 
                    maxExclusive = Math.max(maxExclusive, functionProfile.getExclusive(ppTrial.getSelectedMetricID()));
                    maxInclusive = Math.max(maxInclusive, functionProfile.getInclusive(ppTrial.getSelectedMetricID()));
                    maxExclusivePercent = Math.max(maxExclusivePercent, functionProfile.getExclusivePercent(ppTrial.getSelectedMetricID()));
                    maxInclusivePercent = Math.max(maxInclusivePercent, functionProfile.getInclusivePercent(ppTrial.getSelectedMetricID()));
                    maxNumCalls = Math.max(maxNumCalls, functionProfile.getNumCalls());
                    maxNumSubr = Math.max(maxNumSubr, functionProfile.getNumSubr());
                    maxInclusivePerCall = Math.max(maxInclusivePerCall, functionProfile.getInclusivePerCall(ppTrial.getSelectedMetricID()));
                    
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

    int miscXBeg;
    int miscXEnd;
    int miscYBeg;
    int miscYEnd;

    
    private ParaProfTrial ppTrial;
    edu.uoregon.tau.dms.dss.Thread thread = null;
    Vector functions = new Vector();
    Vector userevents = new Vector();
    //To aid with drawing searches.
    int yDrawCoord = -1;
}
