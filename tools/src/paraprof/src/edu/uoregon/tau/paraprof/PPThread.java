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

    public PPThread() {
    }

    public PPThread(edu.uoregon.tau.dms.dss.Thread thread) {
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

    //Instance data.
    edu.uoregon.tau.dms.dss.Thread thread = null;
    Vector functions = new Vector();
    Vector userevents = new Vector();
    //To aid with drawing searches.
    int yDrawCoord = -1;
}
