/*
 * Name: CallPathUtilFuncs.java Author: Robert Bell Description:
 */

package edu.uoregon.tau.dms.dss;

import java.util.*;

public class CallPathUtilFuncs {

    public CallPathUtilFuncs() {
    }

    public static boolean isAvailable(Iterator l) {
        Function func = null;
        String s = null;
        boolean result = false;
        while (l.hasNext()) {
            func = (Function) l.next();
            s = func.getName();
            if (s != null) {
                if (s.indexOf("=>") > 0) {
                    func.setCallPathObject(true);
                    result = true;
                }
            }
        }
        return result;
    }

    public static int buildRelations(TrialData td) {
        Function callPath = null;
        Function child = null;
        Function parent = null;
        String s = null;
        String parentName = null;
        String childName = null;
        int location = -1;

        for (Iterator it = td.getFunctions(); it.hasNext();) {
            callPath = (Function) it.next();
            s = callPath.getName();
            location = s.lastIndexOf("=>");
            if (location > 0) {
                childName = s.substring(location + 3, (s.length() - 2));
                s = s.substring(0, location - 1);
                location = s.lastIndexOf("=>");
                if (location > 0) {
                    parentName = s.substring(location + 3);
                } else
                    parentName = s;

                //Update parent/child relationships.
                parent = td.getFunction(parentName);
                child = td.getFunction(childName);

                if (parent == null || child == null) {
                    return -1;
                }

                if (parent != null)
                    parent.addChild(child, callPath);
                if (child != null)
                    child.addParent(parent, callPath);
            }
        }
        return 0;
    }

    public static void buildThreadRelations(TrialData td, edu.uoregon.tau.dms.dss.Thread thread) {

        if (thread.relationsBuilt())
            return;

        for (Iterator it = thread.getFunctionListIterator(); it.hasNext();) {
            FunctionProfile callPath = (FunctionProfile) it.next();
            if (callPath != null && callPath.isCallPathObject()) {

                String s = callPath.getName();
                int location = s.lastIndexOf("=>");

                if (location > 0) {
                    String childName = s.substring(location + 3, (s.length() - 2));
                    s = s.substring(0, location - 1);
                    location = s.lastIndexOf("=>");

                    String parentName = null;

                    if (location > 0)
                        parentName = s.substring(location + 3);
                    else
                        parentName = s;

                    //Update parent/child relationships.
                    FunctionProfile parent = thread.getFunctionProfile(td.getFunction(parentName));
                    FunctionProfile child = thread.getFunctionProfile(td.getFunction(childName));

                    if (parent == null || child == null) {
                        
                        //System.out.println("Something has gone horribly wrong!");
                        return;
                    }

                    if (parent != null)
                        parent.addChild(child);
                    if (child != null)
                        child.addParent(parent);
                }
            }
        }
    }

    public static void trimCallPathData(TrialData td, edu.uoregon.tau.dms.dss.Thread thread) {

        //Create a pruned list from the global list.
        //Want to grab a reference to the global list as
        //this list contains null references for mappings
        //which do not exist. Makes lookup much faster.
        Vector threadFunctionList = thread.getFunctionList();

        //Check to make sure that we have not trimmed before.
        if (thread.trimmed())
            return;

        for (Iterator l1 = td.getFunctions(); l1.hasNext();) {
            Function function = (Function) l1.next();

            if ((function.getID()) < (threadFunctionList.size())) { // only consider those that are possible on this thread

                FunctionProfile fp = (FunctionProfile) threadFunctionList.elementAt(function.getID());
                if ((!(function.isCallPathObject())) && (fp != null)) {
                    for (Iterator l2 = function.getParents(); l2.hasNext();) {
                        // get parent
                        Function parent = (Function) l2.next();

                        // iterate through the set of the parents callpaths

                        for (Iterator l3 = function.getParentCallPathIterator(parent); l3.hasNext();) {
                            // Only add this parent if there is an existing
                            // callpath to which this rightfully parent belongs.

                            Function callPath = (Function) l3.next();

                            if ((callPath.getID() < threadFunctionList.size())
                                    && (threadFunctionList.elementAt(callPath.getID()) != null))
                                fp.addParent(parent, callPath);
                            // Since the callpath is present, parent is, so this is safe.
                        }
                    }

                    for (Iterator l2 = function.getChildren(); l2.hasNext();) {

                        // get child 
                        Function child = (Function) l2.next();

                        for (Iterator l3 = function.getChildCallPathIterator(child); l3.hasNext();) {
                            Function callPath = (Function) l3.next();

                            //Only add this child if there is an existing
                            // callpath to which
                            //this rightfully child belongs.

                            if ((callPath.getID() < threadFunctionList.size())
                                    && (threadFunctionList.elementAt(callPath.getID()) != null))
                                fp.addChild(child, callPath);
                        }
                    }
                }
            }
        }

        //Set this thread to indicate that it has been trimmed.
        thread.setTrimmed(true);

    }
}