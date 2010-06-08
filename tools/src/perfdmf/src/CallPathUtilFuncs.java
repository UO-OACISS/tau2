/*
 * Name: CallPathUtilFuncs.java 
 * Author: Robert Bell 
 * Description:
 */

package edu.uoregon.tau.perfdmf;

import java.util.Iterator;

public class CallPathUtilFuncs {

    private CallPathUtilFuncs() {
    }

    public static boolean checkCallPathsPresent(Iterator l) {
        while (l.hasNext()) {
            Function function = (Function) l.next();
            
            if (function.isCallPathFunction()) {
                return true;
            }
        }
        return false;
    }

    
    public static boolean containsDoublePath(String str) {
        int loc = str.indexOf("=>");
        if (loc != -1) {
            str = str.substring(loc+2);
            if (str.indexOf("=>") != -1) {
                return true;
            }
        }
        return false;
    }

    public static void buildThreadRelations(DataSource dataSource, edu.uoregon.tau.perfdmf.Thread thread) {

        if (thread.relationsBuilt()) {
            return;
        }

        // we want to skip the TAU_CALLPATH_DERIVED data
        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
        
        for (Iterator it = thread.getFunctionProfileIterator(); it.hasNext();) {
            FunctionProfile callPath = (FunctionProfile) it.next();
            if (callPath != null && callPath.isCallPathFunction() && !callPath.getFunction().isGroupMember(derived)) {

                String s = callPath.getName();
                int location = s.lastIndexOf("=>");

                if (location > 0) {
                    String childName = s.substring(location + 2, s.length());
                    s = s.substring(0, location);
                    location = s.lastIndexOf("=>");

                    String parentName = null;

                    if (location > 0) {
                        parentName = s.substring(location + 2);
                    } else {
                        parentName = s;
                    }
                    
                    Function parentFunction = dataSource.getFunction(parentName);
                    Function childFunction = dataSource.getFunction(childName);

                    if (parentFunction == null || childFunction == null) {
                        System.err.println ("Warning: Callpath data not complete: " + parentName + " => " + childName);
                        continue;

                    }
                    //Update parent/child relationships.
                    FunctionProfile parent = thread.getFunctionProfile(dataSource.getFunction(parentName));
                    FunctionProfile child = thread.getFunctionProfile(dataSource.getFunction(childName));

                    if (parent == null || child == null) {
                        System.err.println ("Warning: Callpath data not complete: " + parentName + " => " + childName);
                        continue;
                    }

                    if (parent != null) {
                        parent.addChildProfile(child, callPath);
                    }
                    if (child != null) {
                        child.addParentProfile(parent, callPath);
                    }
                }
            }
        }
    }
}