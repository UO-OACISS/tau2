/*
 * Created on Mar 2, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package edu.uoregon.tau.paraprof.enums;

import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.dms.dss.*;

/**
 * @author amorris
 *
 * TODO ...
 */
public abstract class ValueType {

    private final String name;

    private ValueType(String name) {
        this.name = name;
    }

    public String toString() {
        return name;
    }

    public static final ValueType EXCLUSIVE = new ValueType("Exclusive") {

        //        double getValue (PPFunctionProfile ppFunctionProfile) {
        //            return ppFunctionProfile.getExclusiveValue();
        //        }

        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getExclusive(metric);
        }

        public double getThreadMaxValue(edu.uoregon.tau.dms.dss.Thread thread, int metric) {
        	return thread.getMaxExclusive(metric);
        }

        //        int makeComparison(FunctionProfile a, FunctionProfile b, int metric) {
        //            return Double.compare(a.getExclusive(metric), b.getExclusive(metric));
        //        }

    };

    public static final ValueType EXCLUSIVE_PERCENT = new ValueType("Exclusive percent") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getExclusivePercent(metric);
        }
        public double getThreadMaxValue(edu.uoregon.tau.dms.dss.Thread thread, int metric) {
        	return thread.getMaxExclusivePercent(metric);
        }
    };

    public static final ValueType INCLUSIVE = new ValueType("Inclusive") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getInclusive(metric);
        }
        public double getThreadMaxValue(edu.uoregon.tau.dms.dss.Thread thread, int metric) {
        	return thread.getMaxInclusive(metric);
        }
    };

    public static final ValueType INCLUSIVE_PERCENT = new ValueType("Inclusive percent") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getInclusivePercent(metric);
        }
        public double getThreadMaxValue(edu.uoregon.tau.dms.dss.Thread thread, int metric) {
        	return thread.getMaxInclusivePercent(metric);
        }
    };

    public static final ValueType NUMCALLS = new ValueType("Number of Calls") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getNumCalls();
        }
        public double getThreadMaxValue(edu.uoregon.tau.dms.dss.Thread thread, int metric) {
        	return thread.getMaxNumCalls();
        }
    };

    public static final ValueType NUMSUBR = new ValueType("Number of Child Calls") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getNumSubr();
        }
        public double getThreadMaxValue(edu.uoregon.tau.dms.dss.Thread thread, int metric) {
        	return thread.getMaxNumSubr();
        }
    };
    
    public static final ValueType INCLUSIVE_PER_CALL = new ValueType("Inclusive per Call") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getInclusivePerCall(metric);
        }
        public double getThreadMaxValue(edu.uoregon.tau.dms.dss.Thread thread, int metric) {
        	return thread.getMaxInclusivePerCall(metric);
        }
    };
    
    public static final ValueType EXCLUSIVE_PER_CALL = new ValueType("Exclusive per Call") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getExclusivePerCall(metric);
        }
        public double getThreadMaxValue(edu.uoregon.tau.dms.dss.Thread thread, int metric) {
        	return thread.getMaxExclusivePerCall(metric);
        }
    };

    //    public static final ValueType EXCLUSIVE_PERCENT = new ValueType("exclusive_percent");
    //    public static final ValueType INCLUSIVE = new ValueType("inclusive");
    //    public static final ValueType INCLUSIVE_PERCENT = new ValueType("inclusive_percent");
    //    public static final ValueType NUMCALLS = new ValueType("numcalls");
    //    public static final ValueType NUMSUBR = new ValueType("numsubr");
    //    public static final ValueType INCLUSIVE_PER_CALL = new ValueType("inclusive_per_call");
    //    public static final ValueType EXCLUSIVE_PER_CALL = new ValueType("exclusive_per_call");

    //    abstract int makeComparison (FunctionProfile a, FunctionProfile b, int metric);
    //abstract double getValue (PPFunctionProfile ppFunctionProfile);

    public abstract double getValue(FunctionProfile functionProfile, int metric);
    public abstract double getThreadMaxValue(edu.uoregon.tau.dms.dss.Thread thread, int metric);
}

//public class ValueType {
//    
//    private final String name;
//    
//    private ValueType(String name) { this.name = name; }
//    
//    public String toString() { return name; }
//    
//    public static final ValueType EXCLUSIVE = new ValueType("exclusive");
//    public static final ValueType EXCLUSIVE_PERCENT = new ValueType("exclusive_percent");
//    public static final ValueType INCLUSIVE = new ValueType("inclusive");
//    public static final ValueType INCLUSIVE_PERCENT = new ValueType("inclusive_percent");
//    public static final ValueType NUMCALLS = new ValueType("numcalls");
//    public static final ValueType NUMSUBR = new ValueType("numsubr");
//    public static final ValueType INCLUSIVE_PER_CALL = new ValueType("inclusive_per_call");
//    public static final ValueType EXCLUSIVE_PER_CALL = new ValueType("exclusive_per_call");
//}
