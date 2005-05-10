package edu.uoregon.tau.paraprof.enums;

import edu.uoregon.tau.paraprof.*;
/**
 * type-safe enum pattern
 */

//public abstract class SortType {
//
//    private final String name;
//    
//    private SortType(String name) { this.name = name; }
//    
//    public String toString() { return name; }
//    
//    
//    public static final SortType MEAN = new SortType("mean") {
//        int makeComparison(PPFunctionProfile a, PPFunctionProfile b, ValueType valueType) {
//            return Double.compare(a.getComparisonValue(), b.getComparisonValue());
//            
////            return valueType.makeComparison(a.getMeanProfile(), b.getMeanProfile(), a.getDataSorter().getSelectedMetricID());
//        }
//    };
//
//    public static final SortType NCT = new SortType("nct") {
//        int makeComparison(PPFunctionProfile a, PPFunctionProfile b, ValueType valueType) {
//          if (a.getNodeID() != b.getNodeID())
//              return a.getNodeID() - b.getNodeID();
//          else if (a.getContextID() != b.getContextID())
//              return a.getContextID() - b.getContextID();
//          else
//              return a.getThreadID() - b.getThreadID();
//        }
//    };
//
//
//    public static final SortType NAME = new SortType("name") {
//        int makeComparison(PPFunctionProfile a, PPFunctionProfile b, ValueType valueType) {
//            return (a.getFunctionName()).compareTo(b.getFunctionName());
//        }
//    };
//
//    public static final SortType VALUE = new SortType("value") {
//        int makeComparison(PPFunctionProfile a, PPFunctionProfile b, ValueType valueType) {
//            return valueType.makeComparison(a.getFunctionProfile(), b.getFunctionProfile(), a.getDataSorter().getSelectedMetricID());
//        }
//    };
//
//    
////    public static final SortType MEAN = new SortType("mean");
////    public static final SortType NCT = new SortType("nct");
////    public static final SortType VALUE = new SortType("value");
////    
//    abstract int makeComparison (PPFunctionProfile a, PPFunctionProfile b, ValueType valueType);
//    
//}

public class SortType {

    private final String name;
    
    private SortType(String name) { this.name = name; }
    
    public String toString() { return name; }

    public static final SortType MEAN_VALUE = new SortType("mean_value");
    public static final SortType NCT = new SortType("nct");
    public static final SortType VALUE = new SortType("value");
    public static final SortType NAME = new SortType("name");
}