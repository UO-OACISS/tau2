package edu.uoregon.tau.paraprof.enums;

import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.perfdmf.*;

/**
 * type-safe enum pattern for type of sorting
 *    
 * TODO : nothing, this class is complete
 *
 * <P>CVS $Id: ValueType.java,v 1.1 2005/09/26 21:12:46 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */

public abstract class ValueType {

    private final String name;
    public abstract double getValue(FunctionProfile functionProfile, int metric);
    public abstract double getThreadMaxValue(edu.uoregon.tau.perfdmf.Thread thread, int metric);
    public abstract String getSuffix(int units, ParaProfMetric ppMetric);
    public abstract int getUnits(int units, ParaProfMetric ppMetric);
        
    

    private ValueType(String name) {
        this.name = name;
    }

    public String toString() {
        return name;
    }

    private static String timeUnits(int units) {
        String string;
        switch (units) {
        case 0:
            string = " microseconds";
            break;
        case 1:
            string = " milliseconds";
            break;
        case 2:
            string = " seconds";
            break;
        case 3:
            string = " hh:mm:ss";
            break;

        default:
            throw new ParaProfException("Unexpected unit type: " + units);
        }   
        return string;
    }
    
    public static final ValueType EXCLUSIVE = new ValueType("Exclusive") {

        public double getValue(FunctionProfile functionProfile, int metric) {
            if (functionProfile.getFunction().isPhase()) {
                return functionProfile.getInclusive(metric);
            } else {
                return functionProfile.getExclusive(metric);
            }
        }

        public double getThreadMaxValue(edu.uoregon.tau.perfdmf.Thread thread, int metric) {
        	return thread.getMaxExclusive(metric);
        }

        public String getSuffix(int units, ParaProfMetric ppMetric) {
            if (!ppMetric.isTimeMetric())
                return " counts";
            return timeUnits(units);
        }

        public int getUnits(int units, ParaProfMetric ppMetric) {
            if (ppMetric.isTimeMetric())
                return units;
            return 0;
        }
    };

    public static final ValueType EXCLUSIVE_PERCENT = new ValueType("Exclusive percent") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            if (functionProfile.getFunction().isPhase()) {
                return functionProfile.getInclusivePercent(metric);
            } else {
                return functionProfile.getExclusivePercent(metric);
            }
        }
        public double getThreadMaxValue(edu.uoregon.tau.perfdmf.Thread thread, int metric) {
        	return thread.getMaxExclusivePercent(metric);
        }

        public String getSuffix(int units, ParaProfMetric ppMetric) {
            return " %";
        }

        public int getUnits(int units, ParaProfMetric ppMetric) {
            return 0;
        }
    };

    public static final ValueType INCLUSIVE = new ValueType("Inclusive") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getInclusive(metric);
        }
        public double getThreadMaxValue(edu.uoregon.tau.perfdmf.Thread thread, int metric) {
        	return thread.getMaxInclusive(metric);
        }
        public String getSuffix(int units, ParaProfMetric ppMetric) {
            if (!ppMetric.isTimeMetric())
                return " counts";
            return timeUnits(units);
        }
        public int getUnits(int units, ParaProfMetric ppMetric) {
            if (ppMetric.isTimeMetric())
                return units;
            return 0;
        }
    };

    public static final ValueType INCLUSIVE_PERCENT = new ValueType("Inclusive percent") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getInclusivePercent(metric);
        }
        public double getThreadMaxValue(edu.uoregon.tau.perfdmf.Thread thread, int metric) {
        	return thread.getMaxInclusivePercent(metric);
        }
        public String getSuffix(int units, ParaProfMetric ppMetric) {
            return " %";
        }
        public int getUnits(int units, ParaProfMetric ppMetric) {
            return 0;
        }
    };

    public static final ValueType NUMCALLS = new ValueType("Number of Calls") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getNumCalls();
        }
        public double getThreadMaxValue(edu.uoregon.tau.perfdmf.Thread thread, int metric) {
        	return thread.getMaxNumCalls();
        }
        public String getSuffix(int units, ParaProfMetric ppMetric) {
            return " calls";
        }
        public int getUnits(int units, ParaProfMetric ppMetric) {
            return 0;
        }
    };

    public static final ValueType NUMSUBR = new ValueType("Number of Child Calls") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getNumSubr();
        }
        public double getThreadMaxValue(edu.uoregon.tau.perfdmf.Thread thread, int metric) {
        	return thread.getMaxNumSubr();
        }
        public String getSuffix(int units, ParaProfMetric ppMetric) {
            return " calls";
        }
        public int getUnits(int units, ParaProfMetric ppMetric) {
            return 0;
        }
    };
    
    public static final ValueType INCLUSIVE_PER_CALL = new ValueType("Inclusive per Call") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getInclusivePerCall(metric);
        }
        public double getThreadMaxValue(edu.uoregon.tau.perfdmf.Thread thread, int metric) {
        	return thread.getMaxInclusivePerCall(metric);
        }
        public String getSuffix(int units, ParaProfMetric ppMetric) {
            if (!ppMetric.isTimeMetric())
                return " counts per call";
            return timeUnits(units) + " per call";
        }
        public int getUnits(int units, ParaProfMetric ppMetric) {
            if (ppMetric.isTimeMetric())
                return units;
            return 0;
        }
    };
    
    public static final ValueType EXCLUSIVE_PER_CALL = new ValueType("Exclusive per Call") {
        public double getValue(FunctionProfile functionProfile, int metric) {
            return functionProfile.getExclusivePerCall(metric);
        }
        public double getThreadMaxValue(edu.uoregon.tau.perfdmf.Thread thread, int metric) {
        	return thread.getMaxExclusivePerCall(metric);
        }
        public String getSuffix(int units, ParaProfMetric ppMetric) {
            if (!ppMetric.isTimeMetric())
                return " counts per call";
            return timeUnits(units) + " per call";
        }
        public int getUnits(int units, ParaProfMetric ppMetric) {
            if (ppMetric.isTimeMetric())
                return units;
            return 0;
        }
    };

    public static ValueType[] VALUES = { EXCLUSIVE, INCLUSIVE, EXCLUSIVE_PERCENT, INCLUSIVE_PERCENT, NUMCALLS, NUMSUBR, INCLUSIVE_PER_CALL, EXCLUSIVE_PER_CALL };

}

