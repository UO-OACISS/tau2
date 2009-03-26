/**
 * Created on Feb 13, 2006
 *
 */
package edu.uoregon.tau.perfexplorer.common;

import java.io.Serializable;


/**
 * This class is used as a typesafe enumeration.
 *
 * <P>CVS $Id: ChartDataType.java,v 1.5 2009/03/26 18:09:57 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.2
 * @since   0.2
 */
public final class ChartDataType implements Serializable {

    /**
     * One attribute, the name - it is transient so it is not serialized.
     */
    private final transient String _name;

    /**
     * Static instances of the engine.
     */
    public static final ChartDataType FRACTION_OF_TOTAL = 
        new ChartDataType("Fraction Of Total");
    public static final ChartDataType RELATIVE_EFFICIENCY = 
        new ChartDataType("Relative Efficiency");
    public static final ChartDataType TIMESTEPS_PER_SECOND = 
        new ChartDataType("Timesteps Per Second");
    public static final ChartDataType TOTAL_FOR_GROUP = 
        new ChartDataType("Total For Group");
    public static final ChartDataType RELATIVE_EFFICIENCY_EVENTS = 
        new ChartDataType("Relative Efficiency For All Events");
    public static final ChartDataType RELATIVE_EFFICIENCY_ONE_EVENT = 
        new ChartDataType("Relative Efficiency For One Event");
    public static final ChartDataType RELATIVE_EFFICIENCY_PHASES = 
        new ChartDataType("Relative Efficiency Across Phases");
    public static final ChartDataType FRACTION_OF_TOTAL_PHASES = 
        new ChartDataType("Phase Fraction of Total");
    public static final ChartDataType IQR_DATA = 
        new ChartDataType("IQR Data");
    public static final ChartDataType CORRELATION_DATA = 
        new ChartDataType("Correlation Data");
    public static final ChartDataType DISTRIBUTION_DATA = 
        new ChartDataType("Distribution Data");
    public static final ChartDataType PARAMETRIC_STUDY_DATA = 
        new ChartDataType("General Parametric Data");
    public static final ChartDataType CUSTOM = 
    	new ChartDataType("Custom");
    
    /**
     * The constructor is private, so this class cannot be instantiated.
     * @param name
     */
    private ChartDataType(String name) {
        this._name = name;
    }
    
    /**
     * Only one public method, to return the name of the data type
     * @return
     */
    public String toString() {
        return this._name;
    }

    // The following declarations are necessary for serialization
    private static int nextOrdinal = 0;
    private final int ordinal = nextOrdinal++;
    private static final ChartDataType[] VALUES = 
        {FRACTION_OF_TOTAL, RELATIVE_EFFICIENCY, TIMESTEPS_PER_SECOND,
         TOTAL_FOR_GROUP, RELATIVE_EFFICIENCY_EVENTS, 
         RELATIVE_EFFICIENCY_ONE_EVENT, RELATIVE_EFFICIENCY_PHASES,
         FRACTION_OF_TOTAL_PHASES, IQR_DATA, CORRELATION_DATA,
         DISTRIBUTION_DATA, PARAMETRIC_STUDY_DATA, CUSTOM };
    
    /**
     * This method is necessary, because we are serializing the object.
     * When the object is serialized, a NEW INSTANCE is created, and we
     * can't compare reference pointers.  When that is the case, replace the
     * newly instantiated object with a static reference.
     * 
     * @return
     * @throws java.io.ObjectStreamException
     */
    Object readResolve () throws java.io.ObjectStreamException
    {
        return VALUES[ordinal];    // Canonicalize
    }
}
