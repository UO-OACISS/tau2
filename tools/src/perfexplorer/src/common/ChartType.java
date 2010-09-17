/**
 * Created on Feb 13, 2006
 *
 */
package edu.uoregon.tau.perfexplorer.common;

import java.io.Serializable;


/**
 * This class is used as a typesafe enumeration.
 *
 * <P>CVS $Id: ChartType.java,v 1.2 2009/02/24 00:53:36 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.2
 * @since   0.2
 */
public final class ChartType implements Serializable {

    /**
	 * 
	 */
	private static final long serialVersionUID = -7558040081827033537L;

	/**
     * One attribute, the name - it is transient so it is not serialized
     */
    private final transient int _value;

    /**
     * Static instances of the engine.
     * NOTE: you may add to this list, but DO NOT CHANGE THE ORDER!
     * 		 These values are stored in the database!!!
     */
    public final static ChartType DENDROGRAM = new ChartType(0);
    public final static ChartType HISTOGRAM = new ChartType(1);
    public final static ChartType VIRTUAL_TOPOLOGY = new ChartType(2);
    public final static ChartType CLUSTER_AVERAGES = new ChartType(3);
    public final static ChartType CLUSTER_MAXIMUMS = new ChartType(4);
    public final static ChartType CLUSTER_MINIMUMS = new ChartType(5);
    public final static ChartType PCA_SCATTERPLOT = new ChartType(6);
    public final static ChartType CORRELATION_SCATTERPLOT = new ChartType(7);
    
    /**
     * The constructor is private, so this class cannot be instantiated.
     * @param name
     */
    private ChartType(int value) {
        this._value = value;
    }
    
    // The following declarations are necessary for serialization
    private static int nextOrdinal = 0;
    private final int ordinal = nextOrdinal++;
    private static final ChartType[] VALUES = {DENDROGRAM, HISTOGRAM, 
    	VIRTUAL_TOPOLOGY, CLUSTER_AVERAGES, CLUSTER_MAXIMUMS, 
    	CLUSTER_MINIMUMS, PCA_SCATTERPLOT, CORRELATION_SCATTERPLOT};
    
    /**
     * Only one public method, to return the name of the engine
     * @return
     */
    public String toString() {
        return Integer.toString(_value);
    }
    
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
