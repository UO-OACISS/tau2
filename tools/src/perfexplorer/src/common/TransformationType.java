/**
 * Created on Feb 17, 2006
 *
 */
package edu.uoregon.tau.perfexplorer.common;

import java.io.Serializable;


/**
 * This class is used as a typesafe enumeration.
 *
 * <P>CVS $Id: TransformationType.java,v 1.6 2009/02/24 00:53:37 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.2
 * @since   0.2
 */
public final class TransformationType implements Serializable {

    /**
	 * 
	 */
	private static final long serialVersionUID = 6319821230629921757L;

	/**
     * One attribute, the name - it is transient so it is not serialized
     */
    private final transient String _name;

    /**
     * Static instances of the engine.
     */
    public final static TransformationType LINEAR_PROJECTION = 
        new TransformationType("Random Linear Projection (disabled)");
    public final static TransformationType OVER_X_PERCENT = 
        new TransformationType("Over X Percent");
    public final static TransformationType REGRESSION = 
        new TransformationType("PCA (disabled)");
    public final static TransformationType PERCENTAGE_OF_TOTAL = 
        new TransformationType("Percentage of Total");
    public final static TransformationType RANGE_OF_TOTAL = 
        new TransformationType("Range of Total");
    public final static TransformationType NONE = 
        new TransformationType("None");

//	public final static String PERCENTAGE_OF_TOTAL = "Percentage of Total";
// 	public final static String RANGE_OF_TOTAL = "Range of Total";

    /**
     * The constructor is private, so this class cannot be instantiated.
     * @param name
     */
    private TransformationType(String name) {
        this._name = name;
    }
    
    /**
     * Only one public method, to return the name of the engine
     * @return
     */
    public String toString() {
        return this._name;
    }
    
   
    // The following declarations are necessary for serialization
    private static int nextOrdinal = 0;
    private final int ordinal = nextOrdinal++;
    private static final TransformationType[] VALUES = 
        {LINEAR_PROJECTION, OVER_X_PERCENT, REGRESSION, PERCENTAGE_OF_TOTAL, RANGE_OF_TOTAL, NONE};
    
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

    /**
     * Method to return the dimension reduction methods.
     * @return
     */
    public static Object[] getDimensionReductions() {
//        Object[] options = {LINEAR_PROJECTION, OVER_X_PERCENT, REGRESSION, NONE};
        Object[] options = {NONE, OVER_X_PERCENT};
        return options;
    }
    
    /**
     * Method to return the normalization methods.
     * 
     * @return
     */
    public static Object[] getNormalizations() {
        Object[] options = {PERCENTAGE_OF_TOTAL, RANGE_OF_TOTAL, NONE};
        return options;
    }



}
