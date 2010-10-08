/**
 * Created on Feb 17, 2006
 *
 */
package edu.uoregon.tau.perfexplorer.common;

import java.io.Serializable;


/**
 * This class is used as a typesafe enumeration.
 *
 * <P>CVS $Id: AnalysisType.java,v 1.5 2009/11/18 17:45:41 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.2
 * @since   0.2
 */
public final class AnalysisType implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5829378530524627781L;

	/**
     * One attribute, the name - it is transient so it is not serialized
     */
    private final transient String _name;

    /**
     * Static instances of the engine.
     */
    public final static AnalysisType K_MEANS = new AnalysisType("K Means");
    public final static AnalysisType K_HARMONIC_MEANS = 
        new AnalysisType("K Harmonic Means");
    public final static AnalysisType GEM = 
        new AnalysisType("Gaussian Expectation-Maximization");
    public final static AnalysisType FUZZY_K_MEANS = 
        new AnalysisType("Fuzzy K Means");
    public final static AnalysisType CORRELATION_ANALYSIS = 
        new AnalysisType("Correlation Analysis");
    public final static AnalysisType HIERARCHICAL = new AnalysisType("Hierarchical");
    public final static AnalysisType DBSCAN = new AnalysisType("Density Based (DBSCAN)");
    
    /**
     * The constructor is private, so this class cannot be instantiated.
     * @param name
     */
    private AnalysisType(String name) {
        this._name = name;
    }
    
    /**
     * Only one public method, to return the name of the engine
     * @return
     */
    public String toString() {
        return this._name;
    }
    
    /**
     * Returns the AnalysisType specified by the String.
     * 
     * @param typeString
     * @return
     */
    public static AnalysisType fromString(String typeString) {
        for (int i = 0 ; i < VALUES.length ; i++) {
            if (typeString.equals(VALUES[i]._name))
                return VALUES[i];
        }
        return null;
    }
    
    // The following declarations are necessary for serialization
    private static int nextOrdinal = 0;
    private final int ordinal = nextOrdinal++;
    private static final AnalysisType[] VALUES = 
        {K_MEANS, K_HARMONIC_MEANS, GEM, FUZZY_K_MEANS, CORRELATION_ANALYSIS, HIERARCHICAL, DBSCAN};
    
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
     * Method to return the cluster methods.
     * @return
     */
    public static Object[] getClusterMethods() {
        //Object[] options = {K_MEANS, K_HARMONIC_MEANS, GEM, FUZZY_K_MEANS, HIERARCHICAL, DBASE};
        Object[] options = {K_MEANS, HIERARCHICAL, DBSCAN};
        return options;
    }
}
