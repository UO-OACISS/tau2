package glue;

import common.PerfExplorerException;

public final class PerformanceDifferenceType {
    /**
     * One attribute, the name - it is transient so it is not serialized
     */
    private final transient String _name;

    /**
     * Static instances of the difference values.
     */
    public final static PerformanceDifferenceType SAME = new PerformanceDifferenceType("same");
    public final static PerformanceDifferenceType SLOWER = new PerformanceDifferenceType("slower");
    public final static PerformanceDifferenceType FASTER = new PerformanceDifferenceType("faster");
    
    /**
     * The constructor is private, so this class cannot be instantiated.
     * @param name
     */
    private PerformanceDifferenceType(String name) {
        this._name = name;
    }
    
    /**
     * Only one public method, to return the name of the difference type.
     * @return the name of the difference type
     */
    public String toString() {
        return this._name;
    }
    
	/**
	 * Convert a string value to an actual difference type.
	 *
	 * @param type engine type string
	 * @return the difference type
	 */
	public static PerformanceDifferenceType getType (String type) throws PerfExplorerException {
		PerfExplorerException e = new PerfExplorerException ("Unknown difference type.");
		if (type == null)
			throw (e);
		if (type.equalsIgnoreCase(SAME.toString()))
			return SAME;
		if (type.equalsIgnoreCase(SLOWER.toString()))
			return SLOWER;
		if (type.equalsIgnoreCase(FASTER.toString()))
			return FASTER;
		throw (e);
	}	    
}
