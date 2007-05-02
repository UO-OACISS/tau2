package edu.uoregon.tau.paraprof.enums;


/**
 * type-safe enum pattern for type of sorting
 *    
 * TODO : nothing, this class is complete
 *
 * <P>CVS $Id: SortType.java,v 1.2 2007/05/02 19:45:06 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class SortType {

    private final String name;
    
    private SortType(String name) { this.name = name; }
    
    public String toString() { return name; }

    public static final SortType MEAN_VALUE = new SortType("mean_value");
    public static final SortType NCT = new SortType("nct");
    public static final SortType VALUE = new SortType("value");
    public static final SortType NAME = new SortType("name");
}