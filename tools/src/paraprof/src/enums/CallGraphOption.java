package edu.uoregon.tau.paraprof.enums;


/**
 * type-safe enum pattern for callgraph options
 *    
 * TODO : nothing, this class is complete
 *
 * <P>CVS $Id: CallGraphOption.java,v 1.1 2005/09/26 21:12:45 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public class CallGraphOption {

    private final String name;
    
    private CallGraphOption(String name) { this.name = name; }
    
    public String toString() { return name; }

    public static final CallGraphOption STATIC = new CallGraphOption("Static");
    public static final CallGraphOption NAME_LENGTH = new CallGraphOption("Name Length");
    public static final CallGraphOption EXCLUSIVE = new CallGraphOption("Exclusive");
    public static final CallGraphOption INCLUSIVE = new CallGraphOption("Inclusive");
    public static final CallGraphOption EXCLUSIVE_PER_CALL = new CallGraphOption("Exclusive per Call");
    public static final CallGraphOption INCLUSIVE_PER_CALL = new CallGraphOption("Inclusive per Call");
    public static final CallGraphOption NUMCALLS = new CallGraphOption("Number of Calls");
    public static final CallGraphOption NUMSUBR = new CallGraphOption("Number of Child Calls");


}
