/*
 * Created on Mar 3, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package edu.uoregon.tau.paraprof.enums;

/**
 * @author amorris
 *
 * TODO ...
 */
public class CallGraphOption {

    private final String name;
    
    private CallGraphOption(String name) { this.name = name; }
    
    public String toString() { return name; }

    public static final CallGraphOption STATIC = new CallGraphOption("Static");
    public static final CallGraphOption NAME_LENGTH = new CallGraphOption("Name Length");
    public static final CallGraphOption EXCLUSIVE = new CallGraphOption("Exclusive Value");
    public static final CallGraphOption INCLUSIVE = new CallGraphOption("Inclusive Value");
    public static final CallGraphOption EXCLUSIVE_PER_CALL = new CallGraphOption("Exclusive Per Call Value");
    public static final CallGraphOption INCLUSIVE_PER_CALL = new CallGraphOption("Inclusive Per Call Value");
    public static final CallGraphOption NUMCALLS = new CallGraphOption("Number of Calls");
    public static final CallGraphOption NUMSUBR = new CallGraphOption("Number of Subroutines");


}
