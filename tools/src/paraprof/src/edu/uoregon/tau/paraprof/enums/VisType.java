/*
 * Created on Mar 15, 2005
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
public class VisType {

    private final String name;
    private VisType(String name) { this.name = name; }
    public String toString() { return name; }
    
    public static final VisType TRIANGLE_MESH_PLOT = new VisType("Triangle Mesh");
    public static final VisType BAR_PLOT = new VisType("Bar Plot");
    public static final VisType SCATTER_PLOT = new VisType("Scatter Plot");
    //public static final VisType KIVIAT_TUBE = new VisType("Kiviat Tube");
    
    
    
}
