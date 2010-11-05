package edu.uoregon.tau.paraprof.enums;

/**
 * type-safe enum pattern for visualization type
 *    
 * TODO : nothing, this class is complete
 *
 * <P>CVS $Id: VisType.java,v 1.2 2007/12/07 02:05:21 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class VisType {

    private final String name;
    private VisType(String name) { this.name = name; }
    public String toString() { return name; }
    
    public static final VisType TRIANGLE_MESH_PLOT = new VisType("Triangle Mesh");
    public static final VisType BAR_PLOT = new VisType("Bar Plot");
    public static final VisType SCATTER_PLOT = new VisType("Scatter Plot");
    public static final VisType CALLGRAPH = new VisType("Callgraph");
    public static final VisType TOPO_PLOT = new VisType("Topology Plot");
    //public static final VisType KIVIAT_TUBE = new VisType("Kiviat Tube");
    
    
    
}
