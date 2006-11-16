/*
 * Shape.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;



/**
 * This interface is implemented by anything that the visRenderer draws.
 *    
 * <P>CVS $Id: Shape.java,v 1.4 2006/11/16 17:50:36 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.4 $
 */
public interface Shape {

    /**
     * Renders to the given VisRenderer.
     * @param visRenderer the VisRenderer to use.
     */
    public void render(VisRenderer visRenderer);
   
}
