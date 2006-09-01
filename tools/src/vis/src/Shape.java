/*
 * Shape.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;

import net.java.games.jogl.*;


/**
 * This interface is implemented by anything that the visRenderer draws.
 *    
 * <P>CVS $Id: Shape.java,v 1.3 2006/09/01 20:18:08 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.3 $
 */
public interface Shape {

    /**
     * Renders to the given VisRenderer.
     * @param visRenderer the VisRenderer to use.
     */
    public void render(VisRenderer visRenderer);
   
}
