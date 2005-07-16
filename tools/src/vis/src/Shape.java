/*
 * Shape.java
 *
 * Copyright 2005                                                 
 * Department of Computer and Information Science, University of Oregon
 */
package edu.uoregon.tau.vis;

import net.java.games.jogl.*;


/**
 * This interface is implemented by anything that the visRenderer draws.
 *    
 * <P>CVS $Id: Shape.java,v 1.2 2005/07/16 00:21:07 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.2 $
 */
public interface Shape {

    /**
     * Renders to the given VisRenderer.
     * @param visRenderer the VisRenderer to use.
     */
    public void render(VisRenderer visRenderer);
   
}
