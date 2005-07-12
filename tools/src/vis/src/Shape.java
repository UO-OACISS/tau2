package edu.uoregon.tau.vis;

import net.java.games.jogl.*;


/**
 * This interface is implemented by anything that the visRenderer draws
 *    
 * <P>CVS $Id: Shape.java,v 1.1 2005/07/12 18:02:17 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.1 $
 */
public interface Shape {

    /**
     * Renders to the given VisRenderer.
     * @param visRenderer the VisRenderer to use.
     */
    public void render(VisRenderer visRenderer);
   
}
