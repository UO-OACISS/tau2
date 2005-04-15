package edu.uoregon.tau.paraprof.vis;

import net.java.games.jogl.*;


/**
 * This interface is implemented by anything that the visRenderer draws
 *    
 * <P>CVS $Id: Shape.java,v 1.2 2005/04/15 01:29:03 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.2 $
 */
public interface Shape {

    public void render(VisRenderer visRenderer);
   
}
