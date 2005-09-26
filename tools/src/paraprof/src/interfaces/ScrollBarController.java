package edu.uoregon.tau.paraprof.interfaces;

import java.awt.Dimension;

/**
 * Interface that allows for control over scrollbar positioning, needed for searching
 *    
 * TODO : ...
 *
 * <P>CVS $Id: ScrollBarController.java,v 1.1 2005/09/26 21:12:47 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public interface ScrollBarController {
    public void setVerticalScrollBarPosition(int position);
    public void setHorizontalScrollBarPosition(int position);


    public Dimension getThisViewportSize();
}
