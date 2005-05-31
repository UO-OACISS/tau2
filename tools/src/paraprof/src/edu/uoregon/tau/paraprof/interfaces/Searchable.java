
package edu.uoregon.tau.paraprof.interfaces;

import java.awt.Color;

/**
 * Interface for searching text
 *    
 * TODO : ...
 *
 * <P>CVS $Id: Searchable.java,v 1.3 2005/05/31 23:21:50 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.3 $
 */
public interface Searchable {
    
    // returns false if not found
    public boolean setSearchString(String searchString);
    
    public void setSearchHighlight(boolean highlight);
    public void setSearchMatchCase(boolean matchCase);
    
    public boolean searchNext();
    public boolean searchPrevious();
    
    
    public final static Color searchColor = new Color(102, 255, 102);
    public final static Color highlightColor = Color.yellow;
    public final static Color selectionColor = new Color(184, 207, 229);
}
