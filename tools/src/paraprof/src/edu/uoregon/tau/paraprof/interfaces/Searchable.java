
package edu.uoregon.tau.paraprof.interfaces;

import java.awt.Color;


public interface Searchable {
    
    // returns false if not found
    public boolean setSearchString(String searchString);
    
    public void setSearchHighlight(boolean highlight);
    public void setSearchMatchCase(boolean matchCase);
    
    public boolean searchNext();
    public boolean searchPrevious();
    
    
    public final static Color searchColor = new Color(102, 255, 102);
    public final static Color highlightColor = Color.YELLOW;
    public final static Color selectionColor = new Color(184, 207, 229);
}
