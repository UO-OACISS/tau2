/*
 * Plot.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;


import java.util.Observer;

import javax.swing.JPanel;

/**
 * Interface for plots.
 *    
 * TODO: selected row/column only apply to a subset of the plots, maybe they 
 * don't belong here.
 *
 * <P>CVS $Id: Plot.java,v 1.3 2006/09/01 20:18:08 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.3 $
 */
public interface Plot extends Shape, Observer {

    
    /**
     * Sets the size of this Plot.
     * @param xSize size in the x direction.
     * @param ySize size in the y direction.
     * @param zSize size in the z direction.
     */
    public void setSize(float xSize, float ySize, float zSize);    
    
    
    /**
     * Returns the width.
     * @return the width.
     */
    public float getWidth();
    /**
     * Returns the depth
     * @return the depth.
     */
    public float getDepth();
    /**
     * Returns the height.
     * @return the height.
     */
    public float getHeight();
    
    
    /**
     * Creates a Swing JPanel with controls for this object.  These controls will 
     * change the state of the axes and automatically call visRenderer.redraw().<p>
     * 
     * When getControlPanel() is called, the controls will represent the current
     * values for the object, but currently, they will not stay in sync if the values
     * are changed using the public methods.  For example, if you call "setEnabled(false)"
     * The JCheckBox will not be set to unchecked.  This functionality could be added if
     * requested.
     * 
     * @param visRenderer The associated VisRenderer
     * @return the control panel for this component
     */
    public JPanel getControlPanel(VisRenderer visRenderer);

    /**
     * Returns the Axes associated with this Plot.
     * @return the associated Axes.
     * @see edu.uoregon.tau.vis.Plot#getAxes()
     */
    public Axes getAxes();
    
    
    
    /**
     * Sets the <tt>Axes</tt> for this <tt>Plot</tt>
     * @param axes the <tt>Axes</tt> to use.
     */
    public void setAxes(Axes axes);

    
    /**
     * Get the current associated <tt>ColorScale</tt>.
     * @return the currently associated <tt>ColorScale</tt>.
     */
    public ColorScale getColorScale();

    /**
     * Sets the associated <tt>ColorScale</tt>.  
     * This <tt>Plot</tt> will use this <tt>ColorScale</tt> to resolve colors.
     * @param colorScale The <tt>ColorScale</tt>
     */
    public void setColorScale(ColorScale colorScale);
        
        
    /**
     * Cleans up display lists.
     */
    public void cleanUp();
    
    
    /**
     * Returns the name of this Plot.
     * @return the name of this Plot.
     */
    public String getName();
    
    

    /**
     * Returns the currently selected row.
     * @return the currently selected row.
     */
    public int getSelectedRow();
    
    /**
     * Sets the selected row.
     * @param selectedRow the selected row.
     */
    public void setSelectedRow(int selectedRow);
    
    /**
     * Returns the currently selected column.
     * @return the currently selected column.
     */
    public int getSelectedCol();

    /**
     * Sets the selected column.
     * @param selectedCol the selected column.
     */
    public void setSelectedCol(int selectedCol);
    
}
