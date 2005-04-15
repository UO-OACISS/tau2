package edu.uoregon.tau.paraprof.vis;


import java.util.Observer;

import javax.swing.JPanel;

/**
 * Interface for plots
 *    
 * TODO: selected row/column only apply to a subset of the plots, maybe they 
 * don't belong here.
 *
 * <P>CVS $Id: Plot.java,v 1.3 2005/04/15 01:29:02 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.3 $
 */
public interface Plot extends Shape, Observer {

    
    public void setSize(float xSize, float ySize, float zSize);    
    
    
    public float getWidth();
    public float getDepth();
    public float getHeight();
    
    
    public JPanel getControlPanel(VisRenderer visRenderer);

    public Axes getAxes();
    
    public void cleanUp();
    
    public String getName();
    
    
    
    public int getSelectedRow();
    public void setSelectedRow(int selectedRow);
    public int getSelectedCol();
    public void setSelectedCol(int selectedCol);
    
}
