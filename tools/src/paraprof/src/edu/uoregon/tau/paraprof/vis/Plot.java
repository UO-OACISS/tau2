package edu.uoregon.tau.paraprof.vis;


import java.util.Observer;

import javax.swing.JPanel;


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
