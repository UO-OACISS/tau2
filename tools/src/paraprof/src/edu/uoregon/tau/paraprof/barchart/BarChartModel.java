package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;
import java.awt.event.MouseEvent;

import javax.swing.JComponent;

public interface BarChartModel {

    public int getNumRows();

    public String getRowLabel(int row);
    public String getValueLabel(int row, int subIndex);

    public double getValue(int row, int subIndex);


    public Color getValueColor(int row, int subIndex);

    public Color getValueHighlightColor(int row, int subIndex);

    public int getSubSize();
    
    public void reportValueClick(int row, int subIndex, MouseEvent e, JComponent owner);
    public void reportRowLabelClick(int row, MouseEvent e, JComponent owner);
    //public void reportValueLabelClick(int row, int subIndex);
    
    public String getValueToolTipText(int row, int subIndex);
    public String getRowLabelToolTipText(int row);
    
    public String getOtherToolTopText(int row);
    
    public void addBarChartModelListener(BarChartModelListener l);
    public void removeBarChartModelListener(BarChartModelListener l);

    
    public void reloadData();
}
