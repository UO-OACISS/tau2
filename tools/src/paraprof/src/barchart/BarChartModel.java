package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;
import java.awt.event.MouseEvent;

import javax.swing.JComponent;

import edu.uoregon.tau.paraprof.DataSorter;

/**
 * Interface that BarChart will use to interrogate a model for data.
 * Rather than implementing this interface, clients should extend from
 * AbstractBarChartModel, which implements listening code.
 * 
 * <P>CVS $Id: BarChartModel.java,v 1.2 2009/11/05 09:43:32 khuck Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 *
 * @see AbstractBarChartModel
 */
public interface BarChartModel {

    public int getNumRows();

    /**
     * Retrieves the label for a given row
     * @param row the row requested
     * @return the label for the requested row
     */
    public String getRowLabel(int row);
    
    /**
     * Retrieves the label for a given value.  Value labels are not used in 
     * multi-graphs at the present time. 
     * @param row the row requested
     * @param subIndex the subIndex in that row
     * @return the label for the requested row/subIndex pair
     */
    public String getValueLabel(int row, int subIndex);

    /**
     * Retrieves the value for a given row/subIndex pair.  Negative values are
     * interpreted as null at the present time, subject to change.
     * @param row row requested
     * @param subIndex subIndex requested
     * @return value for pair
     */
    public double getValue(int row, int subIndex);


    public Color getValueColor(int row, int subIndex);

    public Color getValueHighlightColor(int row, int subIndex);

    /**
     * Retrieve the number of bars per row (1 unless a multi-graph)
     * @return the number of bars per row
     */
    public int getSubSize();
    
    
    public void fireValueClick(int row, int subIndex, MouseEvent e, JComponent owner);
    public void fireRowLabelClick(int row, MouseEvent e, JComponent owner);
    
    public String getValueToolTipText(int row, int subIndex);
    public String getRowLabelToolTipText(int row);
    

    /**
     * Retrieve the tooltip for the "other" portion of a stacked/unstacked multi bar graph
     * @param row the row requested
     * @return tooltip for the "other" portion
     */
    public String getOtherToolTopText(int row);
    
    
    // Listener handling
    public void addBarChartModelListener(BarChartModelListener l);
    public void removeBarChartModelListener(BarChartModelListener l);

 
    
    /**
     * Requests that the model reload it's data. This method differs from all others
     * in that BarChart will never call it, it's merely a convienience for the real 
     * owners of the BarChartModels 
     */
    public void reloadData();
    
    /**
     * Return the DataSorter object, so we can find out what metric this is
     */
    public DataSorter getDataSorter();
}
