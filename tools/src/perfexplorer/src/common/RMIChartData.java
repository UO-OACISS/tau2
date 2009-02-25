package edu.uoregon.tau.perfexplorer.common;

import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;

/**
 * This class represents the data to be used to create a scalability or
 * runtime breakdown chart.  
 *
 * <P>CVS $Id: RMIChartData.java,v 1.8 2009/02/25 19:51:46 wspear Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class RMIChartData implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5974297775713831594L;
	protected ChartDataType dataType = ChartDataType.FRACTION_OF_TOTAL;
	int rows = 0;
	int columns = 0;

	protected List<List<double[]>> newData = null;
	protected List<String> rowLabels = null;

	/**
	 * Constructor.
	 * 
	 * @param dataType
	 */
	public RMIChartData (ChartDataType dataType) {
		this.dataType = dataType;
		this.newData = new ArrayList<List<double[]>>();
		this.rowLabels = new ArrayList<String>();
	}

	/**
	 * Add a row of data fields (a new line on the chart)
	 * 
	 * @param label
	 */
	public void addRow(String label) {
		newData.add(new ArrayList<double[]>());
		rowLabels.add(label);
		rows++;
	}

	/**
	 * Add a column to the data (new point on a line on the chart)
	 * 
	 * @param rowIndex
	 * @param v1
	 * @param v2
	 */
	public void addColumn(int rowIndex, double v1, double v2) {
		List<double[]> row = newData.get(rowIndex);
		double[] values = new double[2];
		values[0] = v1;
		values[1] = v2;
		row.add(values);
		//PerfExplorerOutput.println("Added: " + rowIndex + ", " + v1 + ", " + v2);
	}
	
	 
	/**
	 * Get the number of rows in the chart data.
	 *
	 * @return
	 */
	public int getRows() { return rows; }

	/**
     * Get the number of columns in the chart data.
     *
     * @return
     */
	public int getColumns() { return columns; }

	/**
     * Get the row labels for the chart data.
     *
     * @return
    */
	public List<String> getRowLabels() { return rowLabels; }

	/**
      * Get the List of values for a particular row (series) in the chart data.
      * 
      * @param index
      * @return
    */
	public List<double[]> getRowData(int index) { return newData.get(index); }
}
