package common;

import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;

/**
 * This class represents the data to be used to create a scalability or
 * runtime breakdown chart.  
 *
 * <P>CVS $Id: RMIChartData.java,v 1.3 2005/09/29 15:45:29 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class RMIChartData implements Serializable {
	/**
	 * These fields define what type of data this is,
	 * or what type of data is being requested.
	 */
	public static final int FRACTION_OF_TOTAL = 0;
	public static final int RELATIVE_EFFICIENCY = 1;
	public static final int TIMESTEPS_PER_SECOND = 2;
	public static final int TOTAL_FOR_GROUP = 3;
	public static final int RELATIVE_EFFICIENCY_EVENTS = 4;
	public static final int RELATIVE_EFFICIENCY_ONE_EVENT = 5;
	public static final int RELATIVE_EFFICIENCY_PHASES = 6;
	public static final int FRACTION_OF_TOTAL_PHASES = 7;
	public static final int IQR_DATA = 8;

	protected int dataType = FRACTION_OF_TOTAL;
	int rows = 0;
	int columns = 0;

	protected List newData = null;
	protected List rowLabels = null;

	/**
	 * Constructor.
	 * 
	 * @param dataType
	 */
	public RMIChartData (int dataType) {
		this.dataType = dataType;
		this.newData = new ArrayList();
		this.rowLabels = new ArrayList();
	}

	/**
	 * Add a row of data fields (a new line on the chart)
	 * 
	 * @param label
	 */
	public void addRow(String label) {
		newData.add(new ArrayList());
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
		List row = (List)newData.get(rowIndex);
		double[] values = new double[2];
		values[0] = v1;
		values[1] = v2;
		row.add(values);
		//System.out.println("Added: " + rowIndex + ", " + v1 + ", " + v2);
	}

	public int getRows() { return rows; }
	public int getColumns() { return columns; }
	public List getRowLabels() { return rowLabels; }
	public List getRowData(int index) { return (List)newData.get(index); }
}
