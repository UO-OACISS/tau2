package client;

import org.jfree.data.xy.AbstractXYDataset;
import org.jfree.data.xy.XYDataset;
import common.RMIChartData;
import clustering.RawDataInterface;
import java.util.List;

/**
 * Dataset to store scatterplot data.
 * The JFreeChart API requires that client applications extend the 
 * AbstractXYDataset class to implement the data to be plotted in a scatterplot.
 * This is essentially a wrapper class around the RawDataInterface class.
 *
 * <P>CVS $Id: CorrelationPlotDataset.java,v 1.1 2005/10/21 19:42:58 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class CorrelationPlotDataset extends AbstractXYDataset implements XYDataset {

	private RMIChartData data = null;
	private List seriesNames = null;
	private int x = 0;
	private int y = 1;
	private boolean useMainValue = false;
	
	/**
	 * Constructor.
	 * 
	 * @param data
	 * @param seriesNames
	 */
	public CorrelationPlotDataset(RMIChartData data) {
		super();
		this.data = data;
		this.seriesNames = data.getRowLabels();
		this.x = x;
		this.y = y;
		this.useMainValue = useMainValue;
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesCount()
	 */
	public int getSeriesCount() {
		// we have n rows, but the first row is the data we are
		// correlating against.
		//return data.getRows() - 1;
		return 1;
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesName(int)
	 */
	public String getSeriesName(int arg0) {
		return (String)(seriesNames.get(arg0+1));
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getItemCount(int)
	 */
	public int getItemCount(int arg0) {
		System.out.println("Item " + arg0 + " Count: " + data.getRowData(arg0+1).size());
		return data.getRowData(arg0+1).size();
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getX(int, int)
	 */
	public Number getX(int arg0, int arg1) {
		// get the n+1 row
		List row = data.getRowData(0);
		// get the mth column from that row
		double[] values = (double[])row.get(arg1);
		System.out.println("x values (" + arg0 + "," + arg1 + "): " + values[0] + ", " + values[1]);
		return new Double(values[y]);
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getY(int, int)
	 */
	public Number getY(int arg0, int arg1) {
		// get the data for the main function
		List row = data.getRowData(arg0+1);
		// get the mth column from that row
		double[] values = (double[])row.get(arg1);
		System.out.println("y values (" + arg0 + "," + arg1 + "): " + values[0] + ", " + values[1]);
		return new Double(values[y]);
	}

}
