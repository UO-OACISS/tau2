package edu.uoregon.tau.perfexplorer.server;

import org.jfree.data.xy.AbstractXYDataset;
import org.jfree.data.xy.XYDataset;

import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;


/**
 * Dataset to store scatterplot data.
 * The JFreeChart API requires that client applications extend the 
 * AbstractXYDataset class to implement the data to be plotted in a scatterplot.
 * This is essentially a wrapper class around the RawDataInterface class.
 *
 * <P>CVS $Id: ScatterPlotDataset.java,v 1.7 2009/02/25 19:51:46 wspear Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class ScatterPlotDataset extends AbstractXYDataset implements XYDataset {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6304604972636260368L;
	private RawDataInterface data = null;
	private String seriesName = null;
	private int x = 0;
	private int y = 1;
	private boolean useMainValue = false;
	//private String debug = null;
	
	/**
	 * Constructor.
	 * 
	 * @param data
	 * @param seriesName
	 * @param x
	 * @param y
	 */
	public ScatterPlotDataset(RawDataInterface data, String seriesName,
	int x, int y, boolean useMainValue) {
		super();
		this.data = data;
		this.seriesName = seriesName;
		this.x = x;
		this.y = y;
		this.useMainValue = useMainValue;
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesCount()
	 */
	public int getSeriesCount() {
		return 1;
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesName(int)
	 */
	public String getSeriesKey(int arg0) {
		return seriesName;
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getItemCount(int)
	 */
	public int getItemCount(int arg0) {
		return data.numVectors();
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getX(int, int)
	 */
	public Number getX(int arg0, int arg1) {
		//if (x == 0) 
			//debug = new String ("Value [" + arg0 + "/" + arg1 + "]: " + data.getValue(arg1,x));
		return new Double(data.getValue(arg1, x));
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getY(int, int)
	 */
	public Number getY(int arg0, int arg1) {
		//if (y == 0 && debug != null) {
			//String debug2 = new String ("Value [" + arg0 + "/" + arg1 + "]: " + data.getValue(arg1,y));
			//PerfExplorerOutput.println(debug + ", " + debug2);
		//}
		if (useMainValue)
			return new Double(data.getMainValue(arg1));
		else
			return new Double(data.getValue(arg1, y));
	}

}
