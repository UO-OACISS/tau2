package edu.uoregon.tau.perfexplorer.server;


import org.jfree.data.xy.AbstractXYDataset;

import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;

/**
 * Dataset to store scatterplot data.
 * The JFreeChart API requires that client applications extend the 
 * AbstractXYDataset class to implement the data to be plotted in a scatterplot.
 * This is essentially a wrapper around the RawDataInterface class.
 * 
 * <P>CVS $Id: PCAPlotDataset.java,v 1.8 2009/02/25 19:51:46 wspear Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class PCAPlotDataset extends AbstractXYDataset {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6708684835661097166L;
	private RawDataInterface[] clusters = null;

	/**
	 * Constructor.
	 * 
	 */ 
	public PCAPlotDataset(RawDataInterface[] clusters) {
		this.clusters = clusters;
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesCount()
	 */
	public int getSeriesCount() {
		if (clusters == null)
			System.exit(1);
		return java.lang.reflect.Array.getLength(clusters);
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesName(int)
	 */
	public String getSeriesKey(int arg0) {
		return new String("Cluster " + arg0);
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getItemCount(int)
	 */
	public int getItemCount(int arg0) {
		return clusters[arg0].numVectors();
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getX(int, int)
	 */
	public Number getX(int arg0, int arg1) {
		//PerfExplorerOutput.print("point[" + arg0 + "/" + arg1 + "]: (" + clusters[arg0].getValue(arg1,0));
		return new Double(clusters[arg0].getValue(arg1,0));
		//return new Double(data.getValue(arg1, x));
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getY(int, int)
	 */
	public Number getY(int arg0, int arg1) {
		//PerfExplorerOutput.println("," + clusters[arg0].getValue(arg1,1) + ")");
		return new Double(clusters[arg0].getValue(arg1,1));
		//return new Double(clusters[arg0].instance(arg1).value(y));
		//return new Double(data.getValue(arg1, y));
	}
}
