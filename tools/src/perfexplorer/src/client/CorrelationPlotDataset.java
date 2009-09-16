package edu.uoregon.tau.perfexplorer.client;

import java.text.DecimalFormat;
import java.text.FieldPosition;
import java.util.List;

import org.jfree.data.xy.AbstractXYDataset;
import org.jfree.data.xy.XYDataset;

import edu.uoregon.tau.perfexplorer.common.RMIChartData;

/**
 * Dataset to store scatterplot data.
 * The JFreeChart API requires that client applications extend the 
 * AbstractXYDataset class to implement the data to be plotted in a scatterplot.
 * This is essentially a wrapper class around the RawDataInterface class.
 *
 * <P>CVS $Id: CorrelationPlotDataset.java,v 1.16 2009/09/16 09:28:10 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class CorrelationPlotDataset extends AbstractXYDataset implements XYDataset {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2371762530237884209L;
	private RMIChartData data = null;
	private List<String> seriesNames = null;
	private int x = 0;
	private int y = 1;
	private boolean main = false;
	private int maxLabels = 0;
	
	/**
	 * Constructor.
	 * 
	 * @param data
	 * @param seriesNames
	 */
	public CorrelationPlotDataset(RMIChartData data, boolean main) {
		super();
		this.data = data;
		this.seriesNames = data.getRowLabels();
		this.main = main;
		
		/*
		this.constantProblem = PerfExplorerModel.getModel().getConstantProblem().booleanValue();
		if (constantProblem) {
			for (int i = 0 ; i < data.getRows() ; i++) {
				List row = data.getRowData(i);
				double[] tmp0 = (double[])row.get(row.size()-1);
				for (int j = 0 ; j < row.size() ; j++ ) {
					double[] tmp = (double[])row.get(j);
					//tmp[y] = tmp[y]*(tmp[x]/tmp0[x]);
					tmp[y] = java.lang.Math.log(tmp[y]*10)/java.lang.Math.log(2);
					tmp[x] = java.lang.Math.log(tmp[x])/java.lang.Math.log(2);
				}
			}
		}
		*/
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesCount()
	 */
	public int getSeriesCount() {
		// we have n rows, but the first row is the data we are
		// correlating against.
		if (main)
			return 1;
		else
			return data.getRows() - 1;
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesName(int)
	 */
	public Comparable<String> getSeriesKey(int arg0) {//TODO: Was getSeriesName
		if (main) {
			return PerfExplorerChart.shortName((seriesNames.get(arg0)));
		} else {
			String tmp = PerfExplorerChart.shortName((seriesNames.get(arg0+1)));
			return tmp + ", r = " + getCorrelation(0, arg0+1);
		}
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getItemCount(int)
	 */
	public int getItemCount(int arg0) {
		return data.getRowData(arg0).size();
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getX(int, int)
	 */
	public Number getX(int arg0, int arg1) {
		//PerfExplorerOutput.println("getX(" + arg0 + ", " + arg1 + ")");
		if (!main)
			arg0++;
		// get the row
		List<double[]> row = data.getRowData(arg0);
		// get the mth column from that row
		System.out.println(arg0);
		try {
			double[] values = row.get(arg1);
			this.maxLabels = values.length;
			//return new Double(java.lang.Math.log(values[x])/java.lang.Math.log(2));
			return new Double(values[x]);
		} catch (Exception e) {
			// we went past the end of the array.  Bummer.  Something is bogus about
			// the data.  Therefore, create some dummy data so we don't crap out.
			return new Double(0.0);			
		}
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getY(int, int)
	 */
	public Number getY(int arg0, int arg1) {
		//PerfExplorerOutput.println("getY(" + arg0 + ", " + arg1 + ")");
		if (!main)
			arg0++;
		// get the row
		List<double[]> row = data.getRowData(arg0);
		// get the mth column from that row
		try {
			double[] values = row.get(arg1);
			//if (constantProblem) {
				//double[] values2 = (double[])row.get(0);
				//return new Double(values[y]*(values[x]/values2[x]));
			//}else {
				//return new Double(java.lang.Math.log(values[y])/java.lang.Math.log(2));
				return new Double(values[y]);
			//}
		} catch (Exception e) {
			// we went past the end of the array.  Bummer.  Something is bogus about
			// the data.  Therefore, create some dummy data so we don't crap out.
			return new Double(0.0);			
		}
	}

	public String getCorrelation (int x, int y) {
		double r = 0.0;
		double xAvg = 0.0;
		double yAvg = 0.0;
		double xStDev = 0.0;
		double yStDev = 0.0;
		//double sum = 0.0;//TODO: Use this
		List<double[]> xRow = data.getRowData(x);
		List<double[]> yRow = data.getRowData(y);

		for (int i = 0 ; (i < xRow.size() && i < yRow.size()) ; i++ ) {
			double[] tmp = xRow.get(i);
			xAvg += tmp[1];
			tmp = yRow.get(i);
			yAvg += tmp[1];
		}

		// find the average for the first vector
		xAvg = xAvg / xRow.size();
		// find the average for the second vector
		yAvg = yAvg / yRow.size();


		for (int i = 0 ; (i < xRow.size() && i < yRow.size()) ; i++ ) {
			double[] tmp = xRow.get(i);
			xStDev += (tmp[1] - xAvg) * (tmp[1] - xAvg);
			tmp = yRow.get(i);
			yStDev += (tmp[1] - yAvg) * (tmp[1] - yAvg);
		}

		// find the standard deviation for the first vector
		xStDev = xStDev / (xRow.size() - 1);
		xStDev = Math.sqrt(xStDev);
		// find the standard deviation for the second vector
		yStDev = yStDev / (yRow.size() - 1);
		yStDev = Math.sqrt(yStDev);


		// solve for r
		double tmp1 = 0.0;
		double tmp2 = 0.0;
		for (int i = 0 ; (i < xRow.size() && i < yRow.size()) ; i++ ) {
			double[] tmp = xRow.get(i);
			tmp1 = (tmp[1] - xAvg) / xStDev;
			tmp = yRow.get(i);
			tmp2 = (tmp[1] - yAvg) / yStDev;
			r += tmp1 * tmp2;
		}
		r = r / (xRow.size() - 1);

		DecimalFormat format = new DecimalFormat("0.00");
		FieldPosition f = new FieldPosition(0);
		StringBuffer s = new StringBuffer();
		format.format(new Double(r), s, f);
		return s.toString();
	}

}
