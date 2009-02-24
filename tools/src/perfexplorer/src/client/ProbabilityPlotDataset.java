package edu.uoregon.tau.perfexplorer.client;

import org.jfree.data.xy.AbstractXYDataset;
import org.jfree.data.xy.XYDataset;

import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.common.RMIChartData;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.text.DecimalFormat;
import java.text.FieldPosition;

/**
 * Dataset to store scatterplot data.
 * The JFreeChart API requires that client applications extend the 
 * AbstractXYDataset class to implement the data to be plotted in a scatterplot.
 * This is essentially a wrapper class around the RawDataInterface class.
 *
 * <P>CVS $Id: ProbabilityPlotDataset.java,v 1.3 2009/02/24 00:53:33 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class ProbabilityPlotDataset extends AbstractXYDataset implements XYDataset {

	private List seriesNames = null;
	private List dataset = null;  // List of List of Points
	private List metrics = null;
	private List normalizedMetrics = null;

	/**
	 * Constructor.
	 * 
	 * @param data
	 * @param seriesNames
	 */
	public ProbabilityPlotDataset(RMIChartData data) {
		super();
		this.seriesNames = data.getRowLabels();
		this.dataset = new ArrayList();
		this.metrics = new ArrayList();
		this.normalizedMetrics = new ArrayList();

		for (int y = 0 ; y < data.getRows() ; y++) {
		//for (int y = 2 ; y < 3 ; y++) {
			List row = data.getRowData(y);

			// put the values in a sortable list, and capture the min and max
			List points = new ArrayList();
			double max = 0.0, min = 0.0;
			double normalizedMax = 0.0, normalizedMin = 0.0;

			// initialize min and max
			double[] tmp = (double[])(row.get(0));
			min = tmp[1];
			max = tmp[1];

			// initialize avg and stdev
			double avg = 0.0;
			double normalizedAvg = 0.0;
			double stDev = 0.0;
			double normalizedStDev = 0.0;

			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				Point p = new Point(values[0], values[1]);
				points.add(p);
				// update min and max
				if (min > values[1])
					min = values[1];
				if (max < values[1])
					max = values[1];
				//System.out.print(values[1] + ", ");
			}
			//System.out.println("");
			//System.out.println("min = " + min);
			//System.out.println("max = " + max);

			double range = max - min;
			// normalize data from 0.0 to 1.0
			for (int x = 0 ; x < points.size() ; x++) {
				Point p = (Point)points.get(x);
				p.n = (p.y - min)/range;
				// update avg
				avg += p.y;
				normalizedAvg += p.n;
				//System.out.print(p.n + ", ");
			}
			//System.out.println("");
			normalizedMax = 1.0;
			normalizedMin = 0.0;

			// get the average
			avg = avg / points.size();
			normalizedAvg = normalizedAvg / points.size();

			//System.out.println("avg = " + avg);
			//System.out.println("normalizedAvg = " + normalizedAvg);

			// calculate the residuals and the standard deviations
			for (int x = 0 ; x < points.size() ; x++) {
				Point p = (Point)points.get(x);
				p.rn = p.n - normalizedAvg;
				p.r = p.y - avg;
				//normalizedStDev += p.rn * p.rn;
				normalizedStDev += p.r * p.r;
				stDev += p.r * p.r;
				//System.out.print(p.r + ", ");
			}
			//System.out.println("");
			stDev = stDev / (points.size() -1);
			stDev = java.lang.Math.sqrt(stDev);
			normalizedStDev = normalizedStDev / (points.size() -1);
			normalizedStDev = java.lang.Math.sqrt(normalizedStDev);

			//System.out.println("stDev = " + stDev);
			//System.out.println("normalizedStDev = " + normalizedStDev);

			// convert values to z-score
			for (int x = 0 ; x < points.size() ; x++) {
				Point p = (Point)points.get(x);
				p.z = (p.r)/stDev;
			}

			Metric metric = new Metric(min, max, avg, stDev);
			metrics.add(metric);
			Metric normalizedMetric = new Metric(0.0, 1.0, normalizedAvg, 
				normalizedStDev);
			normalizedMetrics.add(normalizedMetric);

			// rank the data from smallest to largest
			Collections.sort(points);
			double ppp = 0;
			for (int x = 0 ; x < points.size() ; x++) {
				// calculate probability plot position, F_i
				ppp = (x+0.5)/points.size();
				Point p = (Point)points.get(x);
				p.p = ppp;
				p.y = StatUtil.getInvCDF(ppp, false);
			}
			dataset.add(points);
		}
		List points = new ArrayList();
		Point p = new Point(-3, -3);
		points.add(p);
		Point p2 = new Point(3, 3);
		points.add(p2);
        dataset.add(points);
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesCount()
	 */
	public int getSeriesCount() {
		return dataset.size();
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesName(int)
	 */
	public String getSeriesName(int arg0) {
		if (arg0 == dataset.size() -1)
			return new String("Ideal normal");
		else {
			String tmp = (String)(seriesNames.get(arg0));
			return tmp;
		}
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getItemCount(int)
	 */
	public int getItemCount(int arg0) {
		List series = (List)dataset.get(arg0);
		return series.size();
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getX(int, int)
	 */
	public Number getX(int arg0, int arg1) {
		// get the row
		List row = (List)dataset.get(arg0);
		// get the mth column from that row
		Point p = (Point)row.get(arg1);
		return new Double(p.y);
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getX(int, int)
	 */
	public String getTooltip(int arg0, int arg1) {
		// get the row
		List row = (List)dataset.get(arg0);
		// get the mth column from that row
		Point p = (Point)row.get(arg1);
		DecimalFormat format = new DecimalFormat("0.00");
		FieldPosition f = new FieldPosition(0);
		StringBuffer buf = new StringBuffer();
        StringBuffer s = new StringBuffer();
		s.append("<html>process rank: ");
		format.format(p.x, buf, f);
		s.append(buf.toString()); 
		buf = new StringBuffer();
   		s.append("<BR>measured value: ");
		format.format(p.m, buf, f);
		s.append(buf.toString()); 
		buf = new StringBuffer();
   		s.append("<BR>expected percentile: ");
		format.format(p.p, buf, f);
		s.append(buf.toString()); 
		buf = new StringBuffer();
   		s.append("<BR>expected z-score: ");
		format.format(p.y, buf, f);
		s.append(buf.toString()); 
		buf = new StringBuffer();
   		s.append("<BR>z-score: ");
		format.format(p.z, buf, f);
		s.append(buf.toString()); 
		buf = new StringBuffer();
   		s.append("<BR>normalized value: ");
		format.format(p.n, buf, f);
		s.append(buf.toString()); 
		buf = new StringBuffer();
   		s.append("<BR>residual: ");
		format.format(p.r, buf, f);
		s.append(buf.toString()); 
		buf = new StringBuffer();
   		s.append("<BR>normalized residual: ");
		format.format(p.rn, buf, f);
		s.append(buf.toString()); 
		if (arg0 < (dataset.size() - 1)) {
			Metric m = (Metric)metrics.get(arg0);
			buf = new StringBuffer();
   			s.append("<BR>min: ");
			format.format(m.min, buf, f);
			s.append(buf.toString()); 
			buf = new StringBuffer();
   			s.append("<BR>max: ");
			format.format(m.max, buf, f);
			s.append(buf.toString()); 
			buf = new StringBuffer();
   			s.append("<BR>avg: ");
			format.format(m.avg, buf, f);
			s.append(buf.toString()); 
			buf = new StringBuffer();
   			s.append("<BR>stDev: ");
			format.format(m.stDev, buf, f);
			s.append(buf.toString()); 
		}
		s.append("</html>");
		return s.toString();
}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getY(int, int)
	 */
	public Number getY(int arg0, int arg1) {
		// get the row
		List row = (List)dataset.get(arg0);
		// get the mth column from that row
		Point p = (Point)row.get(arg1);
		return new Double(p.z);
	}

	public String getCorrelation (int series) {
		double r = 0.0;
		List row = (List)dataset.get(series);

		// solve for r
		double tmp1 = 0.0;
		double tmp2 = 0.0;
		for (int i = 0 ; i < row.size() ; i++ ) {
			Point p = (Point)row.get(i);
			r += p.z * p.y;
		}
		r = r / (row.size() - 1);

		DecimalFormat format = new DecimalFormat("0.00");
		FieldPosition f = new FieldPosition(0);
		StringBuffer s = new StringBuffer();
		format.format(new Double(r), s, f);
		return s.toString();
	}

	static class Point implements Comparable {
		public double x = 0.0; // original values
		public double m = 0.0; // expected ordered statistic medians
		public double y = 0.0; // expected ordered statistic medians
		public double p = 0.0; // expected percentile
		public double n = 0.0; // normalized x
		public double r = 0.0; // residual (x-avg)
		public double rn = 0.0; // normalized residual (x-avg)
		public double z = 0.0; // normalized y, converted to z-score
		public Point(double x, double y) {
			this.x = x;
			this.z = x;  // just for initialization for "Ideal" line
			this.y = y;
			this.m = y;
		}
		public int compareTo(Object o) {
			Point p = (Point)o;
			if (this.y < p.y) {
				return -1;
			} else if (this.y > p.y) {
				return 1;
			}else
				return 0;
		}
	}

	static class Metric {
		private double min = 0.0;
		private double max = 0.0;
		private double avg = 0.0;
		private double stDev = 0.0;
		public Metric(double min, double max, double avg, double stDev) {
			this.min = min;
			this.max = max;
			this.avg = avg;
			this.stDev = stDev;
		}
	}
}
