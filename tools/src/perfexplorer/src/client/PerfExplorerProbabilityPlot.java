package client;

import common.*;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import javax.swing.JFrame;
import java.lang.Math;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class PerfExplorerProbabilityPlot extends PerfExplorerChartWindow {

	public PerfExplorerProbabilityPlot(JFreeChart chart, String name) {
		super(chart, name);
	}

	public static JFrame doProbabilityPlot() {
		// for each event, get the variation across all threads.
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(PerfExplorerModel.getModel(), RMIChartData.IQR_DATA);

		XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		for (int y = 0 ; y < rawData.getRows() ; y++) {
		//for (int y = 0 ; y < 1 ; y++) {
			List row = rawData.getRowData(y);
			XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);

			// put the values in a sortable list, and capture the min and max
			List points = new ArrayList();
			double max = 0.0, min = 0.0;

			// initialize min and max
			double[] tmp = (double[])(row.get(0));
			min = tmp[1];
			max = tmp[1];

			// initialize avg and stdev
			double avg = 0.0;
			double stDev = 0.0;

			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				Point p = new Point(values[0], values[1]);
				points.add(p);
				// update min and max
				if (min > values[1])
					min = values[1];
				if (max < values[1])
					max = values[1];
			}

			double range = max - min;
			// normalize data from 0.0 to 1.0
			for (int x = 0 ; x < points.size() ; x++) {
				Point p = (Point)points.get(x);
				p.n = (p.y - min)/range;
				// update avg
				avg += p.n;
			}

			// get the average
			avg = avg / points.size();

			// calculate the standard deviation
			for (int x = 0 ; x < points.size() ; x++) {
				Point p = (Point)points.get(x);
				p.r = p.n - avg;
				stDev += p.r * p.r;
			}
			stDev = stDev / (points.size() -1);

			// convert values to z-score
			for (int x = 0 ; x < points.size() ; x++) {
				Point p = (Point)points.get(x);
				p.z = (p.n - avg)/stDev;
			}

			// get the average and standard deviation values
			// convert the values to z-scores

			// rank the data from smallest to largest
			Collections.sort(points);
			double ppp = 0;
			for (int x = 0 ; x < row.size() ; x++) {
				// calculate probability plot position, F_i
				ppp = (x+0.5)/row.size();
				Point p = (Point)points.get(x);
				s.add(StatUtil.getInvCDF(ppp, false), p.z);
			}
			dataset.addSeries(s);
		}

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Normal Probability Plot", 
            "Normal N(0,1) Ordered Statistic Medians", 
            "Ordered Measurements", 
            dataset, 
            PlotOrientation.VERTICAL, 
            true, 
            true, 
            false
        );
		
		XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		for (int y = 0 ; y < rawData.getRows() ; y++) {
        	renderer.setSeriesLinesVisible(y, false);
		}
        plot.setRenderer(renderer);

        // change the auto tick unit selection to integer units only...
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

		return new PerfExplorerProbabilityPlot(chart, "TEST");
    }

	private static double SND(double r) {
		// standard normal distribution
		double firstTerm = java.lang.Math.sqrt(2 * java.lang.Math.PI);
		double secondTerm = 0.0 - ((r*r)/2);
		double y = java.lang.Math.pow(java.lang.Math.E, secondTerm) / firstTerm;
		return y;
	}

	static class Point implements Comparable {
		public double x = 0.0;
		public double y = 0.0;
		public double n = 0.0; // normalized y
		public double r = 0.0; // residual (x-avg)
		public double z = 0.0; // normalized y, converted to z-score
		public Point(double x, double y) {
			this.x = x;
			this.y = y;
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
}
