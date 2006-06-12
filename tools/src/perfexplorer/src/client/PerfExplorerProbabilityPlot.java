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

		ProbabilityPlotDataset dataset = new ProbabilityPlotDataset(rawData);

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
        SpeedupXYLineAndShapeRenderer renderer = 
			new SpeedupXYLineAndShapeRenderer(dataset.getSeriesCount()-1);
		for (int y = 0 ; y < dataset.getSeriesCount() ; y++) {
			if (y == dataset.getSeriesCount() - 1)
				renderer.setSeriesShapesVisible(y, false);
			else
        		renderer.setSeriesLinesVisible(y, false);
		}
        plot.setRenderer(renderer);

        // change the auto tick unit selection to integer units only...
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

		return new PerfExplorerProbabilityPlot(chart, "Normal Probability Plot");
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
