package edu.uoregon.tau.perfexplorer.client;

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
import org.jfree.data.xy.XYDataset;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import java.text.DecimalFormat;
import java.text.FieldPosition;
import java.net.URL;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.common.*;

import java.awt.Toolkit;

public class PerfExplorerProbabilityPlot extends PerfExplorerChartWindow {

	public PerfExplorerProbabilityPlot(JFreeChart chart, String name) {
		super(chart, name);
	}

	public static JFrame doProbabilityPlot() {
		// for each event, get the variation across all threads.
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		// get the data
		PerfExplorerModel model = PerfExplorerModel.getModel();
        Object selection = model.getCurrentSelection();
		RMIChartData data = null;
        if (selection instanceof RMISortableIntervalEvent) {
			data = server.requestChartData(model, ChartDataType.DISTRIBUTION_DATA);
		} else {
			data = server.requestChartData(model, ChartDataType.IQR_DATA);
		}

		ProbabilityPlotDataset dataset = new ProbabilityPlotDataset(data);

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
        renderer.setToolTipGenerator(new StandardXYToolTipGenerator() {
            public String generateToolTip(XYDataset inDataset, int arg1, int arg2) {
				ProbabilityPlotDataset dataset = (ProbabilityPlotDataset) inDataset;
				return dataset.getTooltip(arg1, arg2);
            }
        });
        plot.setRenderer(renderer);

        // change the auto tick unit selection to integer units only...
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

		JFrame window = new PerfExplorerProbabilityPlot(chart, "Normal Probability Plot");
        URL url = Utility.getResource("tau32x32.gif");
		if (url != null) 
			window.setIconImage(Toolkit.getDefaultToolkit().getImage(url));
		return window;
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
