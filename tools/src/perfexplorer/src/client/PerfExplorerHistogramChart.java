/* -------------------
 * HistogramDemo1.java
 * -------------------
 * (C) Copyright 2004, by Object Refinery Limited.
 *
 */

package edu.uoregon.tau.perfexplorer.client;

import java.io.IOException;
import java.util.List;
import javax.swing.JFrame;
import java.lang.Math;
import org.jfree.data.Range;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.xy.IntervalXYDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.labels.XYToolTipGenerator;
import org.jfree.data.xy.XYDataset;
import java.net.URL;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.common.*;

import java.awt.Toolkit;

/**
 * A demo of the {@link HistogramDataset} class.
 */
public class PerfExplorerHistogramChart extends PerfExplorerChartWindow {

	public PerfExplorerHistogramChart(JFreeChart chart, String name) {
		super(chart, name);
	}

	public static JFrame doHistogram() {
		// for each event, get the variation across all threads.
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		PerfExplorerModel model = PerfExplorerModel.getModel();
        Object selection = model.getCurrentSelection();
		RMIChartData data = null;
        if (selection instanceof RMISortableIntervalEvent) {
			data = server.requestChartData(model, ChartDataType.DISTRIBUTION_DATA);
		} else {
			data = server.requestChartData(model, ChartDataType.IQR_DATA);
		}

		// build the chart
        IntervalXYDataset dataset = createDataset(data);
        JFreeChart chart = createChart(dataset);
        
		ChartPanel panel = new ChartPanel(chart);
		panel.setDisplayToolTips(true);
		XYItemRenderer renderer = chart.getXYPlot().getRenderer();
        renderer.setToolTipGenerator(new XYToolTipGenerator() {
            public String generateToolTip(XYDataset dataset, int arg1, int arg2) {
                return "<html>Event: " + dataset.getSeriesName(arg1) + 
                "<BR>Count: " + dataset.getYValue(arg1, arg2) + "</html>";
            }
        });

		JFrame window = new PerfExplorerHistogramChart (chart, 
			"Distributions of Significant Events");
        URL url = Utility.getResource("tau32x32.gif");
		if (url != null) 
			window.setIconImage(Toolkit.getDefaultToolkit().getImage(url));
		return window;
	}

    private static IntervalXYDataset createDataset(RMIChartData inData) {
        HistogramDataset dataset = new HistogramDataset();
        List names = inData.getRowLabels();
        double[] values = null;
		for (int i = 0 ; i < inData.getRows(); i++) {
			List doubles = inData.getRowData(i);
			values = new double[doubles.size()];
    		double min = ((double[])(doubles.get(0)))[1];
     		double max = min;  
    		for (int j = 1; j < doubles.size(); j++) {
    			if (min > ((double[])(doubles.get(j)))[1])
    				min = ((double[])(doubles.get(j)))[1];
    			if (max < ((double[])(doubles.get(j)))[1])
    				max = ((double[])(doubles.get(j)))[1];
    		}
    		double range = max - min;
    		//System.out.println("Min: " + min + ", Max: " + max + ", Range: " + range);
    		for (int j = 0; j < doubles.size(); j++) {
    			values[j] = (((double[])(doubles.get(j)))[1]-min)/range;   
    		}
			int bins = 10;
			if (doubles.size() >= 2098)
				bins = 200;
				//bins = Math.max(200, doubles.size() / 100);
			else if (doubles.size() >= 256)
				bins = 50;
			else if (doubles.size() >= 16)
				bins = 20;
         	dataset.addSeries((String)names.get(i), values, bins);
		}
        return dataset;     
    }
    
    /**
     * Creates a sample {@link HistogramDataset}.
     * 
     * @return the dataset.
     */
    private static IntervalXYDataset createDataset() {
        HistogramDataset dataset = new HistogramDataset();
        double[] values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        dataset.addSeries("H1", values, 10, 0.0, 10.0);
        return dataset;     
    }

    /**
     * Creates a chart.
     * 
     * @param dataset  a dataset.
     * 
     * @return The chart.
     */
    private static JFreeChart createChart(IntervalXYDataset dataset) {
        JFreeChart chart = ChartFactory.createHistogram(
            "TAU/PerfExplorer: Significant (>2.0% of runtime) Event Histograms", 
            "Percentiles", 
            "Count", 
            dataset, 
            PlotOrientation.VERTICAL, 
            true, 
            false, 
            false
        );
		NumberAxis axis = new NumberAxis("Percentiles");
		axis.setRange(new Range(0,1.0));
        chart.getXYPlot().setDomainAxis(0, axis);
        chart.getXYPlot().setForegroundAlpha(0.75f);
        return chart;
    }
        
}
