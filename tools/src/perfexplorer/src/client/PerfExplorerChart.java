package client;

import common.RMIChartData;
import java.awt.*;
import java.util.List;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.DefaultTableXYDataset;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.data.xy.XYSeries;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.StandardLegend;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

public class PerfExplorerChart {

	public static void doFractionChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.FRACTION_OF_TOTAL);

        DefaultTableXYDataset dataset = new DefaultTableXYDataset();
		List rowLabels = rawData.getRowLabels();
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
			XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				s.add(values[0], values[1]);
			}
			dataset.addSeries(s);
		}
        JFreeChart chart = ChartFactory.createStackedXYAreaChart(
            "Total Runtime Breakdown for " +   // chart title
			PerfExplorerModel.getModel().toString() + ":" +
			PerfExplorerModel.getModel().getMetricName(),
            "Number of Processors",          // domain axis label
            "Percentage of Total Runtime",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		ChartFrame frame = new ChartFrame("Total Runtime Breakdown", chart);
		centerFrame(frame);

		frame.pack();
		frame.setVisible(true);
	}

	public static void doEfficiencyChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.RELATIVE_EFFICIENCY);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		double ideal, ratio = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			double[] baseline = (double[])(row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				ratio = baseline[0]/values[0];
				ideal = baseline[1] * ratio;
        		s.add(values[0], ideal/values[1]);
			}
        	dataset.addSeries(s);
		}

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Relative Efficiency - " +  // chart title
			PerfExplorerModel.getModel().toString() + ":" +
			PerfExplorerModel.getModel().getMetricName(),
            "Number of Processors",          // domain axis label
            "Value",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		customizeChart(chart, rawData.getRows(), false);
		ChartFrame frame = new ChartFrame("Relative Efficiency", chart);
		centerFrame(frame);
		frame.pack();
		frame.setVisible(true);
	}

	public static void doEfficiencyEventsChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.RELATIVE_EFFICIENCY_EVENTS);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		double ideal, ratio = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			double[] baseline = (double[])(row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				ratio = baseline[0]/values[0];
				ideal = baseline[1] * ratio;
        		s.add(values[0], ideal/values[1]);
			}
        	dataset.addSeries(s);
		}

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Relative Efficiency by Event for " +
			PerfExplorerModel.getModel().toString() + ":" +
			PerfExplorerModel.getModel().getMetricName(),  // chart title
            "Number of Processors",          // domain axis label
            "Value",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		customizeChart(chart, rawData.getRows(), false);
		ChartFrame frame = new ChartFrame("Relative Efficiency by Event", chart);
		centerFrame(frame);
		frame.pack();
		frame.setVisible(true);
	}

	public static void doEfficiencyOneEventChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.RELATIVE_EFFICIENCY_ONE_EVENT);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		double ideal, ratio = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			double[] baseline = (double[])(row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				ratio = baseline[0]/values[0];
				ideal = baseline[1] * ratio;
        		s.add(values[0], ideal/values[1]);
			}
        	dataset.addSeries(s);
		}

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Relative Efficiency for " +
			PerfExplorerModel.getModel().getEventName() + ":" +
			PerfExplorerModel.getModel().getMetricName(),  // chart title
            "Number of Processors",          // domain axis label
            "Value",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		customizeChart(chart, rawData.getRows(), false);
		ChartFrame frame = new ChartFrame("Relative Efficiency for Event", chart);
		centerFrame(frame);
		frame.pack();
		frame.setVisible(true);
	}

	public static void doSpeedupChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.RELATIVE_EFFICIENCY);

        XYSeriesCollection dataset = new XYSeriesCollection();

		List rowLabels = rawData.getRowLabels();
		double minx = 99999, maxx = 0;
		double ideal, ratio, efficiency = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			double[] baseline = (double[])(row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				ratio = baseline[0]/values[0];
				ideal = baseline[1] * ratio;
				efficiency = ideal/values[1];
				// System.out.println("adding: " + values[0] + ", " + efficiency/ratio);
        		s.add(values[0], efficiency / ratio);
				if (maxx < values[0])
					maxx = values[0];
			}
			if (minx > baseline[0])
				minx = baseline[0];
        	dataset.addSeries(s);
		}
        XYSeries s = new XYSeries("ideal", true, false);
        s.add(minx, 1);
        s.add(maxx, maxx/minx);
        dataset.addSeries(s);

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Relative Speedup - " +   // chart title
			PerfExplorerModel.getModel().toString() + ":" +
			PerfExplorerModel.getModel().getMetricName(),
            "Number of Processors",          // domain axis label
            "Value",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		customizeChart(chart, rawData.getRows(), true);
		ChartFrame frame = new ChartFrame("Relative Speedup", chart);
		centerFrame(frame);
		frame.pack();
		frame.setVisible(true);
	}

	public static void doSpeedupEventsChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.RELATIVE_EFFICIENCY_EVENTS);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		double minx = 99999, maxx = 0;
		double ideal = 0, ratio = 0, efficiency = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			double[] baseline = (double[])(row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				ratio = baseline[0]/values[0];
				ideal = baseline[1] * ratio;
				efficiency = ideal/values[1];
        		s.add(values[0], efficiency / ratio);
				if (maxx < values[0])
					maxx = values[0];
			}
			if (minx > baseline[0])
				minx = baseline[0];
        	dataset.addSeries(s);
		}
        XYSeries s = new XYSeries("ideal", true, false);
        s.add(minx, 1);
        s.add(maxx, maxx/minx);
        dataset.addSeries(s);

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Relative Speedup by Event for " + 
			PerfExplorerModel.getModel().toString() + ":" +
			PerfExplorerModel.getModel().getMetricName(),  // chart title
            "Number of Processors",          // domain axis label
            "Value",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		customizeChart(chart, rawData.getRows(), true);
		ChartFrame frame = new ChartFrame("Relative Speedup by Event", chart);
		centerFrame(frame);
		frame.pack();
		frame.setVisible(true);
	}

	public static void doSpeedupOneEventChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.RELATIVE_EFFICIENCY_ONE_EVENT);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		double minx = 99999, maxx = 0;
		double ideal = 0, ratio = 0, efficiency = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			double[] baseline = (double[])(row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				ratio = baseline[0]/values[0];
				ideal = baseline[1] * ratio;
				efficiency = ideal/values[1];
        		s.add(values[0], efficiency / ratio);
				if (maxx < values[0])
					maxx = values[0];
			}
			if (minx > baseline[0])
				minx = baseline[0];
        	dataset.addSeries(s);
		}
        XYSeries s = new XYSeries("ideal", true, false);
        s.add(minx, 1);
        s.add(maxx, maxx/minx);
        dataset.addSeries(s);

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Relative Speedup for " + 
			PerfExplorerModel.getModel().getEventName() + ":" +
			PerfExplorerModel.getModel().getMetricName(),  // chart title
            "Number of Processors",          // domain axis label
            "Value",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		customizeChart(chart, rawData.getRows(), true);
		ChartFrame frame = new ChartFrame("Relative Speedup for Event", chart);
		centerFrame(frame);
		frame.pack();
		frame.setVisible(true);
	}


	public static void doTimestepsChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.RELATIVE_EFFICIENCY);

		int timesteps = Integer.parseInt(PerfExplorerModel.getModel().getTotalTimesteps());
        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
        		s.add(values[0], timesteps/values[1]);
			}
        	dataset.addSeries(s);
		}

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Timesteps Per Second (" + timesteps + " total timesteps):" +
			PerfExplorerModel.getModel().getMetricName(),  // chart title
            "Number of Processors",          // domain axis label
            "Timesteps",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );

		customizeChart(chart, rawData.getRows(), false);
		ChartFrame frame = new ChartFrame("Timesteps per Second", chart);
		centerFrame(frame);
		frame.pack();
		frame.setVisible(true);
	}

	public static void doCommunicationChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData1 = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.TOTAL_FOR_GROUP);

		RMIChartData rawData2 = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.RELATIVE_EFFICIENCY);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData1.getRowLabels();
		for (int y = 0 ; y < rawData1.getRows() ; y++) {
			List row1 = rawData1.getRowData(y);
			List row2 = rawData2.getRowData(y);
        	XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			for (int x = 0 ; x < row1.size() ; x++) {
				double[] values1 = (double[])(row1.get(x));
				double[] values2 = (double[])(row2.get(x));
        		s.add(values1[0], values1[1]/values2[1]);
			}
        	dataset.addSeries(s);
		}

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Communication Time / Total Runtime" +   // chart title
			PerfExplorerModel.getModel().toString() + ":" +
			PerfExplorerModel.getModel().getMetricName(),
            "Number of Processors",          // domain axis label
            "Fraction",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );

		customizeChart(chart, rawData1.getRows(), false);
		ChartFrame frame = new ChartFrame("Transpose Time / Total Runtime", chart);
		centerFrame(frame);
		frame.pack();
		frame.setVisible(true);
	}

	public static void doEfficiencyPhasesChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.RELATIVE_EFFICIENCY_PHASES);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		double ideal, ratio = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			double[] baseline = (double[])(row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				ratio = baseline[0]/values[0];
				ideal = baseline[1] * ratio;
        		s.add(values[0], ideal/values[1]);
			}
        	dataset.addSeries(s);
		}

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Relative Efficiency by Phase for " +
			PerfExplorerModel.getModel().toString() + ":" +
			PerfExplorerModel.getModel().getMetricName(),  // chart title
            "Number of Processors",          // domain axis label
            "Value",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		customizeChart(chart, rawData.getRows(), false);
		ChartFrame frame = new ChartFrame("Relative Efficiency by Event", chart);
		centerFrame(frame);
		frame.pack();
		frame.setVisible(true);
	}

	public static void doSpeedupPhasesChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.RELATIVE_EFFICIENCY_PHASES);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		double minx = 99999, maxx = 0;
		double ideal, ratio, efficiency = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			double[] baseline = (double[])(row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				ratio = baseline[0]/values[0];
				ideal = baseline[1] * ratio;
				efficiency = ideal/values[1];
				// System.out.println("adding: " + values[0] + ", " + efficiency/ratio);
        		s.add(values[0], efficiency / ratio);
				if (maxx < values[0])
					maxx = values[0];
			}
			if (minx > baseline[0])
				minx = baseline[0];
        	dataset.addSeries(s);
		}
        XYSeries s = new XYSeries("ideal", true, false);
        s.add(minx, 1);
        s.add(maxx, maxx/minx);
        dataset.addSeries(s);

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Relative Speedup by Phase for " +
			PerfExplorerModel.getModel().toString() + ":" +
			PerfExplorerModel.getModel().getMetricName(),  // chart title
            "Number of Processors",          // domain axis label
            "Value",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		customizeChart(chart, rawData.getRows(), true);
		ChartFrame frame = new ChartFrame("Relative Speedup by Event", chart);
		centerFrame(frame);
		frame.pack();
		frame.setVisible(true);
	}

	public static void doFractionPhasesChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			RMIChartData.FRACTION_OF_TOTAL_PHASES);

        DefaultTableXYDataset dataset = new DefaultTableXYDataset();
		List rowLabels = rawData.getRowLabels();
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
			XYSeries s = new XYSeries((String)rowLabels.get(y), true, false);
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				s.add(values[0], values[1]);
			}
			dataset.addSeries(s);
		}

		JFreeChart chart = ChartFactory.createStackedXYAreaChart(
            "Total Runtime Breakdown for " + 
			PerfExplorerModel.getModel().toString() + ":" +
			PerfExplorerModel.getModel().getMetricName(),  // chart title
            "Number of Processors",          // domain axis label
            "Percentage of Total Runtime",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		ChartFrame frame = new ChartFrame("Total Runtime Breakdown", chart);
		centerFrame(frame);
		frame.pack();
		frame.setVisible(true);
	}


	private static void customizeChart (JFreeChart chart, int rows, boolean lastLineIdeal) {
		// customize the chart!
        StandardLegend legend = (StandardLegend) chart.getLegend();
        legend.setDisplaySeriesShapes(true);
        
        // get a reference to the plot for further customisation...
        XYPlot plot = chart.getXYPlot();
     
        //StandardXYItemRenderer renderer = (StandardXYItemRenderer) plot.getRenderer();
		XYLineAndShapeRenderer renderer = null;
		if (lastLineIdeal)
			renderer = new SpeedupXYLineAndShapeRenderer(rows);
		else
			renderer = new XYLineAndShapeRenderer();
        renderer.setDefaultShapesFilled(true);
        //renderer.setPlotShapes(true);
        renderer.setItemLabelsVisible(true);

		for (int i = 0 ; i < rows ; i++) {
			renderer.setSeriesStroke(i, new BasicStroke(2.0f));
		}

		if (lastLineIdeal) {
			renderer.setSeriesShapesVisible(rows, false);
		}
        // change the auto tick unit selection to integer units only...
        //NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        //rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

		// if the last line is the "ideal" line, make it black, with no points.
		plot.setRenderer(renderer);

	}

	private static void centerFrame(ChartFrame frame) {
        //Window Stuff.
        int windowWidth = 700;
        int windowHeight = 450;
        
        //Grab paraProfManager position and size.
        Point parentPosition = PerfExplorerClient.getMainFrame().getLocationOnScreen();
        Dimension parentSize = PerfExplorerClient.getMainFrame().getSize();
        int parentWidth = parentSize.width;
        int parentHeight = parentSize.height;
        
        //Set the window to come up in the center of the screen.
        int xPosition = (parentWidth - windowWidth) / 2;
        int yPosition = (parentHeight - windowHeight) / 2;

        xPosition = (int) parentPosition.getX() + xPosition;
        yPosition = (int) parentPosition.getY() + yPosition;

        frame.setLocation(xPosition, yPosition);
        frame.setSize(new java.awt.Dimension(windowWidth, windowHeight));
 	}

}
