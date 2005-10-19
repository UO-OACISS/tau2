package client;

import common.RMIChartData;
import java.awt.*;
import java.awt.event.*;
import javax.swing.JFrame;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import java.util.List;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.DefaultTableXYDataset;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.data.xy.XYSeries;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.StandardLegend;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import java.text.DecimalFormat;
import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.common.VectorExport;

public class PerfExplorerChart extends PerfExplorerChartWindow {

	public PerfExplorerChart (JFreeChart chart, String name) {
		super(chart, name);
	}

	public static PerfExplorerChart doFractionChart () {
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

		return new PerfExplorerChart(chart, "Total Runtime Breakdown");
	}

	public static PerfExplorerChart doEfficiencyChart () {
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
		return new PerfExplorerChart(chart, "Relative Efficiency");
	}

	public static PerfExplorerChart doEfficiencyEventsChart () {
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
		return new PerfExplorerChart(chart, "Relative Efficiency by Event");
	}

	public static PerfExplorerChart doEfficiencyOneEventChart () {
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
		return new PerfExplorerChart(chart, "Relative Efficiency for Event");
	}

	public static PerfExplorerChart doSpeedupChart () {
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
		return new PerfExplorerChart(chart, "Relative Speedup");
	}

	public static PerfExplorerChart doSpeedupEventsChart () {
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
		return new PerfExplorerChart(chart, "Relative Speedup by Event");
	}

	public static PerfExplorerChart doSpeedupOneEventChart () {
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
		return new PerfExplorerChart(chart, "Relative Speedup for Event");
	}


	public static PerfExplorerChart doTimestepsChart () {
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
		return new PerfExplorerChart(chart, "Timesteps per Second");
	}

	public static PerfExplorerChart doCommunicationChart () {
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
		return new PerfExplorerChart(chart, "Transpose Time / Total Runtime");
	}

	public static PerfExplorerChart doEfficiencyPhasesChart () {
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
		return new PerfExplorerChart(chart, "Relative Efficiency by Event");
	}

	public static PerfExplorerChart doSpeedupPhasesChart () {
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
		return new PerfExplorerChart(chart, "Relative Speedup by Event");
	}

	public static PerfExplorerChart doFractionPhasesChart () {
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
		return new PerfExplorerChart(chart, "Total Runtime Breakdown");
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
		renderer.setToolTipGenerator(
			new StandardXYToolTipGenerator(
				StandardXYToolTipGenerator.DEFAULT_TOOL_TIP_FORMAT,
					new DecimalFormat("processors: #######"), new DecimalFormat("value: #,##0.00")));

        //renderer.setToolTipGenerator(new XYToolTipGenerator() {
            //public String generateToolTip(XYDataset dataset, int arg1, int arg2) {
                //return "<html>Number of threads: " + (int)dataset.getXValue(arg1, arg2) + 
                //"<BR>Value: " + dataset.getYValue(arg1, arg2) + "</html>";
            //}
        //});


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

}
