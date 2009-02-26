package edu.uoregon.tau.perfexplorer.client;

import java.awt.BasicStroke;
import java.awt.Color;
import java.text.DecimalFormat;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.DefaultTableXYDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import edu.uoregon.tau.perfexplorer.common.ChartDataType;
import edu.uoregon.tau.perfexplorer.common.RMIChartData;
import edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData;

public class PerfExplorerChart extends PerfExplorerChartWindow {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8313521188602064044L;

	public PerfExplorerChart (JFreeChart chart, String name) {
		super(chart, name);
	}

	public static PerfExplorerChart doFractionChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			ChartDataType.FRACTION_OF_TOTAL);

        DefaultTableXYDataset dataset = new DefaultTableXYDataset();
		List<String> rowLabels = rawData.getRowLabels();
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List<double[]> row = rawData.getRowData(y);
			XYSeries s = new XYSeries(shortName(rowLabels.get(y)), true, false);
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (row.get(x));
				s.add(values[0], values[1]);
			}
			dataset.addSeries(s);
		}
        JFreeChart chart = ChartFactory.createStackedXYAreaChart(
            "Total " + PerfExplorerModel.getModel().getMetricName() + " Breakdown for " +   // chart title
			PerfExplorerModel.getModel().toString(),
            "Number of Processors",          // domain axis label
            "Percentage of Total " + PerfExplorerModel.getModel().getMetricName(),     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
        //XYPlot plot = chart.getXYPlot();
		//NumberAxis axis = new NumberAxis("Percentage of Total " + PerfExplorerModel.getModel().getMetricName());
        //axis.setRange(new Range(0,100));
        //plot.setRangeAxis(0, axis);

		return new PerfExplorerChart(chart, "Total " + PerfExplorerModel.getModel().getMetricName() + " Breakdown");
	}

	public static PerfExplorerChart doCorrelationChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), ChartDataType.CORRELATION_DATA);

        XYDataset dataset = new CorrelationPlotDataset(rawData, false);
        //JFreeChart chart = ChartFactory.createScatterPlot(
        JFreeChart chart = ChartFactory.createXYLineChart(
            "Correlation Results: " +  // chart title
			PerfExplorerModel.getModel().toString() + ": " +
			PerfExplorerModel.getModel().getMetricName(),
            //"Inclusive Time for " + (String)rawData.getRowLabels().get(0), //domain axis
            "Processors", //domain axis
            "Exclusive " + PerfExplorerModel.getModel().getMetricName() + " for Event",  // range axis
            dataset,					// data
            PlotOrientation.VERTICAL,		// the orientation
            true,						// legend
            false,						// tooltips
            false						// urls
        );

		customizeChart(chart, rawData.getRows(), true);
        XYPlot plot = chart.getXYPlot();

		if (PerfExplorerModel.getModel().getConstantProblem().booleanValue()) {
			// log axis, to make the chart more readable
        	LogarithmicAxis axis = new LogarithmicAxis("Exclusive " + PerfExplorerModel.getModel().getMetricName() + " for Event");
        	axis.setAutoRangeIncludesZero(true);
        	axis.setAllowNegativesFlag(true);
        	axis.setLog10TickLabelsFlag(true);
        	plot.setRangeAxis(0, axis);
 		}else {
			// otherwise, give the inclusive time its own axis
        	NumberAxis axis2 = new NumberAxis("Inclusive " + PerfExplorerModel.getModel().getMetricName() + " for " + (String)rawData.getRowLabels().get(0));
        	axis2.setAutoRangeIncludesZero(false);
        	plot.setRangeAxis(1, axis2);
		}
        XYDataset dataset2 = new CorrelationPlotDataset(rawData, true);
        plot.setDataset(1, dataset2);
        plot.mapDatasetToRangeAxis(1, 1);
        StandardXYItemRenderer renderer2 = new StandardXYItemRenderer();
        renderer2.setSeriesPaint(0, Color.black);
        renderer2.setBaseShapesVisible(true);
        renderer2.setSeriesStroke(
            0, new BasicStroke(
                2.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                1.0f, new float[] {10.0f, 6.0f}, 0.0f
            )
        );
		renderer2.setBaseToolTipGenerator(new StandardXYToolTipGenerator(
			StandardXYToolTipGenerator.DEFAULT_TOOL_TIP_FORMAT,
			new DecimalFormat("processors: #######"), 
			new DecimalFormat("value: #,##0.00")));

        plot.setRenderer(1, renderer2);
		return new PerfExplorerChart(chart, "Total " + PerfExplorerModel.getModel().getMetricName() + " Breakdown");
	}

	public static PerfExplorerChart doEfficiencyChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			ChartDataType.RELATIVE_EFFICIENCY);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List<String> rowLabels = rawData.getRowLabels();
		double ideal, ratio = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List<double[]> row = rawData.getRowData(y);
        	XYSeries s = new XYSeries(shortName((String)rowLabels.get(y)), true, false);
				double[] baseline = (row.get(0));
				for (int x = 0 ; x < row.size() ; x++) {
					double[] values = (row.get(x));
					ratio = baseline[0]/values[0];
					if (PerfExplorerModel.getModel().getConstantProblem().booleanValue())
						ideal = baseline[1] * ratio;
					else 
						ideal = baseline[1];
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

	/**
	 * This method will produce a general line chart, with 
	 * one or more series of data, with anything you want on
	 * the x-axis, and some measurement on the y-axis.
	 *
	 */
	public static PerfExplorerChart doGeneralChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		PerfExplorerModel model = PerfExplorerModel.getModel();
		RMIGeneralChartData rawData = server.requestGeneralChartData(
			model, ChartDataType.PARAMETRIC_STUDY_DATA);

        PECategoryDataset dataset = new PECategoryDataset();
		if (rawData.getCategoryType() == Integer.class) {
			if (model.getChartScalability()) {

				// create an "ideal" line.
        		dataset.addValue(1.0, "Ideal", new Integer(rawData.getMinimum()));
        		dataset.addValue(rawData.getMaximum()/rawData.getMinimum(), "Ideal", 
					new Integer(rawData.getMaximum()));

				// get the baseline values
				edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData.CategoryDataRow baseline = rawData.getRowData(0);

				// iterate through the values
				for (int i = 0 ; i < rawData.getRows() ; i++) {
					edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
					if (!shortName(row.series).equals(shortName(baseline.series))) {
						//System.out.println(shortName(row.series));
						baseline = row;
					}
					if (model.getConstantProblem().booleanValue()) {
						//System.out.println("value: " + row.value);
						
						double ratio = baseline.categoryInteger.doubleValue() / row.categoryInteger.doubleValue();
						//System.out.println("ratio: " + ratio);
						double efficiency = baseline.value/row.value;
						//System.out.println("efficiency: " + efficiency);
        				dataset.addValue(efficiency / ratio, shortName(row.series), row.categoryInteger);
					} else {
        				dataset.addValue(baseline.value / row.value, shortName(row.series), row.categoryInteger);
					}
				}

				// create an "ideal" line.
				List<Integer> keys = dataset.getColumnKeys();
				for (int i = 0 ; i < keys.size() ; i++) {
					Integer key = (Integer)keys.get(i);
        			dataset.addValue(key.doubleValue()/rawData.getMinimum(), "Ideal", key);
				}

			} else {
				// iterate through the values
				for (int i = 0 ; i < rawData.getRows() ; i++) {
					edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
        			dataset.addValue(row.value / 1000000, shortName(row.series), row.categoryInteger);
				}
			}
		} else {
			// iterate through the values
			for (int i = 0 ; i < rawData.getRows() ; i++) {
				edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
        		dataset.addValue(row.value / 1000000, shortName(row.series), row.categoryString);
			}
		}

		PlotOrientation orientation = PlotOrientation.VERTICAL;
		if (model.getChartHorizontal()) {
            orientation = PlotOrientation.HORIZONTAL;
		}

        JFreeChart chart = ChartFactory.createLineChart(
            model.getChartTitle(),  // chart title
            model.getChartXAxisLabel(),  // domain axis label
            model.getChartYAxisLabel(),  // range axis label
            dataset,                         // data
            orientation,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		// customize the chart! TODO: This is probably unnecessary
       // LegendTitle legend = chart.getLegend();
       // legend.setDisplaySeriesShapes(true);
        
        // get a reference to the plot for further customisation...
        CategoryPlot plot = (CategoryPlot)chart.getPlot();
     
        //StandardXYItemRenderer renderer = (StandardXYItemRenderer) plot.getRenderer();
		LineAndShapeRenderer renderer = (LineAndShapeRenderer)plot.getRenderer();
        renderer.setBaseShapesFilled(true);
        renderer.setBaseShapesVisible(true);
        renderer.setDrawOutlines(true);
        renderer.setBaseItemLabelsVisible(true);
		if (model.getChartScalability()) {
			//renderer.setDrawShapes(false);
		}

		for (int i = 0 ; i < rawData.getRows() ; i++) {
			renderer.setSeriesStroke(i, new BasicStroke(2.0f));
		}

        // change the auto tick unit selection to integer units only...
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
		rangeAxis.setAutoRangeIncludesZero(true);

		if (rawData.getCategoryType() == Integer.class) {
			// don't mess with the domain axis
		} else {
        	CategoryAxis domainAxis = plot.getDomainAxis();
			//domainAxis.setSkipCategoryLabelsToFit(true);//TODO: This was removed but can be faked. Do we need it?
			domainAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_45);
		}

		if (model.getChartLogYAxis()) {
        	LogarithmicAxis axis = new LogarithmicAxis(
				PerfExplorerModel.getModel().getChartYAxisLabel());
        	axis.setAutoRangeIncludesZero(true);
        	axis.setAllowNegativesFlag(true);
        	axis.setLog10TickLabelsFlag(true);
        	plot.setRangeAxis(0, axis);
 		}

		return new PerfExplorerChart(chart, "General Chart");
	}

	public static PerfExplorerChart doEfficiencyEventsChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			ChartDataType.RELATIVE_EFFICIENCY_EVENTS);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List<String> rowLabels = rawData.getRowLabels();
		double ideal, ratio = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List<double[]> row = rawData.getRowData(y);
        	XYSeries s = new XYSeries(shortName(rowLabels.get(y)), true, false);
			double[] baseline = (row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (row.get(x));
				ratio = baseline[0]/values[0];
				if (PerfExplorerModel.getModel().getConstantProblem().booleanValue())
					ideal = baseline[1] * ratio;
				else 
					ideal = baseline[1];
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
			ChartDataType.RELATIVE_EFFICIENCY_ONE_EVENT);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List<String> rowLabels = rawData.getRowLabels();
		double ideal, ratio = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List<double[]> row = rawData.getRowData(y);
        	XYSeries s = new XYSeries(shortName(rowLabels.get(y)), true, false);
			double[] baseline = (row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (row.get(x));
				ratio = baseline[0]/values[0];
				if (PerfExplorerModel.getModel().getConstantProblem().booleanValue())
					ideal = baseline[1] * ratio;
				else 
					ideal = baseline[1];
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
			ChartDataType.RELATIVE_EFFICIENCY);

        XYSeriesCollection dataset = new XYSeriesCollection();

		List<String> rowLabels = rawData.getRowLabels();
		double minx = 99999, maxx = 0;
		double ideal, ratio, efficiency = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List<double[]> row = rawData.getRowData(y);
        	XYSeries s = new XYSeries(shortName(rowLabels.get(y)), true, false);
			double[] baseline = (row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (row.get(x));
				ratio = baseline[0]/values[0];
				if (PerfExplorerModel.getModel().getConstantProblem().booleanValue())
					ideal = baseline[1] * ratio;
				else 
					ideal = baseline[1];
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
			ChartDataType.RELATIVE_EFFICIENCY_EVENTS);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List<String> rowLabels = rawData.getRowLabels();
		double minx = 99999, maxx = 0;
		double ideal = 0, ratio = 0, efficiency = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List<double[]> row = rawData.getRowData(y);
        	XYSeries s = new XYSeries(shortName(rowLabels.get(y)), true, false);
			double[] baseline = (row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (row.get(x));
				ratio = baseline[0]/values[0];
				if (PerfExplorerModel.getModel().getConstantProblem().booleanValue())
					ideal = baseline[1] * ratio;
				else 
					ideal = baseline[1];
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
			ChartDataType.RELATIVE_EFFICIENCY_ONE_EVENT);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		double minx = 99999, maxx = 0;
		double ideal = 0, ratio = 0, efficiency = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries(shortName((String)rowLabels.get(y)), true, false);
			double[] baseline = (double[])(row.get(0));
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				ratio = baseline[0]/values[0];
				if (PerfExplorerModel.getModel().getConstantProblem().booleanValue())
					ideal = baseline[1] * ratio;
				else 
					ideal = baseline[1];
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
			ChartDataType.RELATIVE_EFFICIENCY);

		int timesteps = Integer.parseInt(PerfExplorerModel.getModel().getTotalTimesteps());
        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries(shortName((String)rowLabels.get(y)), true, false);
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

	public static PerfExplorerChart doTotalTimeChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			ChartDataType.RELATIVE_EFFICIENCY);

		int decreasing = 0, total = 0;
		double lastValue = 0.0;
        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries(shortName((String)rowLabels.get(y)), true, false);
			total = total + row.size();
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
        		s.add(values[0], values[1]);
				if (lastValue > values[1])
					decreasing++;
				lastValue = values[1];
			}
        	dataset.addSeries(s);
		}

        JFreeChart chart = ChartFactory.createXYLineChart(
            "Total Execution:" +
			PerfExplorerModel.getModel().getMetricName(),  // chart title
            "Number of Processors",          // domain axis label
            PerfExplorerModel.getModel().getMetricNameUnits(),     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );

		customizeChart(chart, rawData.getRows(), false);

		// if decreasing, assume strong scaling, and do Log axis for range.
		if (decreasing > total/2) {
        	XYPlot plot = chart.getXYPlot();
        	LogarithmicAxis axis = new LogarithmicAxis(
				PerfExplorerModel.getModel().getMetricName());
        	axis.setAutoRangeIncludesZero(true);
        	axis.setAllowNegativesFlag(true);
        	axis.setLog10TickLabelsFlag(true);
        	plot.setRangeAxis(0, axis);
 		}

		return new PerfExplorerChart(chart, "Total Execution " + PerfExplorerModel.getModel().getMetricName() + "");
	}

	public static PerfExplorerChart doCommunicationChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData1 = server.requestChartData(
			PerfExplorerModel.getModel(), 
			ChartDataType.TOTAL_FOR_GROUP);

		RMIChartData rawData2 = server.requestChartData(
			PerfExplorerModel.getModel(), 
			ChartDataType.RELATIVE_EFFICIENCY);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData1.getRowLabels();
		for (int y = 0 ; y < rawData1.getRows() ; y++) {
			List row1 = rawData1.getRowData(y);
			List row2 = rawData2.getRowData(y);
        	XYSeries s = new XYSeries(shortName((String)rowLabels.get(y)), true, false);
			for (int x = 0 ; x < row1.size() ; x++) {
				double[] values1 = (double[])(row1.get(x));
				double[] values2 = (double[])(row2.get(x));
        		s.add(values1[0], values1[1]/values2[1]);
			}
        	dataset.addSeries(s);
		}

        JFreeChart chart = ChartFactory.createXYLineChart(
            PerfExplorerModel.getModel().getGroupName() + 
			" " + PerfExplorerModel.getModel().getMetricName() + " / Total " + PerfExplorerModel.getModel().getMetricName() + " - " +   // chart title
			PerfExplorerModel.getModel().toString(),
            "Number of Processors",          // domain axis label
            "Fraction",     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );

		customizeChart(chart, rawData1.getRows(), false);
		return new PerfExplorerChart(chart, PerfExplorerModel.getModel().getGroupName() + " " + PerfExplorerModel.getModel().getMetricName() + " / Total " + PerfExplorerModel.getModel().getMetricName());
	}

	public static PerfExplorerChart doEfficiencyPhasesChart () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData rawData = server.requestChartData(
			PerfExplorerModel.getModel(), 
			ChartDataType.RELATIVE_EFFICIENCY_PHASES);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		double ideal, ratio = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries(shortName((String)rowLabels.get(y)), true, false);
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
			ChartDataType.RELATIVE_EFFICIENCY_PHASES);

        XYSeriesCollection dataset = new XYSeriesCollection();
		List rowLabels = rawData.getRowLabels();
		double minx = 99999, maxx = 0;
		double ideal, ratio, efficiency = 0;
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
        	XYSeries s = new XYSeries(shortName((String)rowLabels.get(y)), true, false);
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
			ChartDataType.FRACTION_OF_TOTAL_PHASES);

        DefaultTableXYDataset dataset = new DefaultTableXYDataset();
		List rowLabels = rawData.getRowLabels();
		for (int y = 0 ; y < rawData.getRows() ; y++) {
			List row = rawData.getRowData(y);
			XYSeries s = new XYSeries(shortName((String)rowLabels.get(y)), true, false);
			for (int x = 0 ; x < row.size() ; x++) {
				double[] values = (double[])(row.get(x));
				s.add(values[0], values[1]);
			}
			dataset.addSeries(s);
		}

		JFreeChart chart = ChartFactory.createStackedXYAreaChart(
            "Total " + PerfExplorerModel.getModel().getMetricName() + " Breakdown for " + 
			PerfExplorerModel.getModel().toString(),
            "Number of Processors",          // domain axis label
            "Percentage of Total " + PerfExplorerModel.getModel().getMetricName(),     // range axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		return new PerfExplorerChart(chart, "Total " + PerfExplorerModel.getModel().getMetricName() + " Breakdown");
	}


	private static void customizeChart (JFreeChart chart, int rows, boolean lastLineIdeal) {
		// customize the chart!  TODO: THis is probably unnecessary
        //StandardLegend legend = (StandardLegend) chart.getLegend();
        //legend.setDisplaySeriesShapes(true);
        
        // get a reference to the plot for further customisation...
        XYPlot plot = chart.getXYPlot();
     
        //StandardXYItemRenderer renderer = (StandardXYItemRenderer) plot.getRenderer();
		XYLineAndShapeRenderer renderer = null;
		if (lastLineIdeal)
			renderer = new SpeedupXYLineAndShapeRenderer(rows);
		else
			renderer = new XYLineAndShapeRenderer();
        renderer.setBaseShapesFilled(true);
        //renderer.setPlotShapes(true);
        renderer.setBaseItemLabelsVisible(true);
		renderer.setBaseToolTipGenerator(
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

	public static String shortName(String longName) {
		StringTokenizer st = new StringTokenizer(longName, "(");
		String shorter = null;
		try {
			shorter = st.nextToken();
			if (shorter.length() < longName.length()) {
				shorter = shorter + "()";
			}
		} catch (NoSuchElementException e) {
			shorter = longName;
		}
		return shorter;
	}

}
