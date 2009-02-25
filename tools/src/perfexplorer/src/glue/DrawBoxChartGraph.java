/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.renderer.category.BoxAndWhiskerRenderer;
import org.jfree.chart.renderer.category.CategoryItemRenderer;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerChart;

/**
 * @author khuck
 *
 */
public class DrawBoxChartGraph extends DrawGraph {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6431239612339815694L;

	/**
	 * @param input
	 */
	public DrawBoxChartGraph(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public DrawBoxChartGraph(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public DrawBoxChartGraph(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {

		DefaultBoxAndWhiskerCategoryDataset dataset = new DefaultBoxAndWhiskerCategoryDataset();

        for (PerformanceResult input : inputs) {
        	// THESE ARE LOCAL COPIES!
            Set<String> events = null;
            Set<String> metrics = null;
            Set<Integer> threads = null;
            
            if (this._events == null) {
            	events = input.getEvents();
            } else {
            	events = this._events;
            }
            
            if (this._metrics == null) {
            	metrics = input.getMetrics();
            } else {
            	metrics = this._metrics;
            }

            if (this._threads == null) {
            	threads = input.getThreads();
            } else {
            	threads = this._threads;
            }

            String seriesName = "";
            String categoryName = "";
            
            for (String event : events) {
            	for (String metric : metrics) {
            		List<Double> valueList = new ArrayList<Double>();
            		for (Integer thread : threads) {
            			// set the series name
            			if (seriesType == TRIALNAME) {
            				seriesName = input.toString();
            			} else if (seriesType == EVENTNAME) {
            				seriesName = event;
            			} else if (seriesType == METRICNAME) {
            				seriesName = metric;
            			} else if (seriesType == THREADNAME) {
            				seriesName = thread.toString();
            			}
            			
            			// set the category name
            			if (categoryType == TRIALNAME) {
            				categoryName = input.toString();
            			} else if (categoryType == EVENTNAME) {
            				categoryName = event;
            			} else if (categoryType == METRICNAME) {
            				categoryName = metric;
            			} else if (categoryType == THREADNAME) {
            				categoryName = thread.toString();
            			}
            			valueList.add(input.getDataPoint(thread, event, metric, valueType));
            		}
        			dataset.add(valueList, seriesName, categoryName);
            	}
            }
        }
        
/*        JFreeChart chart = ChartFactory.createLineChart(
            this.title,  // chart title
            this.xAxisLabel,  // domain Axis label
            this.yAxisLabel,  // range Axis label
            dataset,                         // data
            PlotOrientation.VERTICAL,        // the plot orientation
            true,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		// customize the chart!
        StandardLegend legend = (StandardLegend) chart.getLegend();
        legend.setDisplaySeriesShapes(true);
        
        // get a reference to the plot for further customisation...
        CategoryPlot plot = (CategoryPlot)chart.getPlot();
     
        //StandardXYItemRenderer renderer = (StandardXYItemRenderer) plot.getRenderer();
		LineAndShapeRenderer renderer = (LineAndShapeRenderer)plot.getRenderer();
        renderer.setDefaultShapesFilled(true);
        renderer.setDrawShapes(true);
        renderer.setDrawLines(true);
        renderer.setItemLabelsVisible(true);

		for (int i = 0 ; i < dataset.getRowCount() ; i++) {
			renderer.setSeriesStroke(i, new BasicStroke(2.0f));
		}

		if (this.logYAxis) {
        	LogarithmicAxis axis = new LogarithmicAxis(yAxisLabel);
        	axis.setAutoRangeIncludesZero(true);
        	axis.setAllowNegativesFlag(true);
        	axis.setLog10TickLabelsFlag(true);
        	plot.setRangeAxis(0, axis);
 		}

        // change the auto tick unit selection to integer units only...
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
		rangeAxis.setAutoRangeIncludesZero(false);

        CategoryAxis domainAxis = plot.getDomainAxis();
        domainAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_45);
*/		

        CategoryAxis domainAxis = new CategoryAxis(null);
        NumberAxis rangeAxis = new NumberAxis("Value");
        CategoryItemRenderer renderer = new BoxAndWhiskerRenderer();
        CategoryPlot plot = new CategoryPlot(
            dataset, domainAxis, rangeAxis, renderer
        );
        JFreeChart chart = new JFreeChart("Box-and-Whisker Chart Demo 1", plot);

        chart.setBackgroundPaint(Color.white);

        plot.setBackgroundPaint(Color.lightGray);
        plot.setDomainGridlinePaint(Color.white);
        plot.setDomainGridlinesVisible(true);
        plot.setRangeGridlinePaint(Color.white);

        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

		PerfExplorerChart chartWindow = new PerfExplorerChart(chart, "General Chart");
		return null;
	}

}
