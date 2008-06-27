/**
 * 
 */
package glue;

import java.awt.BasicStroke;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.Collections;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.StandardLegend;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.statistics.DefaultStatisticalCategoryDataset;
import org.jfree.data.xy.DefaultHighLowDataset;

import client.PerfExplorerChart;
import client.MyCategoryAxis;
import edu.uoregon.tau.common.AlphanumComparator;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class DrawMMMGraph extends DrawGraph {

	private boolean sortXAxis = false;
	private String stripValue = null;

	/**
	 * @param input
	 */
	public DrawMMMGraph(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public DrawMMMGraph(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public DrawMMMGraph(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {

		DefaultCategoryDataset dataset = new DefaultCategoryDataset();
		
		List<PerformanceResult> series = new ArrayList<PerformanceResult>();
		series.add(inputs.get(1));
		series.add(inputs.get(4));
		series.add(inputs.get(5));
		Set<String> categories = new HashSet<String>();

		for (PerformanceResult input : series) {
			// THESE ARE LOCAL COPIES!
	        Set<String> events = null;
	        Set<String> metrics = null;
	        Set<Integer> threads = null;
	        
	        if (this._events == null) {
	        	events = input.getEvents();
	        } else {
	        	events = this._events;
	        }

			// sort the events alphanumerically?
			if (sortXAxis) {
       			if (categoryType == TRIALNAME) {
					// do nothing
       			} else if (categoryType == EVENTNAME) {
   					Set<String> tmpSet = new TreeSet(new AlphanumComparator());
					for (String event : events) {
						tmpSet.add(event);
					}
					events = tmpSet;
       			} else if (categoryType == METRICNAME) {
   					Set<String> tmpSet = new TreeSet(new AlphanumComparator());
					for (String metric : metrics) {
						tmpSet.add(metric);
					}
					metrics = tmpSet;
       			} else if (categoryType == THREADNAME) {
					// do nothing - they are sorted.
       			}
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
	            
	        int i = 0;
	        for (String event : events) {
	        	for (String metric : metrics) {
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
	
	        			
	/*        			means[i] = mean.getDataPoint(thread, event, metric, valueType);
	        			stdevs[i] = stDev.getDataPoint(thread, event, metric, valueType);
	        			mins[i] = min.getDataPoint(thread, event, metric, valueType);
	        			maxs[i] = max.getDataPoint(thread, event, metric, valueType);
	        			categories[i] = categoryName;
	        			i++;
	*/
						if (stripValue != null) {
							categoryName = categoryName.replaceAll(stripValue, "");
							// this is a total hack.  Need a true replacement method.
							categoryName = categoryName.replaceAll("> \\]", "");
						}
						dataset.addValue(input.getDataPoint(thread, event, metric, valueType),
	        				seriesName, categoryName);
						categories.add(categoryName);
	        		}
	        	}
	        }
		}
        
        JFreeChart chart = ChartFactory.createLineChart(
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
        //rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
		rangeAxis.setAutoRangeIncludesZero(false);
		
        MyCategoryAxis domainAxis = null;
		domainAxis = new MyCategoryAxis(xAxisLabel);
		if (categories.size() > 40){
			domainAxis.setTickLabelSkip(categories.size()/20);
        	domainAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_45);
		} else if (categories.size() > 20) {
        	domainAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_45);
		}
        plot.setDomainAxis(domainAxis);

        // create categories label...
        
		PerfExplorerChart chartWindow = new PerfExplorerChart(chart, "General Chart");
		return null;
	}

	public void setSortXAxis(boolean sortXAxis) {
		this.sortXAxis = sortXAxis;
	}

	public boolean getSortXAxis() {
		return this.sortXAxis;
	}

	public void setStripXName(String value) {
		this.stripValue = value;
	}


}
