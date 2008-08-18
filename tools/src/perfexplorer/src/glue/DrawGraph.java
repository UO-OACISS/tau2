/**
 * 
 */
package glue;

import java.awt.BasicStroke;
import java.util.List;
import java.util.HashSet;
import java.util.Set;

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

import client.MyCategoryAxis;
import client.PerfExplorerChart;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class DrawGraph extends AbstractPerformanceOperation {

    protected Set<String> _events = null;
    protected Set<String> _metrics = null;
    protected Set<Integer> _threads = null;
	
    public static final int TRIALNAME = 0;
    public static final int EVENTNAME = 1;
    public static final int METRICNAME = 2;
    public static final int THREADNAME = 3;
    public static final int USEREVENTNAME = 4;
    public static final int PROCESSORCOUNT = 5;

    protected int seriesType = METRICNAME;  // sets the series name
    protected int categoryType = THREADNAME;  // sets the X axis
    protected int valueType = AbstractResult.EXCLUSIVE;
    protected boolean logYAxis = false;
    
    protected String title = "My Chart";
    protected String yAxisLabel = "value";
    protected String xAxisLabel = "category";
	protected boolean userEvents = false;
    
	/**
	 * @param input
	 */
	public DrawGraph(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public DrawGraph(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public DrawGraph(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {

        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
		Set<String> categories = new HashSet<String>();

        for (PerformanceResult input : inputs) {
        	// THESE ARE LOCAL COPIES!
            Set<String> events = null;
            Set<String> metrics = null;
            Set<Integer> threads = null;
            
            if (this._events == null) {
				if (userEvents) {
					events = input.getUserEvents();
				} else {
            		events = input.getEvents();
				}
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
            
			if (userEvents) {
            	for (String event : events) {
           			for (Integer thread : threads) {
           				// set the series name
           				if (seriesType == TRIALNAME) {
           					seriesName = input.toString();
           				} else if (seriesType == USEREVENTNAME) {
           					seriesName = event;
           				} else if (seriesType == THREADNAME) {
           					seriesName = thread.toString();
           				}
           				
           				// set the category name
           				if (categoryType == TRIALNAME) {
           					categoryName = input.toString();
           				} else if (categoryType == USEREVENTNAME) {
           					categoryName = event;
           				} else if (categoryType == THREADNAME) {
           					categoryName = thread.toString();
           				} else if (categoryType == PROCESSORCOUNT) {
           					categoryName = Integer.toString(input.getOriginalThreads());
           				}

           				dataset.addValue(input.getDataPoint(thread, event, null, valueType), seriesName, categoryName);
						categories.add(categoryName);
           			}
           		}
			} else {
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
           					} else if (categoryType == PROCESSORCOUNT) {
           						categoryName = Integer.toString(input.getOriginalThreads());
            				}

            				dataset.addValue(input.getDataPoint(thread, event, metric, valueType), seriesName, categoryName);
							categories.add(categoryName);
            			}
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

		PerfExplorerChart chartWindow = new PerfExplorerChart(chart, "General Chart");
		return null;
	}

	/**
	 * @return the eVENTNAME
	 */
	public static int getEVENTNAME() {
		return EVENTNAME;
	}

	/**
	 * @return the mETRICNAME
	 */
	public static int getMETRICNAME() {
		return METRICNAME;
	}

	/**
	 * @return the tHREADNAME
	 */
	public static int getTHREADNAME() {
		return THREADNAME;
	}

	/**
	 * @return the tRIALNAME
	 */
	public static int getTRIALNAME() {
		return TRIALNAME;
	}

	/**
	 * @return the _events
	 */
	public Set<String> get_events() {
		return _events;
	}

	/**
	 * @param _events the _events to set
	 */
	public void set_events(Set<String> _events) {
		this._events = _events;
	}

	/**
	 * @return the _metrics
	 */
	public Set<String> get_metrics() {
		return _metrics;
	}

	/**
	 * @param _metrics the _metrics to set
	 */
	public void set_metrics(Set<String> _metrics) {
		this._metrics = _metrics;
	}

	/**
	 * @return the _threads
	 */
	public Set<Integer> get_threads() {
		return _threads;
	}

	/**
	 * @param _threads the _threads to set
	 */
	public void set_threads(Set<Integer> _threads) {
		this._threads = _threads;
	}

	/**
	 * @return the categoryType
	 */
	public int getCategoryType() {
		return categoryType;
	}

	/**
	 * @param categoryType the categoryType to set
	 */
	public void setCategoryType(int categoryType) {
		this.categoryType = categoryType;
	}

	/**
	 * @return the logYAxis
	 */
	public boolean isLogYAxis() {
		return logYAxis;
	}

	/**
	 * @param logYAxis the logYAxis to set
	 */
	public void setLogYAxis(boolean logYAxis) {
		this.logYAxis = logYAxis;
	}

	/**
	 * @return the seriesType
	 */
	public int getSeriesType() {
		return seriesType;
	}

	/**
	 * @param seriesType the seriesType to set
	 */
	public void setSeriesType(int seriesType) {
		this.seriesType = seriesType;
	}

	/**
	 * @return the title
	 */
	public String getTitle() {
		return title;
	}

	/**
	 * @param title the title to set
	 */
	public void setTitle(String title) {
		this.title = title;
	}

	/**
	 * @return the valueType
	 */
	public int getValueType() {
		return valueType;
	}

	/**
	 * @param valueType the valueType to set
	 */
	public void setValueType(int valueType) {
		this.valueType = valueType;
	}

	/**
	 * @return the xaxisLabel
	 */
	public String getXAxisLabel() {
		return xAxisLabel;
	}

	/**
	 * @param xAxisLabel the xAxisLabel to set
	 */
	public void setXAxisLabel(String xAxisLabel) {
		this.xAxisLabel = xAxisLabel;
	}

	/**
	 * @return the yAxisLabel
	 */
	public String getYAxisLabel() {
		return yAxisLabel;
	}

	/**
	 * @param yAxisLabel the yAxisLabel to set
	 */
	public void setYAxisLabel(String yAxisLabel) {
		this.yAxisLabel = yAxisLabel;
	}

	public void setUserEvents(boolean userEvents) {
		this.userEvents = userEvents;
	}
}
