package edu.uoregon.tau.perfexplorer.glue;

import java.awt.BasicStroke;
import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.StringTokenizer;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;

import edu.uoregon.tau.common.VectorExport;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.client.MyCategoryAxis;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerChart;

/**
 * <p>
 * The DrawGraph class is a PerfExplorer Operation class for drawing a line
 * graph from an analysis script.  The creation of the graph is fairly
 * straightforward.  One or more input {@link PerformanceResult} objects
 * are used as the input for the constructor.  After the constructor,
 * various options for the graph are set.  The fields which are commonly
 * set are the series type ({@link #setSeriesType}), the category type
 * ({@link #setCategoryType}), and the value type ({@link #setValueType}).
 * Once the options are set, the {@link #processData} method is called
 * to generate the graph.
 * </p>
 *
 * <p>
 * This class has undefined behavior when running PerfExplorer without
 * the GUI.
 * </p>
 *
 * <p>
 * Example code from Python script:
 * </p>
 * <pre>

from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

True = 1
False = 0

def loadFile(fileName):
	# load the trial
	files = []
	files.append(fileName)
	input = DataSourceResult(DataSourceResult.PPK, files, False)
	return input

def loadFromFiles():
	inputs = ArrayList()
	inputs.add(loadFile("2.ppk"))
	inputs.add(loadFile("4.ppk"))
	inputs.add(loadFile("6.ppk"))
	inputs.add(loadFile("8.ppk"))
	return inputs

def drawGraph(results):
	metric = "Time"
	grapher = DrawGraph(results)
	metrics = HashSet()
	metrics.add(metric)
	grapher.setMetrics(metrics)
	grapher.setLogYAxis(False)
	grapher.setShowZero(True)
	grapher.setTitle("Graph of Multiple Trials: " + metric)
	grapher.setSeriesType(DrawGraph.EVENTNAME)
	grapher.setUnits(DrawGraph.SECONDS)
	grapher.setCategoryType(DrawGraph.PROCESSORCOUNT)
	grapher.setXAxisLabel("Processor Count")
	grapher.setValueType(AbstractResult.EXCLUSIVE)
	grapher.setYAxisLabel("Exclusive " + metric + " (seconds)")
	grapher.processData()

def main():
	print "--------------- JPython test script start ------------"
	inputs = loadFromFiles()

	# extract the event of interest
	events = ArrayList()
	events.add("MPI_Send()")
	extractor = ExtractEventOperation(inputs, events)
	extracted = extractor.processData()

	drawGraph(extracted)
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()

 * </pre>
 *
 * <P>CVS $Id: DrawGraph.java,v 1.14 2009/04/09 00:23:51 khuck Exp $</P>
 * @author khuck
 * @version 0.2
 * @since   0.2
 *
 */
public class DrawGraph extends AbstractPerformanceOperation {

	private static final long serialVersionUID = -5587605162968129610L;
	protected Set<String> _events = null;
    protected Set<String> _metrics = null;
    protected Set<Integer> _threads = null;
	
    /** 
	 * Constant for specifying that the Trial Name should be used for the
	 * series name or the category axis. 
	 * @see #setSeriesType
	 * @see #setCategoryType
	 */
    public static final int TRIALNAME = 0;
    /** 
	 * Constant for specifying that the Event Name should be used for the
	 * series name or the category axis.
	 * @see #setSeriesType
	 * @see #setCategoryType
	 */
    public static final int EVENTNAME = 1;
    /** 
	 * Constant for specifying that the Metric Name should be used for the
	 * series name or the category axis.  
	 * @see #setSeriesType
	 * @see #setCategoryType
	 */
    public static final int METRICNAME = 2;
    /** 
	 * Constant for specifying that the Thread Name should be used for the
	 * series name or the category axis.  
	 * @see #setSeriesType
	 * @see #setCategoryType
	 */
    public static final int THREADNAME = 3;
    /** 
	 * Constant for specifying that the UserEvent Name should be used for the
	 * series name or the category axis.  
	 * @see #setSeriesType
	 * @see #setCategoryType
	 */
    public static final int USEREVENTNAME = 4;
    /** 
	 * Constant for specifying that the Processor Count should be used for the
	 * series name or the category axis.  
	 * @see #setSeriesType
	 * @see #setCategoryType
	 */
    public static final int PROCESSORCOUNT = 5;
    /** 
	 * Constant for specifying that a Metadata field should be used for the
	 * series name or the category axis.  {@link #setMetadataField} should
	 * be called to specify which metadata field to use.
	 * @see #setSeriesType
	 * @see #setCategoryType
	 * @see #setMetadataField 
	 */
    public static final int METADATA = 6;

    /**
	 * Constant for specifying the Y axis units for the graph should be
	 * microseconds (10xe-6 seconds).  This is the default.
	 * @see #setUnits
     */
	public static final int MICROSECONDS = 1;
    /**
	 * Constant for specifying the Y axis units for the graph should be
	 * milliseconds (10xe-3 seconds).
	 * @see #setUnits
     */
	public static final int MILLISECONDS = 1000;
    /**
	 * Constant for specifying the Y axis units for the graph should be
	 * thousands (10xe3 units).
	 * @see #setUnits
     */
	public static final int THOUSANDS = 1000;
    /**
	 * Constant for specifying the Y axis units for the graph should be seconds.
	 * @see #setUnits
     */
	public static final int SECONDS = 1000000;
    /**
	 * Constant for specifying the Y axis units for the graph should be
	 * millions (10xe6 units).
	 * @see #setUnits
     */
	public static final int MILLIONS = 1000000;
    /**
	 * Constant for specifying the Y axis units for the graph should be minutes.
	 * @see #setUnits
     */
	public static final int MINUTES = 60000000;
    /**
	 * Constant for specifying the Y axis units for the graph should be
	 * thousands (10xe9 units).
	 * @see #setUnits
     */
	public static final int BILLIONS = 1000000000;

	protected int units = MICROSECONDS;
    protected int seriesType = METRICNAME;  // sets the series name
    protected int categoryType = THREADNAME;  // sets the X axis
    protected int valueType = AbstractResult.EXCLUSIVE;
    protected boolean logYAxis = false;
    protected boolean showZero = false;
	protected int categoryNameLength = 0;
    protected String title = "My Chart";
    protected String yAxisLabel = "value";
    protected String xAxisLabel = "category";
	protected boolean userEvents = false;
    protected String metadataField = "";
	protected PerfExplorerChart chartWindow = null;
	protected boolean shortenNames = false;
    
	/**
	 * Creates a graph drawing operator.
	 *
	 * @param input
	 */
	public DrawGraph(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * Creates a graph drawing operator.
	 *
	 * @param trial
	 */
	public DrawGraph(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * Creates a graph drawing operator.
	 *
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
            				seriesName = input.getTrial().getName();
           				} else if (seriesType == USEREVENTNAME) {
           					seriesName = event;
           				} else if (seriesType == THREADNAME) {
           					seriesName = thread.toString();
           				}
           				
           				// set the category name
           				if (categoryType == TRIALNAME) {
            				categoryName = input.getTrial().getName();
           				} else if (categoryType == USEREVENTNAME) {
           					categoryName = event;
           				} else if (categoryType == THREADNAME) {
           					categoryName = thread.toString();
           				} else if (categoryType == PROCESSORCOUNT) {
           					categoryName = Integer.toString(input.getOriginalThreads());
           				}

           				dataset.addValue(input.getDataPoint(thread, event, null, valueType)/this.units, seriesName, categoryName);
						categories.add(categoryName);
						categoryNameLength = categoryNameLength += categoryName.length();
           			}
           		}
			} else {
            	for (String event : events) {
            		for (String metric : metrics) {
            			for (Integer thread : threads) {
            				// set the series name
            				if (seriesType == TRIALNAME) {
            					seriesName = input.getTrial().getName();
            				} else if (seriesType == EVENTNAME) {
            					if (shortenNames) {
            						seriesName = this.shortName(event);
            					} else {
            						seriesName = event;
            					}
            				} else if (seriesType == METRICNAME) {
            					seriesName = metric;
            				} else if (seriesType == THREADNAME) {
            					seriesName = thread.toString();
            				}
            			
            				// set the category name
            				if (categoryType == TRIALNAME) {
            					//categoryName = input.getTrial().getName();
            					categoryName = input.getName();
            				} else if (categoryType == EVENTNAME) {
            					if (shortenNames) {
            						categoryName = this.shortName(event);
            					} else {
            						categoryName = event;
            					}
            				} else if (categoryType == METRICNAME) {
            					categoryName = metric;
            				} else if (categoryType == THREADNAME) {
            					categoryName = thread.toString();
           					} else if (categoryType == PROCESSORCOUNT) {
           						categoryName = Integer.toString(input.getOriginalThreads());
           					} else if (categoryType == METADATA) {
           						TrialMetadata meta = new TrialMetadata(input.getTrial());
           						categoryName = meta.getCommonAttributes().get(this.metadataField);
            				}

            				dataset.addValue(input.getDataPoint(thread, event, metric, valueType)/this.units, seriesName, categoryName);
							categories.add(categoryName);
							categoryNameLength = categoryNameLength += categoryName.length();
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
		// set to a common style
		Utility.applyDefaultChartTheme(chart);
        
        // get a reference to the plot for further customisation...
        CategoryPlot plot = (CategoryPlot)chart.getPlot();
     
        //StandardXYItemRenderer renderer = (StandardXYItemRenderer) plot.getRenderer();
		LineAndShapeRenderer renderer = (LineAndShapeRenderer)plot.getRenderer();
        renderer.setBaseShapesFilled(true);
        renderer.setBaseShapesVisible(true);
        renderer.setDrawOutlines(true);
        renderer.setBaseItemLabelsVisible(true);

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
		rangeAxis.setAutoRangeIncludesZero(this.showZero);
		
        MyCategoryAxis domainAxis = null;
        domainAxis = new MyCategoryAxis(xAxisLabel);
        if (categories.size() > 40){
            domainAxis.setTickLabelSkip(categories.size()/20);
            domainAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_45);
        } else if (categories.size() > 20) {
            domainAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_45);
        } else if (categoryNameLength / categories.size() > 10) {
            domainAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_45);
		}

        plot.setDomainAxis(domainAxis);

		this.chartWindow = new PerfExplorerChart(chart, "General Chart");
		return null;
	}

	/**
	 * Set the events used for the graph.
	 * @param events The Set of events used for the graph.
	 */
	public void setEvents(Set<String> events) {
		this._events = events;
	}

	/**
	 * Set the metrics used for the graph.
	 * @param metrics The Set of metrics used for the graph.
	 */
	public void setMetrics(Set<String> metrics) {
		this._metrics = metrics;
	}

	/**
	 * Set the threads used for the graph.
	 * @param threads The Set of threads used for the graph.
	 */
	public void setThreads(Set<Integer> threads) {
		this._threads = threads;
	}

	/**
	 * Set the category type for the graph.
	 * @param categoryType The category type
	 * @see #TRIALNAME
	 * @see #EVENTNAME
	 * @see #METRICNAME
	 * @see #THREADNAME
	 * @see #USEREVENTNAME
	 * @see #PROCESSORCOUNT
	 * @see #METADATA
	 */
	public void setCategoryType(int categoryType) {
		this.categoryType = categoryType;
	}

	/**
	 * Set whether or not the Y axis is a log scale.
	 * @param logYAxis Whether or not the Y Axis is a Log scale
	 */
	public void setLogYAxis(boolean logYAxis) {
		this.logYAxis = logYAxis;
	}

	/**
	 * Set the series type for the graph.
	 * @param seriesType The series type
	 * @see #TRIALNAME
	 * @see #EVENTNAME
	 * @see #METRICNAME
	 * @see #THREADNAME
	 * @see #USEREVENTNAME
	 * @see #PROCESSORCOUNT
	 * @see #METADATA
	 */
	public void setSeriesType(int seriesType) {
		this.seriesType = seriesType;
	}

	/**
	 * Set the title for the graph.
	 * @param title The title of the graph
	 */
	public void setTitle(String title) {
		this.title = title;
	}

	/**
	 * Set the value type for the graph.
	 * @param valueType The value type
	 * @see AbstractResult#CALLS
	 * @see AbstractResult#EXCLUSIVE
	 * @see AbstractResult#INCLUSIVE
	 * @see AbstractResult#SUBROUTINES
	 * @see AbstractResult#USEREVENT_MAX
	 * @see AbstractResult#USEREVENT_MEAN
	 * @see AbstractResult#USEREVENT_MIN
	 * @see AbstractResult#USEREVENT_NUMEVENTS
	 * @see AbstractResult#USEREVENT_SUMSQR
	 */
	public void setValueType(int valueType) {
		this.valueType = valueType;
	}

	/**
	 * Set the label used for the X Axis.
	 * @param xAxisLabel The label used for the X Axis
	 */
	public void setXAxisLabel(String xAxisLabel) {
		this.xAxisLabel = xAxisLabel;
	}

	/**
	 * Set the label used for the Y Axis.
	 * @param yAxisLabel The label used for the Y Axis
	 */
	public void setYAxisLabel(String yAxisLabel) {
		this.yAxisLabel = yAxisLabel;
	}

	/**
	 * Set whether or not to use user events from the trials.  If false,
	 * interval events are used.
	 * @param userEvents Whether or not to use user events from the trials.
	 */
	public void setUserEvents(boolean userEvents) {
		this.userEvents = userEvents;
	}

	/**
	 * Set whether or not to have the Y axis go all the way from 0 as
	 * a minimum value.
	 * @param showZero Whether or not to have the Y axis go all the way from 0.
	 */
	public void setShowZero(boolean showZero) {
		this.showZero = showZero;
	}

	/**
	 * The metadata field to use for either the series name or the category name.
	 * This method only has meaning if the series type or category type have been
	 * set to {@link #METADATA}.
	 * @param metadataField The metadata field to use for either the series name or
	 * the category name.
	 * @see #setSeriesType
	 * @see #setCategoryType
	 * @see #METADATA
	 */
	public void setMetadataField(String metadataField) {
		this.metadataField = metadataField;
	}

	/**
	 * Set the units to use for the graph.
	 * @param units The units to use for the graph.
	 * @see #MICROSECONDS
	 * @see #MILLISECONDS
	 * @see #SECONDS
	 * @see #MINUTES
	 * @see #THOUSANDS
	 * @see #MILLIONS
	 * @see #BILLIONS
	 */
	public void setUnits(int units) {
		this.units = units;
	}

	/**
	 * Draws the graph to the file name specified.
	 * @param fileName The filename for the graph output.
	 */
	public void drawChartToFile(String fileName) {
		try {
			VectorExport.export(chartWindow, new File(fileName), true, "PerfExplorer", true, true);
		} catch (Exception e) {
			System.err.println("Could not write graph to file:");
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
	}
	
	private String shortName(String longName) {
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
		longName = shorter;
		st = new StringTokenizer(longName, "[{");
		shorter = null;
		try {
			shorter = st.nextToken();
		} catch (NoSuchElementException e) {
			shorter = longName;
		}
		return shorter;
	}

	/**
	 * Sets whether to remove parameters and line numbers from function names.
	 * @param shortenNames Whether to remove parameters and line numbers from function names
	 */
	public void setShortenNames(boolean shortenNames) {
		this.shortenNames = shortenNames;
	}
}
