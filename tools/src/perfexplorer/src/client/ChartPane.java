package client;

import javax.swing.*;

import java.awt.*;
import java.util.Hashtable;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import common.*;
import edu.uoregon.tau.perfdmf.*;
import common.RMIChartData;
import common.RMIGeneralChartData;
import common.ChartDataType;
import java.awt.*;
import java.awt.event.*;
import java.awt.BasicStroke;
import javax.swing.JFrame;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import java.util.List;
import java.util.Vector;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.DefaultTableXYDataset;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.Range;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.StandardLegend;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import java.text.DecimalFormat;
import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.common.VectorExport;


public class ChartPane extends JScrollPane implements ActionListener {

	private static ChartPane thePane = null;

	private JPanel mainPanel = null;
	private ScriptFacade facade = null;
	private static String UPDATE_COMMAND = "UPDATE_COMMAND";
	private JPanel chartPanel = null;

	private JToggleButton mainOnly = new JToggleButton ("Main Only");
	private JToggleButton callPath = new JToggleButton ("Call Paths");
	private JToggleButton logY = new JToggleButton ("Log Y");
	private JToggleButton scalability = new JToggleButton ("Scalability");
	private JToggleButton efficiency = new JToggleButton ("Efficiency");
	private JToggleButton constantProblem = new JToggleButton ("Weak Scaling");
	private JToggleButton horizontal = new JToggleButton ("Horizontal");
	private JLabel titleLabel = new JLabel("Chart Title:");
	private JTextField chartTitle = new MyJTextField(10);
	private JLabel seriesLabel = new JLabel("Series Name/Value:");
	private JLabel xaxisNameLabel = new JLabel("X Axis Name:");
	private JLabel yaxisNameLabel = new JLabel("Y Axis Name:");
	private JLabel xaxisValueLabel = new JLabel("X Axis Value:");
	private JLabel yaxisValueLabel = new JLabel("Y Axis Value:");
	private JLabel dimensionLabel = new JLabel("Dimension reduction:");
	private JLabel eventLabel = new JLabel("Event:");
	private JLabel metricLabel = new JLabel("Metric:");
	private JLabel valueLabel = new JLabel("Value:");
	private JLabel xmlNameLabel = new JLabel("XML Field:");
	private JLabel xmlValueLabel = new JLabel("XML Value:");

	public static ChartPane getPane () {
		if (thePane == null) {
			JPanel mainPanel = new JPanel(new BorderLayout());
			//mainPanel.setPreferredScrollableViewportSize(new Dimension(400, 400));
			thePane = new ChartPane(mainPanel);
		}
		thePane.repaint();
		return thePane;
	}


	private ChartPane (JPanel mainPanel) {
		super(mainPanel);
		this.mainPanel = mainPanel;
		this.facade = new ScriptFacade();
		JScrollBar jScrollBar = this.getVerticalScrollBar();
		jScrollBar.setUnitIncrement(35);
		// create the top options
		this.mainPanel.add(createTopMenu(), BorderLayout.NORTH);
		// create the left options
		this.mainPanel.add(createLeftMenu(), BorderLayout.WEST);
		// create the dummy chart panel
		this.mainPanel.add(createChartPanel(), BorderLayout.CENTER);
	}

	public JPanel createLeftMenu() {
		// create a new panel, with a vertical box layout
		JPanel left = new JPanel();
		left.setLayout(new BoxLayout(left, BoxLayout.Y_AXIS));

		left.add(titleLabel);
		left.add(chartTitle);

		PerfExplorerConnection server = PerfExplorerConnection.getConnection();

		List dummy = server.getChartFieldNames();
		left.add(seriesLabel);
		JComboBox series = new MyJComboBox(dummy);
		left.add(series);

		left.add(xaxisNameLabel);
		JTextField xaxisName = new MyJTextField(10);
		left.add(xaxisName);

		left.add(xaxisValueLabel);
		JComboBox xaxisValue = new MyJComboBox(dummy);
		left.add(xaxisValue);

		left.add(yaxisNameLabel);
		JTextField yaxisName = new MyJTextField(10);
		left.add(yaxisName);

		left.add(yaxisValueLabel);
		JComboBox yaxisValue = new MyJComboBox(dummy);
		left.add(yaxisValue);

		left.add(dimensionLabel);
		JComboBox dimension = new MyJComboBox(dummy);
		left.add(dimension);

		left.add(eventLabel);
		JComboBox event = new MyJComboBox(dummy);
		left.add(event);

		left.add(metricLabel);
		JComboBox metric = new MyJComboBox(dummy);
		left.add(metric);

		left.add(valueLabel);
		JComboBox value = new MyJComboBox(dummy);
		left.add(value);

		left.add(xmlNameLabel);
		JComboBox xmlName = new MyJComboBox(dummy);
		left.add(xmlName);

		left.add(xmlValueLabel);
		JComboBox xmlValue = new MyJComboBox(dummy);
		left.add(xmlValue);

		JButton apply = new JButton ("Apply");
		apply.setToolTipText("Apply changes and redraw chart");
		apply.setActionCommand(UPDATE_COMMAND);
		apply.addActionListener(this);
		left.add(apply);

		JButton reset = new JButton ("Reset");
		reset.setToolTipText("Reset changes and clear chart");
		reset.setActionCommand(UPDATE_COMMAND);
		reset.addActionListener(this);
		left.add(reset);

		return (left);
	}

	public JPanel createTopMenu() {
		JPanel top = new JPanel();
		top.setLayout(new BoxLayout(top, BoxLayout.X_AXIS));

		this.mainOnly.setToolTipText("Only select the \"main\" event (i.e. maximum inclusive)");
		this.mainOnly.setSelected(true);
		top.add(this.mainOnly);

		this.callPath.setToolTipText("Include \"call path\" events (i.e. main() => foo())");
		this.callPath.setSelected(false);
		top.add(this.callPath);

		// excl100.setToolTipText("");
		// top.add(excl100);

		this.logY.setToolTipText("Use a Logarithmic Y axis");
		this.logY.setSelected(false);
		top.add(this.logY);

		this.scalability.setToolTipText("Create a Scalability Chart");
		this.scalability.setSelected(false);
		top.add(this.scalability);

		this.efficiency.setToolTipText("Create a Relative Efficiency Chart");
		this.efficiency.setSelected(false);
		top.add(this.efficiency);

		this.constantProblem.setToolTipText("Strong Scaling problem or not (else Weak Scaling)");
		this.constantProblem.setSelected(false);
		top.add(this.constantProblem);

		this.horizontal.setToolTipText("Create a horizontal chart");
		this.horizontal.setSelected(false);
		top.add(this.horizontal);

		return (top);
	}

	public JPanel createChartPanel() {
		this.chartPanel = new JPanel(new BorderLayout());
		return (this.chartPanel);
	}

	public void updateChart () {
		// the user has selected the application, experiment, trial 
		// from the navigation tree.  Now set the other parameters.
		// We will use the ScriptFacade class to set the parameters -
		// all options should be set using the scripting interface.

		// TESTING!
	    facade.resetChartDefaults();
    	facade.setMetricName("Time");
    	facade.setChartTitle(chartTitle.getText());
    	facade.setChartSeriesName("experiment.name");
    	facade.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution");
    	facade.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Total Time (seconds)");
    	facade.setChartMainEventOnly(this.mainOnly.isSelected()?1:0);
    	facade.setChartEventNoCallPath(this.callPath.isSelected()?0:1); //reversed logic
    	facade.setChartLogYAxis(this.logY.isSelected()?1:0);
    	facade.setChartScalability(this.scalability.isSelected()?1:0);
    	facade.setChartEfficiency(this.efficiency.isSelected()?1:0);
    	facade.setChartConstantProblem(this.constantProblem.isSelected()?1:0);
    	facade.setChartHorizontal(this.horizontal.isSelected()?1:0);

		// create the Chart
		this.chartPanel.setVisible(false);
		this.chartPanel.removeAll();
		this.chartPanel.add(new ChartPanel(doGeneralChart()), BorderLayout.CENTER);
		this.repaint();
		this.chartPanel.setVisible(true);
	}

	public void actionPerformed(ActionEvent e) {
		// if action is "apply", update the chart
		updateChart();
	}

	/**
	 * This method will produce a general line chart, with 
	 * one or more series of data, with anything you want on
	 * the x-axis, and some measurement on the y-axis.
	 *
	 */
	public static JFreeChart doGeneralChart () {
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
				common.RMIGeneralChartData.CategoryDataRow baseline = rawData.getRowData(0);

				// iterate through the values
				for (int i = 0 ; i < rawData.getRows() ; i++) {
					common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
					if (!row.series.equals(baseline.series)) {
						//System.out.println(row.series);
						baseline = row;
					}
					if (model.getConstantProblem().booleanValue()) {
						//System.out.println("value: " + row.value);
						
						double ratio = baseline.categoryInteger.doubleValue() / row.categoryInteger.doubleValue();
						//System.out.println("ratio: " + ratio);
						double efficiency = baseline.value/row.value;
						//System.out.println("efficiency: " + efficiency);
        				dataset.addValue(efficiency / ratio, row.series, row.categoryInteger);
					} else {
        				dataset.addValue(baseline.value / row.value, row.series, row.categoryInteger);
					}
				}

				// create an "ideal" line.
				List keys = dataset.getColumnKeys();
				for (int i = 0 ; i < keys.size() ; i++) {
					Integer key = (Integer)keys.get(i);
        			dataset.addValue(key.doubleValue()/rawData.getMinimum(), "Ideal", key);
				}

			} else {
				// iterate through the values
				for (int i = 0 ; i < rawData.getRows() ; i++) {
					common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
        			dataset.addValue(row.value / 1000000, row.series, row.categoryInteger);
				}
			}
		} else {
			// iterate through the values
			for (int i = 0 ; i < rawData.getRows() ; i++) {
				common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
        		dataset.addValue(row.value / 1000000, row.series, row.categoryString);
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

/*
		if (rawData.getCategoryType() == Integer.class) {
			// don't mess with the domain axis
		} else {
        	CategoryAxis domainAxis = plot.getDomainAxis();
			domainAxis.setSkipCategoryLabelsToFit(true);
			domainAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_90);
		}
*/

		if (model.getChartLogYAxis()) {
        	LogarithmicAxis axis = new LogarithmicAxis(
				PerfExplorerModel.getModel().getChartYAxisLabel());
        	axis.setAutoRangeIncludesZero(true);
        	axis.setAllowNegativesFlag(true);
        	axis.setLog10TickLabelsFlag(true);
        	plot.setRangeAxis(0, axis);
 		}

		return chart;
	}

	public class MyJTextField extends javax.swing.JTextField
	{   
    	public MyJTextField() {
        	super();
    	}
    	public MyJTextField(String value, int columns) {
        	super(value, columns);
    	}
    	public MyJTextField(int columns) {
        	super(columns);
    	}
    	public MyJTextField(String value) {
        	super(value);
    	}

    	public Dimension getPreferredSize() {
        	Dimension size = super.getPreferredSize();
        	if (isMinimumSizeSet()) {
            	Dimension minSize = getMinimumSize();
            	if (minSize.width>size.width)
                	size.width = minSize.width;
        	}
        	return size;
    	}

    	public Dimension getMaximumSize() {
        	Dimension maxSize = super.getMaximumSize();
        	Dimension prefSize = getPreferredSize();
        	maxSize.height = prefSize.height;
        	return maxSize;
    	}
	}

	public class MyJComboBox extends javax.swing.JComboBox
	{   
    	public MyJComboBox(Object[] items) {
        	super(items);
    	}

    	public MyJComboBox(List items) {
        	super(items.toArray());
    	}

    	public Dimension getPreferredSize() {
        	Dimension size = super.getPreferredSize();
        	if (isMinimumSizeSet()) {
            	Dimension minSize = getMinimumSize();
            	if (minSize.width>size.width)
                	size.width = minSize.width;
        	}
        	return size;
    	}

    	public Dimension getMaximumSize() {
        	Dimension maxSize = super.getMaximumSize();
        	Dimension prefSize = getPreferredSize();
        	maxSize.height = prefSize.height;
        	return maxSize;
    	}
	}
}
