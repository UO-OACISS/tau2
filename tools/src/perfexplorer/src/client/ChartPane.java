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
import java.util.Iterator;
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
import java.util.StringTokenizer;
import java.util.NoSuchElementException;
import javax.swing.plaf.metal.*;
import javax.swing.plaf.basic.*;

public class ChartPane extends JScrollPane implements ActionListener, ImageExport {

	private static ChartPane thePane = null;
	private PerfExplorerConnection server = null;

	private JPanel mainPanel = null;
	private ScriptFacade facade = null;
	private static String UPDATE_COMMAND = "UPDATE_COMMAND";
	private JPanel chartPanel = null;

	private JToggleButton mainOnly = new JToggleButton ("Main Only");
	private JToggleButton callPath = new JToggleButton ("Call Paths");
	private JToggleButton logY = new JToggleButton ("Log Y");
	private JToggleButton scalability = new JToggleButton ("Scalability");
	private JToggleButton efficiency = new JToggleButton ("Efficiency");
	private JToggleButton constantProblem = new JToggleButton ("Strong Scaling");
	private JToggleButton horizontal = new JToggleButton ("Horizontal");
	private JToggleButton showZero = new JToggleButton ("Show Y-Axis Zero");

	private List tableColumns = null;
	private JLabel titleLabel = new JLabel("Chart Title:");
	private JTextField chartTitle = new MyJTextField(5);
	private JLabel seriesLabel = new JLabel("Series Name/Value:");
	private JComboBox series = null;
	private JLabel xaxisNameLabel = new JLabel("X Axis Name:");
	private JTextField xaxisName = new MyJTextField(5);
	private JLabel yaxisNameLabel = new JLabel("Y Axis Name:");
	private JTextField yaxisName = new MyJTextField(5);
	private JLabel xaxisValueLabel = new JLabel("X Axis Value:");
   	private JComboBox xaxisValue = null;
	private JLabel yaxisValueLabel = new JLabel("Y Axis Value:");
   	private JComboBox yaxisValue = null;
	private JLabel dimensionLabel = new JLabel("Dimension reduction:");
   	private JComboBox dimension = new MyJComboBox();
	private JLabel dimensionXLabel = new JLabel("Cutoff (0<x<100):");
	private JTextField dimensionXValue = new MyJTextField(5);
	private JLabel eventLabel = new JLabel("Event:");
   	private JComboBox event = new MyJComboBox();
	private JLabel metricLabel = new JLabel("Metric:");
   	private JComboBox metric = new MyJComboBox();
	private JLabel unitsLabel = new JLabel("Units:");
   	private JComboBox units = new MyJComboBox();
	private JLabel valueLabel = new JLabel("Value:");
   	private JComboBox value = new MyJComboBox();
	private JLabel xmlNameLabel = new JLabel("XML Field:");
   	private JComboBox xmlName = new MyJComboBox();
	//private JLabel xmlValueLabel = new JLabel("XML Value:");
   	//private JComboBox xmlValue = new MyJComboBox();
	private String[] unitOptions = {
				"microseconds", 
				"milliseconds", 
				"seconds",
				"minutes",
				"hours",
				"units",
				"thousands (x 1.0E3)",
				"millions (x 1.0E6)",
				"billions (x 1.0E9)",
				"trillions (x 1.0E12)"
				};
	

	private JButton apply = null;
	private JButton reset = null;
	private static final String INTERVAL_EVENT_NAME = "interval_event.name";
	private static final String MEAN_INCLUSIVE = "mean.inclusive";
	private static final String MEAN_EXCLUSIVE = "mean.exclusive";

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
		this.server = PerfExplorerConnection.getConnection();
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
		resetChartSettings();
	}

	private void resetChartSettings() {
		// top toggle buttons
		this.mainOnly.setSelected(true);
		this.callPath.setSelected(false);
		this.logY.setSelected(false);
		this.scalability.setSelected(false);
		this.efficiency.setSelected(false);
		this.constantProblem.setSelected(false);
		this.horizontal.setSelected(false);
		this.showZero.setSelected(true);
		// left text fields
		// left combo boxes
		this.dimension.setSelectedIndex(0);
		this.dimensionXLabel.setEnabled(false);
		this.dimensionXValue.setEnabled(false);
		this.eventLabel.setEnabled(false);
		this.event.setEnabled(false);
		this.xmlNameLabel.setEnabled(false);
		this.xmlName.setEnabled(false);
		//this.xmlValueLabel.setEnabled(false);
		//this.xmlValue.setEnabled(false);

		// series name 
		for (Iterator itr = tableColumns.iterator() ; itr.hasNext() ; ) {
			Object o = itr.next();
			String tmp = (String)o;
			if (tmp.equalsIgnoreCase("experiment.name")) {
				this.series.setSelectedItem(o);
			//} else if (tmp.equalsIgnoreCase("trial.threads_of_execution")) {
				//this.xaxisValue.setSelectedItem(o);
			} else if (tmp.equalsIgnoreCase("trial.xml_metadata")) {
				this.xaxisValue.setSelectedItem(o);
			}
		}
		this.yaxisValue.setSelectedIndex(0);
		refreshDynamicControls(true, true, false);
	}

	public void refreshDynamicControls(boolean getMetrics, boolean getEvents, boolean getXML) {
		PerfExplorerModel theModel = PerfExplorerModel.getModel();
		Object selection = theModel.getCurrentSelection();
		String oldMetric = "";
		String oldEvent = "";
		String oldXML = "";
		Object obj = null;
		if (getMetrics) {
			obj = this.metric.getSelectedItem();
			if (obj != null)
				oldMetric = (String)obj;
			this.metric.removeAllItems();
		}
		if (getEvents) {
			obj = this.event.getSelectedItem();
			if (obj != null)
				oldEvent = (String)obj;
			this.event.removeAllItems();
		}
		if (getXML) {
			obj = this.xmlName.getSelectedItem();
			if (obj != null)
				oldXML = (String)obj;
			this.xmlName.removeAllItems();
		}
		if ((selection instanceof Application) ||
		    (selection instanceof Experiment) ||
		    (selection instanceof Trial)) {
			if (getMetrics) {
				List metrics = server.getPotentialMetrics(theModel);
				//this.metric.setSelectedIndex(0);
				boolean gotTime = false;
				for (Iterator itr = metrics.iterator() ; itr.hasNext() ; ) {
					String next = (String)itr.next();
					if (next.toUpperCase().indexOf("TIME") > 0) {
						gotTime = true;
					}
					this.metric.addItem(next);
					if (oldMetric.equals(next))
						this.metric.setSelectedItem(next);
				}
				if (!gotTime) {
					this.metric.addItem("TIME");
					if (oldMetric.equals("TIME"))
						this.metric.setSelectedItem("TIME");
				}
			} 
			if (getEvents && !this.mainOnly.isSelected()) {
				Object obj2 = series.getSelectedItem();
				String tmp = (String)obj2;
				if (tmp.equalsIgnoreCase("interval_event.group_name")) {
					List events = server.getPotentialGroups(theModel);
					this.event.addItem("All Groups");
					this.eventLabel.setText("Group:");
					this.event.setSelectedIndex(0);
					for (Iterator itr = events.iterator() ; itr.hasNext() ; ) {
						String next = (String)itr.next();
						this.event.addItem(next);
						if (oldEvent.equals(next))
							this.event.setSelectedItem(next);
					}
				} else {
					series.setSelectedItem(INTERVAL_EVENT_NAME);
					yaxisValue.setSelectedItem(MEAN_EXCLUSIVE);
					List events = server.getPotentialEvents(theModel);
					this.event.addItem("All Events");
					this.eventLabel.setText("Event:");
					this.event.setSelectedIndex(0);
					for (Iterator itr = events.iterator() ; itr.hasNext() ; ) {
						String next = (String)itr.next();
						this.event.addItem(next);
						if (oldEvent.equals(next))
							this.event.setSelectedItem(next);
					}
				}
			} 
			if (getXML) {
				Object obj2 = series.getSelectedItem();
				String tmp = (String)obj2;
				Object obj3 = xaxisValue.getSelectedItem();
				String tmp2 = (String)obj3;
				if (tmp.equalsIgnoreCase("trial.xml_metadata") ||
					tmp2.equalsIgnoreCase("trial.xml_metadata")) {
					List xmlEvents = server.getXMLFields(theModel);
					for (Iterator itr = xmlEvents.iterator(); itr.hasNext();) {
						String next = (String)itr.next();
						this.xmlName.addItem(next);
						if (oldXML.equals("") && next.equalsIgnoreCase("UTC Time"))
							this.xmlName.setSelectedItem(next);
						else if (oldXML.equals(next))
							this.xmlName.setSelectedItem(next);
					}
					//this.xmlName.setSelectedIndex(0);
				}
			} 
		}
	}

	private JPanel createTopMenu() {
		JPanel top = new JPanel();
		top.setLayout(new BoxLayout(top, BoxLayout.X_AXIS));

		this.mainOnly.setToolTipText("Only select the \"main\" event (i.e. maximum inclusive)");
		this.mainOnly.addActionListener(this);
		top.add(this.mainOnly);

		this.callPath.setToolTipText("Include \"call path\" events (i.e. main() => foo())");
		this.callPath.addActionListener(this);
		top.add(this.callPath);

		// excl100.setToolTipText("");
		// top.add(excl100);

		this.logY.setToolTipText("Use a Logarithmic Y axis");
		this.logY.addActionListener(this);
		top.add(this.logY);

		this.scalability.setToolTipText("Create a Scalability Chart");
		this.scalability.addActionListener(this);
		top.add(this.scalability);

		this.efficiency.setToolTipText("Create a Relative Efficiency Chart");
		this.efficiency.addActionListener(this);
		top.add(this.efficiency);

		this.constantProblem.setToolTipText("Scaling type (Strong Scaling or Weak Scaling)");
		this.constantProblem.addActionListener(this);
		top.add(this.constantProblem);

		this.horizontal.setToolTipText("Create a horizontal chart");
		this.horizontal.addActionListener(this);
		top.add(this.horizontal);

		this.showZero.setToolTipText("Include zero value in y-axis range");
		this.showZero.addActionListener(this);
		top.add(this.showZero);

		return (top);
	}

	private JPanel createLeftMenu() {
		// create a new panel, with a vertical box layout
		JPanel left = new JPanel();
		left.setLayout(new BoxLayout(left, BoxLayout.Y_AXIS));
		//left.setLayout(new FlowLayout(FlowLayout.LEFT));
		//left.setLayout(new GridLayout(0,1,0,0));

		// chart title
		left.add(titleLabel);
		this.chartTitle.addActionListener(this);
		left.add(chartTitle);

		this.tableColumns = server.getChartFieldNames();

		// series name
		left.add(seriesLabel);
		series = new MyJComboBox(tableColumns);
		series.addItem(INTERVAL_EVENT_NAME);
		series.addItem("interval_event.group_name");
		series.addActionListener(this);
		left.add(series);

		// x axis value
		left.add(xaxisValueLabel);
		xaxisValue = new MyJComboBox(tableColumns);
		xaxisValue.addActionListener(this);
		this.xaxisValue.addActionListener(this);
		left.add(xaxisValue);
		left.add(xaxisNameLabel);
		this.xaxisName.addActionListener(this);
		left.add(xaxisName);

		// y axis value
		left.add(yaxisValueLabel);
		String[] valueOptions = {
					MEAN_INCLUSIVE,
					MEAN_EXCLUSIVE,
					"mean.inclusive_percentage", 
					"mean.exclusive_percentage", 
					"mean.call", 
					"mean.subroutines", 
					"mean.inclusive_per_call", 
					"mean.sum_exclusive_squared",
					"total.inclusive", 
					"total.exclusive", 
					"total.inclusive_percentage", 
					"total.exclusive_percentage", 
					"total.call", 
					"total.subroutines", 
					"total.inclusive_per_call", 
					"total.sum_exclusive_squared",
		"atomic_location_profile.sample_count",
		"atomic_location_profile.maximum_value",
		"atomic_location_profile.minimum_value",
		"atomic_location_profile.mean_value",
		"atomic_location_profile.standard_deviation"
					};
		yaxisValue = new MyJComboBox(valueOptions);
		this.yaxisValue.addActionListener(this);
		left.add(yaxisValue);
		left.add(yaxisNameLabel);
		this.yaxisName.addActionListener(this);
		left.add(yaxisName);

		// dimension reduction
		left.add(dimensionLabel);
		Object[] dimensionOptions = TransformationType.getDimensionReductions();
		dimension = new MyJComboBox(dimensionOptions);
		dimension.addActionListener(this);
		this.dimension.addActionListener(this);
		left.add(dimension);
		left.add(dimensionXLabel);
		this.dimensionXValue.addActionListener(this);
		left.add(dimensionXValue);

		// metric of interest
		left.add(metricLabel);
		metric = new MyJComboBox();
		this.metric.addActionListener(this);
		left.add(metric);

		// units of interest
		left.add(unitsLabel);
		units = new MyJComboBox(unitOptions);
		this.units.addActionListener(this);
		this.units.setSelectedIndex(2);  // default to seconds
		left.add(this.units);

		// event of interest
		left.add(eventLabel);
		event = new MyJComboBox();
		this.event.addActionListener(this);
		left.add(event);

		// XML metadata
		left.add(xmlNameLabel);
		xmlName = new MyJComboBox();
		this.xmlName.addActionListener(this);
		left.add(xmlName);
		//left.add(xmlValueLabel);
		//xmlValue = new MyJComboBox();
		//left.add(xmlValue);

		// apply button
		apply = new JButton ("Apply");
		apply.setToolTipText("Apply changes and redraw chart");
		apply.addActionListener(this);
		left.add(apply);

		// reset button
		reset = new JButton ("Reset");
		reset.setToolTipText("Change back to default settings");
		reset.addActionListener(this);
		left.add(reset);

		return (left);
	}

	private JPanel createChartPanel() {
		this.chartPanel = new JPanel(new BorderLayout());
		return (this.chartPanel);
	}

	public JPanel getChartPanel() {
		return (this.chartPanel);
	}

	private void updateChart () {
		// the user has selected the application, experiment, trial 
		// from the navigation tree.  Now set the other parameters.
		// We will use the ScriptFacade class to set the parameters -
		// all options should be set using the scripting interface.
		facade.resetChartDefaults();

		// title
		String title = chartTitle.getText();
		if (title.length() == 0) { 
			title = (String)metric.getSelectedItem();
		}
    	facade.setChartTitle(title);

		// series name
		Object obj = series.getSelectedItem();
		String tmp = (String)obj;
		if (tmp.equalsIgnoreCase("trial.threads_of_execution")) {
			tmp = "trial.node_count * trial.contexts_per_node * trial.threads_per_context";
		} else if (tmp.equalsIgnoreCase("trial.XML_METADATA")) {
			tmp = "temp_xml_metadata.metadata_value";
   			Object obj2 = xaxisValue.getSelectedItem();
			String tmp2 = (String)obj2;
		    facade.setChartMetadataFieldName(tmp2);
		}
    	facade.setChartSeriesName(tmp);

		// x axis
   		obj = xaxisValue.getSelectedItem();
		tmp = (String)obj;
		String tmp2 = null;
		if (tmp.equalsIgnoreCase("trial.threads_of_execution")) {
			tmp = "trial.node_count * trial.contexts_per_node * trial.threads_per_context";
		} else if (tmp.equalsIgnoreCase("trial.XML_METADATA")) {
			tmp = "temp_xml_metadata.metadata_value";
   			Object obj2 = xmlName.getSelectedItem();
			tmp2 = (String)obj2;
		    facade.setChartMetadataFieldName(tmp2);
		    facade.setChartMetadataFieldValue(null);
		}
		String label = xaxisName.getText();
		if (label == null || label.length() == 0) {
			if (tmp.equalsIgnoreCase("temp_xml_metadata.metadata_value")) {
				label = tmp2;
			} else {
				label = tmp;
			}
		}
 		facade.setChartXAxisName(tmp, label);

		// y axis
    	obj = yaxisValue.getSelectedItem();
		tmp = (String)obj;
		tmp = tmp.replaceAll("mean", "interval_mean_summary");
		tmp = tmp.replaceAll("total", "interval_total_summary");
		String operation = "avg";
    	if (!this.mainOnly.isSelected()) {
			obj = this.series.getSelectedItem();
			tmp2 = (String)obj;
			if (tmp2.equalsIgnoreCase("interval_event.group_name")) {
				operation = "sum";
			}
		}
		tmp = operation + "(" + tmp + ")";
		label = yaxisName.getText();
		if (label == null || label.length() == 0) {
			// build something intelligible
			//label = tmp;
			if (tmp.indexOf("mean") >= 0) {
				label = "Mean " + (String)this.metric.getSelectedItem();
			} else if (tmp.indexOf("total") >= 0) {
				label = "Total " + (String)this.metric.getSelectedItem();
			} else {
				label = (String)this.metric.getSelectedItem();
			}
		}
		label += " - "  + (String)this.units.getSelectedItem();
		facade.setChartYAxisName(tmp, label);

		// metric name
		obj = metric.getSelectedItem();
		tmp = (String)obj;
		if (tmp.equals("TIME"))
    		facade.setMetricName("%TIME%");
		else
    		facade.setMetricName(tmp);

		// units name
		obj = units.getSelectedItem();
		tmp = (String)obj;
    	facade.setChartUnits(tmp);

		// dimension reduction
		obj = dimension.getSelectedItem();
		TransformationType type = (TransformationType)obj;
		if (type == TransformationType.OVER_X_PERCENT) {
			label = dimensionXValue.getText();
			if (label == null || label.length() == 0) {
    			facade.setDimensionReduction(TransformationType.NONE, null);
			} else {
    			facade.setDimensionReduction(TransformationType.OVER_X_PERCENT, label);
			}
		} else {
    		facade.setDimensionReduction(TransformationType.NONE, null);
		}

		// other options
    	facade.setChartMainEventOnly(this.mainOnly.isSelected()?1:0);
    	if (!this.mainOnly.isSelected()) {
			obj = this.series.getSelectedItem();
			tmp = (String)obj;
			facade.setEventName(null);
			facade.setGroupName(null);
			if (tmp.equalsIgnoreCase(INTERVAL_EVENT_NAME)) {
				obj = this.event.getSelectedItem();
				tmp = (String)obj;
				if (!tmp.equals("All Events")) {
					facade.setEventName(tmp);
				} else {
					facade.setEventName(null);
				}
			} else if (tmp.equalsIgnoreCase("interval_event.group_name")) {
				obj = this.event.getSelectedItem();
				tmp = (String)obj;
				if (!tmp.equals("All Groups")) {
					facade.setGroupName(tmp);
				} else {
					facade.setGroupName(null);
				}
			}
		}

    	facade.setChartEventNoCallPath(this.callPath.isSelected()?0:1); //reversed logic
    	facade.setChartLogYAxis(this.logY.isSelected()?1:0);
    	facade.setChartScalability(this.scalability.isSelected()?1:0);
    	facade.setChartEfficiency(this.efficiency.isSelected()?1:0);
    	facade.setChartConstantProblem(this.constantProblem.isSelected()?1:0);
    	facade.setChartHorizontal(this.horizontal.isSelected()?1:0);
    	facade.setShowZero(this.showZero.isSelected()?1:0);

		// create the Chart
		this.chartPanel.setVisible(false);
		this.chartPanel.removeAll();
		this.chartPanel.add(new ChartPanel(doGeneralChart()), BorderLayout.CENTER);
		this.repaint();
		this.chartPanel.setVisible(true);
	}

	public void actionPerformed(ActionEvent e) {
		// if action is "apply", update the chart
		Object source = e.getSource();
		if (source == apply) {
			PerfExplorerModel theModel = PerfExplorerModel.getModel();
			Object selection = theModel.getCurrentSelection();
			if ((selection instanceof Application) ||
		    	(selection instanceof Experiment) ||
		    	(selection instanceof Trial)) {
				updateChart();
			} else {
				// tell the user they need to select something
				 JOptionPane.showMessageDialog(
				 	PerfExplorerClient.getMainFrame(), 
				 	"Please select one or more Applications, Experiments or Trials.",
					"Selection Error", JOptionPane.ERROR_MESSAGE);
			}
		} else if (source == reset) {
			resetChartSettings();
		} else if (source == scalability) {
			if (scalability.isSelected()) {
				efficiency.setSelected(false);
			}
		} else if (source == efficiency) {
			if (efficiency.isSelected()) {
				scalability.setSelected(false);
			}
		} else if (source == constantProblem) {
			if (constantProblem.isSelected()) {
				constantProblem.setText("Weak Scaling");
			} else {
				constantProblem.setText("Strong Scaling");
			}
		} else if (source == mainOnly) {
			if (mainOnly.isSelected()) {
				this.eventLabel.setEnabled(false);
				this.event.setEnabled(false);
			} else {
				this.eventLabel.setEnabled(true);
				this.event.setEnabled(true);
				refreshDynamicControls(false, true, false);
			}
		} else if (source == dimension) {
			if (dimension.getSelectedIndex() == 0) {
				this.dimensionXLabel.setEnabled(false);
				this.dimensionXValue.setEnabled(false);
			} else {
				this.dimensionXLabel.setEnabled(true);
				this.dimensionXValue.setEnabled(true);
			}
		} else if ((source == series) || 
				   (source == xaxisValue)) {
			Object obj = series.getSelectedItem();
			String tmp = (String)obj;
			Object obj2 = xaxisValue.getSelectedItem();
			String tmp2 = (String)obj2;
			if (tmp.equalsIgnoreCase("trial.xml_metadata") ||
				tmp2.equalsIgnoreCase("trial.xml_metadata")) {
				this.xmlNameLabel.setEnabled(true);
				this.xmlName.setEnabled(true);
				refreshDynamicControls(false, false, true);
			} else {
				this.xmlNameLabel.setEnabled(false);
				this.xmlName.setEnabled(false);
			}
			if (tmp.equalsIgnoreCase(INTERVAL_EVENT_NAME) ||
				tmp.equalsIgnoreCase("interval_event.group_name")) {
				refreshDynamicControls(false, true, false);
			}
		}
		drawChart();
	}

	public void drawChart() {
		// draw the chart!
		/*
		PerfExplorerModel theModel = PerfExplorerModel.getModel();
		Object selection = theModel.getCurrentSelection();
		if ((selection instanceof Application) ||
	    	(selection instanceof Experiment) ||
	    	(selection instanceof Trial)) {
			updateChart();
		}
		*/
	}

	/**
	 * This method will produce a general line chart, with 
	 * one or more series of data, with anything you want on
	 * the x-axis, and some measurement on the y-axis.
	 *
	 */
	private JFreeChart doGeneralChart () {
		// get the data
		PerfExplorerModel model = PerfExplorerModel.getModel();
		RMIGeneralChartData rawData = server.requestGeneralChartData(
			model, ChartDataType.PARAMETRIC_STUDY_DATA);

        PECategoryDataset dataset = new PECategoryDataset();

		String units = model.getChartUnits();
		if (units == null) 
			units = new String("microseconds");

		double conversion = 1.0;
		if (units.equals("milliseconds")) {
			conversion = 1000.0;
		} else if (units.equals("seconds")) {
			conversion = 1000000.0;
		} else if (units.equals("minutes")) {
			conversion = 60000000.0;
		} else if (units.equals("hours")) {
			conversion = 3600000000.0;
		} else if (units.equals("units")) {
			conversion = 1.0;
		} else if (units.equals("thousands (x 1.0E3)")) {
			conversion = 1000.0;
		} else if (units.equals("millions (x 1.0E6)")) {
			conversion = 1000000.0;
		} else if (units.equals("billions (x 1.0E9)")) {
			conversion = 1000000000.0;
		} else if (units.equals("billions (x 1.0E12)")) {
			conversion = 1000000000000.0;
		} 

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
					if (!shortName(row.series).equals(shortName(baseline.series))) {
						baseline = row;
					}
					if (model.getConstantProblem().booleanValue()) {
						
						double ratio = baseline.categoryInteger.doubleValue() / 
							row.categoryInteger.doubleValue();
						double efficiency = baseline.value/row.value;
        				dataset.addValue(efficiency / ratio, shortName(row.series), row.categoryInteger);
					} else {
        				dataset.addValue(baseline.value / row.value, shortName(row.series), row.categoryInteger);
					}
				}

				// create an "ideal" line.
				List keys = dataset.getColumnKeys();
				for (int i = 0 ; i < keys.size() ; i++) {
					Integer key = (Integer)keys.get(i);
        			dataset.addValue(key.doubleValue()/rawData.getMinimum(), "Ideal", key);
				}

			} else if (model.getChartEfficiency()) {

				// create an "ideal" line.
				// If we have categorical data, this method won't work...
				/*
        		dataset.addValue(1.0, "Ideal", new Integer(rawData.getMinimum()));
        		dataset.addValue(rawData.getMaximum()/rawData.getMinimum(), "Ideal", 
					new Integer(rawData.getMaximum()));
				*/

				// get the baseline values
				common.RMIGeneralChartData.CategoryDataRow baseline = rawData.getRowData(0);

				// iterate through the values
				for (int i = 0 ; i < rawData.getRows() ; i++) {
					common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
					if (!shortName(row.series).equals(shortName(baseline.series))) {
						baseline = row;
					}
					if (model.getConstantProblem().booleanValue()) {
						/*
						double ratio = baseline.categoryInteger.doubleValue() / 
							row.categoryInteger.doubleValue();
						double efficiency = baseline.value/row.value;
        				dataset.addValue(efficiency / ratio, row.series, row.categoryInteger);
						*/
        				dataset.addValue(baseline.value / row.value, shortName(row.series), row.categoryInteger);
					} else {
        				dataset.addValue((baseline.value * baseline.categoryInteger.doubleValue())/ (row.value * row.categoryInteger.doubleValue()), shortName(row.series), row.categoryInteger);
					}
				}

				// create an "ideal" line.
				List keys = dataset.getColumnKeys();
				for (int i = 0 ; i < keys.size() ; i++) {
					Integer key = (Integer)keys.get(i);
        			dataset.addValue(1.0, "Ideal", key);
				}

			} else {
				// iterate through the values
				for (int i = 0 ; i < rawData.getRows() ; i++) {
					common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
        			dataset.addValue(row.value / conversion, shortName(row.series), row.categoryInteger);
				}
			}
		} else {
			// iterate through the values
			for (int i = 0 ; i < rawData.getRows() ; i++) {
				common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
        		dataset.addValue(row.value / conversion, shortName(row.series), row.categoryString);
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
        //rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
    	if (model.getShowZero()) {
			rangeAxis.setAutoRangeIncludesZero(true);
    	} else {
			rangeAxis.setAutoRangeIncludesZero(false);
    	}

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
    		if (model.getShowZero()) {
    			axis.setAutoRangeIncludesZero(true);
    		} else {
    			axis.setAutoRangeIncludesZero(false);
    		}
        	axis.setAllowNegativesFlag(true);
        	axis.setLog10TickLabelsFlag(true);
        	plot.setRangeAxis(0, axis);
 		}

		return chart;
	}

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        return this.getChartPanel().getSize();
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        this.getChartPanel().setDoubleBuffered(false);
        this.getChartPanel().paintAll(g2D);
        this.getChartPanel().setDoubleBuffered(true);
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
		return shorter;
	}

	private class MyJTextField extends javax.swing.JTextField
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

	private class MyJComboBox extends javax.swing.JComboBox
	{   
    	public MyJComboBox(Object[] items) {
        	super(items);
			setPrototypeDisplayValue("WWWWW");
    	}

    	public MyJComboBox() {
        	super();
			setPrototypeDisplayValue("WWWWW");
    	}

    	public MyJComboBox(List items) {
        	super(items.toArray());
			setPrototypeDisplayValue("WWWWW");
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

	/**
	 * from http://www.codeguru.com/java/articles/163.shtml
	 */
	private class SteppedComboBoxUI extends MetalComboBoxUI {
  		protected ComboPopup createPopup() {
    		BasicComboPopup popup = new BasicComboPopup( comboBox ) {
      			public void show() {
        			Dimension popupSize = ((SteppedComboBox)comboBox).getPopupSize();
        			popupSize.setSize( popupSize.width,
          				getPopupHeightForRowCount( comboBox.getMaximumRowCount() ) );
        			Rectangle popupBounds = computePopupBounds( 0,
          				comboBox.getBounds().height, popupSize.width, popupSize.height);
        			scroller.setMaximumSize( popupBounds.getSize() );
        			scroller.setPreferredSize( popupBounds.getSize() );
        			scroller.setMinimumSize( popupBounds.getSize() );
        			list.invalidate();            
        			int selectedIndex = comboBox.getSelectedIndex();
        			if ( selectedIndex == -1 ) {
          				list.clearSelection();
        			} else {
          				list.setSelectedIndex( selectedIndex );
        			}            
        			list.ensureIndexIsVisible( list.getSelectedIndex() );
        			setLightWeightPopupEnabled( comboBox.isLightWeightPopupEnabled() );
        			show( comboBox, popupBounds.x, popupBounds.y );
      			}
    		};
    		popup.getAccessibleContext().setAccessibleParent(comboBox);
    		return popup;
  		}
	}
 
	/**
	 * from http://www.codeguru.com/java/articles/163.shtml
	 */
	private class SteppedComboBox extends JComboBox {
  		protected int popupWidth;
  	
  		public SteppedComboBox() {
    		super();
    		setUI(new SteppedComboBoxUI());
    		popupWidth = 0;
    		Dimension d = getPreferredSize();
    		setPreferredSize(new Dimension(50, d.height));
    		setPopupWidth(d.width);
  		}

  		public SteppedComboBox(final Object[] items) {
    		super(items);
    		setUI(new SteppedComboBoxUI());
    		popupWidth = 0;
    		Dimension d = getPreferredSize();
    		setPreferredSize(new Dimension(50, d.height));
    		setPopupWidth(d.width);
  		}
  
  		public SteppedComboBox(List items) {
    		super(items.toArray());
    		setUI(new SteppedComboBoxUI());
    		popupWidth = 0;
    		Dimension d = getPreferredSize();
    		setPreferredSize(new Dimension(50, d.height));
    		setPopupWidth(d.width);
  		}
  
  		public void setPopupWidth(int width) {
    		popupWidth = width;
  		}
  
  		public Dimension getPopupSize() {
    		Dimension size = getSize();
    		if (popupWidth < 1) popupWidth = size.width;
    		return new Dimension(popupWidth, size.height);
  		}

    	public Dimension getMaximumSize() {
        	Dimension maxSize = super.getMaximumSize();
        	Dimension prefSize = getPreferredSize();
        	maxSize.height = prefSize.height;
        	return maxSize;
    	}
	}

	private class ChartPanelException extends Exception {
		ChartPanelException (String message) {
			super(message);
		}
	}
}
