package edu.uoregon.tau.perfexplorer.client;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.border.TitledBorder;
import javax.swing.plaf.basic.BasicComboPopup;
import javax.swing.plaf.basic.ComboPopup;
import javax.swing.plaf.metal.MetalComboBoxUI;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.general.SeriesException;
import org.jfree.data.xy.DefaultTableXYDataset;
import org.jfree.data.xy.XYSeries;
import org.python.antlr.base.mod;

import edu.uoregon.tau.common.FileFilter;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.common.ChartDataType;
import edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData;
import edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData.CategoryDataRow;
import edu.uoregon.tau.perfexplorer.common.TransformationType;

public class ChartPane extends JScrollPane implements ActionListener {

	/**
	 * 
	 */

	private static final String IDEAL="Ideal";

	private static final long serialVersionUID = -8971827392560223964L;
	private static ChartPane thePane = null;
	private PerfExplorerConnection server = null;

	private JPanel mainPanel = null;
	private ScriptFacade facade = null;

	private JCheckBox mainOnly = new JCheckBox  ("Main Only");
	private JCheckBox  callPath = new JCheckBox  ("Call Paths");
	private JCheckBox  logY = new JCheckBox ("Log Y");

	JRadioButton valueRB = new JRadioButton("Value Chart");
	JRadioButton scalaRB = new JRadioButton("Scalability Chart");
	JRadioButton efficRB = new JRadioButton("Efficency Chart");
	ButtonGroup chartType = new ButtonGroup();

	JRadioButton strongScaling = new JRadioButton("Strong Scaling");
	JRadioButton weakScaling = new JRadioButton("Weak Scaling");
	ButtonGroup scalingType = new ButtonGroup();

	private JCheckBox  horizontal = new JCheckBox  ("Horizontal");
	private JCheckBox  showZero = new JCheckBox  ("Show Y-Axis Zero");

	private List<String> tableColumns = null;
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
	private JLabel yaxisStatLabel = new JLabel("Y Axis Statistic:");
	private JComboBox yaxisStat = null;
	private JLabel yaxisValueLabel = new JLabel("Y Axis Value:");
	private JComboBox yaxisValue = null;
	private JLabel dimensionLabel = new JLabel("Dimension Reduction Type:");
	private JComboBox dimension = new MyJComboBox();
	private JLabel dimensionXLabel = new JLabel("Cutoff (0<x<100):");
	private JTextField dimensionXValue = new MyJTextField(5);
	private JLabel eventLabel = new JLabel("Interval Event:");
	//private DefaultListModel eventModel = new DefaultListModel();
	private JList event = null;//new JList();
	private JScrollPane eventScrollPane = null;
	private JLabel metricLabel = new JLabel("Metric:");
	private JComboBox metric = new MyJComboBox();
	private JLabel unitsLabel = new JLabel("Units:");
	private JComboBox units = new MyJComboBox();
	private JLabel seriesXmlNameLabel = new JLabel("Series XML Field:");
	private JComboBox seriesXmlName = new MyJComboBox();

	private JLabel xmlNameLabel = new JLabel("X Axis XML Field:");
	private JComboBox xmlName = new MyJComboBox();

	private JCheckBox angleXLabels= new JCheckBox("Angle X Axis Labels");
	private JCheckBox alwaysCategory= new JCheckBox("Categorical X Axis");
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
	private JCheckBox exportdata = null;
	private JButton reset = null;

	private JFileChooser fc = new JFileChooser(System.getProperty("user.dir"));

	private static final String ATOMIC_EVENT_ALL = "All Atomic Events";
	private static final String INTERVAL_EVENT_ALL = "All Events";
	private static final String INTERVAL_EVENT_GROUP_ALL = "All Groups";

	private static final String ATOMIC_EVENT_NAME = "atomic_event.name";
	private static final String INTERVAL_EVENT_NAME = "interval_event.name";
	private static final String INTERVAL_EVENT_GROUP_NAME = "interval_event.group_name";

	private static final String EXPERIMENT_NAME = "experiment.name";
	private static final String EXPERIMENT_ID = "experiment.id";

	//private static final String MEAN_INCLUSIVE = "mean.inclusive";
	//private static final String MEAN_EXCLUSIVE = "mean.exclusive";
	//private static final String ATOMIC_MEAN_VALUE = "atomic.mean_value";

	private static final String TOTAL="total";
	private static final String MEAN="mean";
	private static final String MAX="max";
	private static final String MIN="min";
	private static final String AVG="avg";
	private static final String ATOMIC="atomic";
	private static final String INCLUSIVE="inclusive";
	private static final String EXCLUSIVE="exclusive";

	public static ChartPane getPane () {
		if (thePane == null) {
			JPanel mainPanel = new JPanel(new GridLayout(1,3,10,5));
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
		this.tableColumns = server.getChartFieldNames();
		// create the left options
		JPanel panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
		panel.add(createChartTypeMenu());

		panel.add(Box.createVerticalStrut(10));
		panel.add(createDataMenu());

		panel.add(Box.createVerticalStrut(10));
		panel.add(createButtonMenu());

		this.mainPanel.add(panel, BorderLayout.WEST);

		// create the middle options
		panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
		panel.add(createXAxisMenu());

		panel.add(Box.createVerticalStrut(10));
		panel.add(createYAxisMenu());
		this.mainPanel.add(panel, BorderLayout.CENTER);

		// create the right options
		panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
		panel.add(createSeriesMenu());

		panel.add(Box.createVerticalStrut(10));
		panel.add(createDimensionReductionMenu());

		this.mainPanel.add(panel, BorderLayout.SOUTH);

		resetChartSettings();
	}

	private JPanel createDimensionReductionMenu() {
		// create a new panel, with a vertical box layout
		JPanel panel = new JPanel();
		TitledBorder tb = BorderFactory.createTitledBorder("Dimension Reduction");
		panel.setBorder(tb);
		panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

		panel.add(Box.createVerticalStrut(10));
		// dimension reduction
		panel.add(dimensionLabel);
		Object[] dimensionOptions = TransformationType.getDimensionReductions();
		dimension = new MyJComboBox(dimensionOptions);
		dimension.addActionListener(this);
		this.dimension.addActionListener(this);
		panel.add(dimension);

		panel.add(Box.createVerticalStrut(10));
		panel.add(dimensionXLabel);
		this.dimensionXValue.addActionListener(this);
		panel.add(dimensionXValue);

		return (panel);
	}


	private JPanel createDataMenu() {
		// create a new panel, with a vertical box layout
		JPanel panel = new JPanel();
		TitledBorder tb = BorderFactory.createTitledBorder("Chart Data");
		panel.setBorder(tb);
		panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

		panel.add(Box.createVerticalStrut(10));
		// chart title
		panel.add(titleLabel);
		this.chartTitle.addActionListener(this);
		panel.add(chartTitle);

		panel.add(Box.createVerticalStrut(10));
		// metric of interest
		panel.add(metricLabel);
		metric = new MyJComboBox();
		this.metric.addActionListener(this);
		panel.add(metric);

		panel.add(Box.createVerticalStrut(10));
		// units of interest
		panel.add(unitsLabel);
		units = new MyJComboBox(unitOptions);
		this.units.addActionListener(this);
		this.units.setSelectedIndex(2);  // default to seconds
		panel.add(this.units);

		panel.add(Box.createVerticalStrut(10));
		// event of interest
		panel.add(eventLabel);
		event = new JList();

		eventScrollPane = new JScrollPane(event);
		JPanel eventPanel = new JPanel();
		eventPanel.setLayout(new BorderLayout());
		eventPanel.add(eventScrollPane);
		//scrollPane.setMaximumSize(metric.getPreferredSize());
		//event.setSize(400, 40);
		//TODO: Does this need an action?
		//this.event.addActionListener(this);
		panel.add(eventPanel);

		panel.add(Box.createVerticalStrut(10));
		this.mainOnly.setToolTipText("Only select the \"main\" event (i.e. maximum inclusive)");
		this.mainOnly.addActionListener(this);
		panel.add(this.mainOnly);

		panel.add(Box.createVerticalStrut(10));
		this.callPath.setToolTipText("Include \"call path\" events (i.e. main() => foo())");
		this.callPath.addActionListener(this);
		panel.add(this.callPath);

		return (panel);
	}

	private JPanel createXAxisMenu() {
		// create a new panel, with a vertical box layout
		JPanel panel = new JPanel();
		TitledBorder tb = BorderFactory.createTitledBorder("X Axis");
		panel.setBorder(tb);
		panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

		panel.add(Box.createVerticalStrut(10));
		// x axis name
		panel.add(xaxisNameLabel);
		this.xaxisName.addActionListener(this);
		panel.add(xaxisName);

		panel.add(Box.createVerticalStrut(10));
		// x axis value
		panel.add(xaxisValueLabel);
		xaxisValue = new MyJComboBox(tableColumns.toArray());
		xaxisValue.addActionListener(this);
		panel.add(xaxisValue);

		panel.add(Box.createVerticalStrut(10));
		// XML metadata
		panel.add(xmlNameLabel);
		xmlName = new MyJComboBox();
		this.xmlName.addActionListener(this);
		panel.add(xmlName);

		panel.add(Box.createVerticalStrut(10));
		panel.add(this.angleXLabels);

		panel.add(Box.createVerticalStrut(10));
		panel.add(this.alwaysCategory);

		return (panel);
	}

	private JPanel createYAxisMenu() {
		JPanel panel = new JPanel();
		TitledBorder tb = BorderFactory.createTitledBorder("Y Axis");
		panel.setBorder(tb);
		panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

		panel.add(Box.createVerticalStrut(10));
		// y axis name
		panel.add(yaxisNameLabel);
		this.yaxisName.addActionListener(this);
		panel.add(yaxisName);

		panel.add(Box.createVerticalStrut(10));

		//y axis stat
		panel.add(yaxisStatLabel);
		String[] valueOptions = {"tmp"};// this will get reset in a few lines...
		yaxisStat = new MyJComboBox(valueOptions);
		this.yaxisStat.addActionListener(this);
		//resetYAxisValues(true);  // ...right here!



		// y axis value

		yaxisValue = new MyJComboBox(valueOptions);
		this.yaxisValue.addActionListener(this);
		resetYAxisValues(true);  // ...right here!

		panel.add(yaxisStat);
		panel.add(yaxisValueLabel);
		panel.add(yaxisValue);

		panel.add(Box.createVerticalStrut(10));
		// log y
		this.logY.setToolTipText("Use a Logarithmic Y axis");
		this.logY.addActionListener(this);
		panel.add(this.logY);

		panel.add(Box.createVerticalStrut(10));
		// show 0
		this.showZero.setToolTipText("Include zero value in y-axis range");
		this.showZero.addActionListener(this);
		panel.add(this.showZero);

		return (panel);
	}

	private JPanel createSeriesMenu() {
		JPanel panel = new JPanel();
		TitledBorder tb = BorderFactory.createTitledBorder("Series");
		panel.setBorder(tb);
		panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

		panel.add(Box.createVerticalStrut(10));
		// series name
		panel.add(seriesLabel);
		series = new MyJComboBox(tableColumns.toArray());
		series.addItem(INTERVAL_EVENT_NAME);
		series.addItem(INTERVAL_EVENT_GROUP_NAME);
		series.addItem(ATOMIC_EVENT_NAME);
		series.addActionListener(this);
		panel.add(series);

		panel.add(Box.createVerticalStrut(10));
		// series xml
		panel.add(seriesXmlNameLabel);
		seriesXmlName = new MyJComboBox();
		this.seriesXmlName.addActionListener(this);
		panel.add(seriesXmlName);


		return (panel);
	}


	private JPanel createChartTypeMenu() {
		JPanel panel = new JPanel();

		TitledBorder tb = BorderFactory.createTitledBorder("Chart Type");
		panel.setBorder(tb);
		panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
		//This is necessary to make this panel fill the column, but I'm not sure why...
		panel.setAlignmentX(CENTER_ALIGNMENT);


		panel.add(Box.createVerticalStrut(10));
		panel.add(valueRB);
		panel.add(scalaRB);
		panel.add(efficRB);
		chartType.add(valueRB);
		chartType.add(scalaRB);
		chartType.add(efficRB);
		panel.add(Box.createVerticalStrut(10));
		panel.add(weakScaling);
		panel.add(strongScaling);
		scalingType.add(strongScaling);
		scalingType.add(weakScaling);
		panel.add(Box.createVerticalStrut(10));

		this.horizontal.setToolTipText("Create a horizontal chart");
		this.horizontal.addActionListener(this);
		panel.add(this.horizontal);

		return (panel);
	}


	private JPanel createButtonMenu() {
		JPanel panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));
		// export button
		exportdata = new JCheckBox ("Export");
		exportdata.setToolTipText("Export chart data to file");
		exportdata.addActionListener(this);
		panel.add(exportdata);

		
		// apply button
		apply = new JButton ("Apply");
		apply.setToolTipText("Apply changes and redraw chart");
		apply.addActionListener(this);
		panel.add(apply);


		// reset button
		reset = new JButton ("Reset");
		reset.setToolTipText("Change back to default settings");
		reset.addActionListener(this);
		panel.add(reset);

		return (panel);
	}

	private void resetChartSettings() {
		// top toggle buttons
		this.mainOnly.setSelected(true);
		this.callPath.setSelected(false);
		this.logY.setSelected(false);
		valueRB.setSelected(true);
		weakScaling.setSelected(true);
		this.horizontal.setSelected(false);
		this.showZero.setSelected(true);
		// left text fields
		// left combo boxes
		this.dimension.setSelectedIndex(0);
		this.dimensionXLabel.setEnabled(false);
		this.dimensionXValue.setEnabled(false);
		this.eventLabel.setEnabled(false);
		this.event.setEnabled(false);

		this.seriesXmlNameLabel.setEnabled(false);
		this.seriesXmlName.setEnabled(false);

		this.xmlNameLabel.setEnabled(false);
		this.xmlName.setEnabled(false);

		this.seriesXmlNameLabel.setEnabled(false);
		this.seriesXmlName.setEnabled(false);

		// series name 
		for (Iterator<String> itr = tableColumns.iterator() ; itr.hasNext() ; ) {
			Object o = itr.next();
			String tmp = (String)o;
			if (tmp.equalsIgnoreCase("experiment.name")) {
				this.series.setSelectedItem(o);
			} else if (tmp.equalsIgnoreCase("trial.threads_of_execution")) {
				//this.xaxisValue.setSelectedItem(o);
				//} else if (tmp.equalsIgnoreCase("trial.xml_metadata")) {
				this.xaxisValue.setSelectedItem(o);
			}
		}
		this.yaxisValue.setSelectedIndex(0);
		this.yaxisStat.setSelectedIndex(0);
		refreshDynamicControls(true, true, false);
	}


	private void refreshEventList(PerfExplorerModel theModel, String label, String all){

		Object[] evt = null;
		Object[] oldEvent = null;

		evt = this.event.getSelectedValues();
		if (evt != null)
			oldEvent = evt;
		else{
			oldEvent= new Object[0];
		}

		int oldWidth=eventScrollPane.getWidth();

		List<String> events = null;

		if(all.equals(INTERVAL_EVENT_GROUP_ALL))
		{
			events=server.getPotentialGroups(theModel);
		}
		else if(all.equals(ATOMIC_EVENT_ALL))
		{
			events=server.getPotentialAtomicEvents(theModel);
		}
		else if(all.equals(INTERVAL_EVENT_ALL))
		{
			events=server.getPotentialEvents(theModel);
		}
		else {
			events = new ArrayList<String>();
		}
		if(!theModel.getEventNoCallpath()){
			List<String> cpevents = server.getPotentialCallPathEvents(theModel);
			for(String e: cpevents){
				events.add(e);
			}
		}
		this.event.removeAll();
		//this.eventModel.addElement(all);
		events.add(0, all);
		this.eventLabel.setText(label);
		this.event.setSelectedIndex(0);
		int dex=0;
		event.setListData(events.toArray());
		eventScrollPane.setPreferredSize(new Dimension(oldWidth,eventScrollPane.getHeight()));//TODO: Keep the standard minimum size in effect
		for (Iterator<String> itr = events.iterator() ; itr.hasNext() ; ) {
			String next = itr.next();
			//this.eventModel.addElement(next);
			List<Object> ol = Arrays.asList(oldEvent);
			if (ol.contains(next))
				this.event.addSelectionInterval(dex, dex);
			dex++;
		}
	}

	public void refreshDynamicControls(boolean getMetrics, boolean getEvents, boolean getXML) {
		PerfExplorerModel theModel = PerfExplorerModel.getModel();
		Object selection = theModel.getCurrentSelection();
		String oldMetric = "";
		String oldXML = "";
		String oldSXML="";
		Object obj = null;
		if (getMetrics) {
			obj = this.metric.getSelectedItem();
			if (obj != null)
				oldMetric = (String)obj;
			this.metric.removeAllItems();
		}

		if (getXML) {
			obj = this.xmlName.getSelectedItem();
			if (obj != null)
				oldXML = (String)obj;
			this.xmlName.removeAllItems();

			obj = this.seriesXmlName.getSelectedItem();
			if(obj != null)
				oldSXML = (String)obj;
			this.seriesXmlName.removeAllItems();
		}
		if ((selection instanceof Application) ||
				(selection instanceof Experiment) ||
				(selection instanceof Trial)) {
			if (getMetrics) {
				List<String> metrics = server.getPotentialMetrics(theModel);
				//this.metric.setSelectedIndex(0);
				//				boolean gotTime = false;
				for (Iterator<String> itr = metrics.iterator() ; itr.hasNext() ; ) {
					String next = itr.next();
					//					if (next.toUpperCase().indexOf("TIME") > 0) {
					//						gotTime = true;
					//					}
					this.metric.addItem(next);
					if (oldMetric.equals(next))
						this.metric.setSelectedItem(next);
				}
				//				if (gotTime) {
				//					this.metric.addItem("TIME");
				//					if (oldMetric.equals("TIME"))
				//						this.metric.setSelectedItem("TIME");
				//				}
			} 
			if (getEvents && !this.mainOnly.isSelected()) {
				String seriesSelection = (String)series.getSelectedItem();
				//String tmp = (String)obj2;
				if (seriesSelection.equalsIgnoreCase(INTERVAL_EVENT_GROUP_NAME)) {
					resetYAxisValues(true);
					yaxisStat.setSelectedItem(MEAN);
					yaxisValue.setSelectedItem(EXCLUSIVE);
					refreshEventList(theModel,"Groups:",INTERVAL_EVENT_GROUP_ALL);
				} else if (seriesSelection.equalsIgnoreCase(ATOMIC_EVENT_NAME)) {
					resetYAxisValues(false);
					refreshEventList(theModel,"Atomic Events:",ATOMIC_EVENT_ALL);
				} else {
					refreshEventList(theModel,"Events:",INTERVAL_EVENT_ALL);
				}
			} 
			if (getXML) {
				//Object objSer = series.getSelectedItem();
				String tmpSer = (String)series.getSelectedItem();
				//Object objX = xaxisValue.getSelectedItem();
				String tmpX = (String)xaxisValue.getSelectedItem();
				List<String> xmlEvents = null;

				if (tmpX.equalsIgnoreCase("trial.xml_metadata")) {
					xmlEvents = server.getXMLFields(theModel);
					for (Iterator<String> itr = xmlEvents.iterator(); itr.hasNext();) {
						String next = itr.next();
						this.xmlName.addItem(next);
						if (oldXML.equals("") && next.equalsIgnoreCase("UTC Time"))
							this.xmlName.setSelectedItem(next);
						else if (oldXML.equals(next))
							this.xmlName.setSelectedItem(next);
					}
					//this.xmlName.setSelectedIndex(0);
				}

				if (tmpSer.equalsIgnoreCase("trial.xml_metadata")) {
					if(xmlEvents==null)
						xmlEvents = server.getXMLFields(theModel);
					for (Iterator<String> itr = xmlEvents.iterator(); itr.hasNext();) {
						String next = itr.next();
						this.seriesXmlName.addItem(next);
						if (oldSXML.equals("") && next.equalsIgnoreCase("UTC Time"))
							this.seriesXmlName.setSelectedItem(next);
						else if (oldSXML.equals(next))
							this.seriesXmlName.setSelectedItem(next);
					}
					//this.xmlName.setSelectedIndex(0);
				}

			} 
		}
	}

	private void resetYAxisValues(boolean intervalEvent) {

		Object oldY = this.yaxisValue.getSelectedItem();
		Object oldYStat = this.yaxisStat.getSelectedItem();
		this.yaxisValue.removeAllItems();
		this.yaxisStat.removeAllItems();
		if (intervalEvent) {

			this.yaxisStat.addItem(MEAN);
			this.yaxisStat.addItem(TOTAL);
			this.yaxisStat.addItem(MAX);
			this.yaxisStat.addItem(MIN);
			this.yaxisStat.addItem(AVG);

			this.yaxisValue.addItem(INCLUSIVE);
			this.yaxisValue.addItem(EXCLUSIVE);
			this.yaxisValue.addItem("inclusive_percentage");
			this.yaxisValue.addItem("exclusive_percentage");
			this.yaxisValue.addItem("call");
			this.yaxisValue.addItem("subroutines");
			this.yaxisValue.addItem("inclusive_per_call");
			this.yaxisValue.addItem("sum_exclusive_squared");
		} else {

			this.yaxisStat.addItem(ATOMIC);

			this.yaxisValue.addItem("sample_count");
			this.yaxisValue.addItem("maximum_value");
			this.yaxisValue.addItem("minimum_value");
			this.yaxisValue.addItem("mean_value");
			this.yaxisValue.addItem("standard_deviation");
			yaxisValue.setSelectedItem("mean_value");
		}
		yaxisValue.setSelectedItem(oldY);  //TODO: this doesn't jive with the 'else' statement here
		yaxisStat.setSelectedItem(oldYStat);
		return;
	}


	private void updateYAxis(){
		// y axis
		String label;
		String series;

		String stat =  (String)yaxisStat.getSelectedItem();
		String value = (String)yaxisValue.getSelectedItem();

		String tmp;// = stat+"."+value;
		String operation = "avg";
		if (stat.equals(ATOMIC)) {
			tmp = "atomic_location_profile"+"."+value;//tmp.replaceAll(ATOMIC, "atomic_location_profile");
		} else if(stat.equals(MEAN)) {	
			tmp = "interval_mean_summary"+"."+value;
		}
		else if(stat.equals(TOTAL)){
			tmp = "interval_total_summary"+"."+value;
		}
		else{
			operation=stat;
			tmp="interval_location_profile"+"."+value;
		}

		if (!this.mainOnly.isSelected()) {
			series = (String)this.series.getSelectedItem();
			if (series.equalsIgnoreCase(INTERVAL_EVENT_GROUP_NAME)) {
				operation = "sum";
			}
		}

		tmp = operation + "(" + tmp + ")";

		label = yaxisName.getText();
		if (label == null || label.length() == 0) {
			series = (String)this.series.getSelectedItem();;
			if (series.equalsIgnoreCase(ATOMIC_EVENT_NAME)) {
				label = (String)yaxisStat.getSelectedItem()+"."+(String)yaxisValue.getSelectedItem();
			} else {
				// build something intelligible

				String labStat=operation;
				if(stat.equals(MEAN)){
					labStat="Mean";
				}else if(stat.equals(TOTAL)){
					labStat="Total";
				}

				label = labStat+" "+(String)this.metric.getSelectedItem();
				//				if (tmp.indexOf("mean") >= 0) {
				//					label = "Mean " + (String)this.metric.getSelectedItem();
				//				} else if (tmp.indexOf("total") >= 0) {
				//					label = "Total " + (String)this.metric.getSelectedItem();
				//				}
			}
		}
		label += " - "  + (String)this.units.getSelectedItem();
		facade.setChartYAxisName(tmp, label);
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
			Object obj = this.series.getSelectedItem();
			String tmp = (String)obj;
			if (tmp.equalsIgnoreCase(ATOMIC_EVENT_NAME)) {
				title = "Atomic Events";//(String)event.getSelectedItem();
				title = title + " : " + (String)yaxisStat.getSelectedItem()+"."+(String)yaxisValue.getSelectedItem();
			} else {
				title = (String)metric.getSelectedItem();
			}
		}
		facade.setChartTitle(title);

		// series name
		Object obj = series.getSelectedItem();
		String tmp = (String)obj;
		facade.setChartSeriesXML(false);
		if (tmp.equalsIgnoreCase("trial.threads_of_execution")) {
			tmp = "trial.node_count * trial.contexts_per_node * trial.threads_per_context";
		} else if (tmp.equalsIgnoreCase("trial.XML_METADATA")) {
			tmp = "temp_xml_metadata.metadata_value";
			Object obj2 = seriesXmlName.getSelectedItem();
			String tmp2 = (String)obj2;
			facade.setChartMetadataFieldName(tmp2);
			tmp=tmp2;
			facade.setChartSeriesXML(true);

		}
		facade.setChartSeriesName(tmp);//TODO: Should this be tmp1?

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

		//y axis
		updateYAxis();

		// metric name
		obj = metric.getSelectedItem();
		tmp = (String)obj;
		//Not Sure why this was here, it's causing problems, so I have commented it out. Suzanne
		//		if (tmp.equals("TIME"))
		//			facade.setMetricName("%TIME%");
		//		else
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
			if (tmp.equalsIgnoreCase(INTERVAL_EVENT_NAME) || tmp.equalsIgnoreCase(EXPERIMENT_NAME)|| tmp.equalsIgnoreCase(EXPERIMENT_ID)) {
				setEvents(INTERVAL_EVENT_ALL);
			} else if (tmp.equalsIgnoreCase(INTERVAL_EVENT_GROUP_NAME)) {
				setEvents(INTERVAL_EVENT_GROUP_ALL);
			} else if (tmp.equalsIgnoreCase(ATOMIC_EVENT_NAME)) {
				setEvents(ATOMIC_EVENT_ALL);
			}

		}

		facade.setChartEventNoCallPath(this.callPath.isSelected()?0:1); //reversed logic
		facade.setChartLogYAxis(this.logY.isSelected()?1:0);
		facade.setChartScalability(this.scalaRB.isSelected()?1:0);
		facade.setChartEfficiency(this.efficRB.isSelected()?1:0);
		facade.setChartConstantProblem(this.strongScaling.isSelected()?0:1);
		facade.setChartHorizontal(this.horizontal.isSelected()?1:0);
		facade.setShowZero(this.showZero.isSelected()?1:0);

		// create the Chart

		try{

			doGeneralChart();
		}
		catch (SeriesException e) {
			// this shouldn't happen, but if it does, handle it gracefully.
			StringBuilder sb = new StringBuilder();
			sb.append("Two or more trials in this selection have the same total number of threads of execution, and an error occurred.\n");
			sb.append("To create a scalability chart, please ensure the trials selected have different numbers of threads.\n");
			//sb.append("To create a different parametric chart, please use the custom chart interface.");	
			//TODO: Check if threads of execution is the x axis, if so suggest switching to categorical
			JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), sb.toString(),
					"Selection Warning", JOptionPane.ERROR_MESSAGE);
		} catch (Exception e) {
			System.err.println("actionPerformed Exception: " + e.getMessage());
			e.printStackTrace();
		} 
	}

	private void setEvents(String all){
		Object[] tmp = this.event.getSelectedValues();
		if (!tmp[0].equals(all)) {
			for(int i=0;i<tmp.length;i++)
				facade.addEventName((String)tmp[i]);
		} else {
			facade.setEventName(null);
		}
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
		} else if (source == mainOnly) {
			if (mainOnly.isSelected()) {
				this.eventLabel.setEnabled(false);
				this.event.setEnabled(false);
				series.setSelectedItem(EXPERIMENT_NAME);
				yaxisValue.setSelectedItem(INCLUSIVE);
			} else {
				this.eventLabel.setEnabled(true);
				this.event.setEnabled(true);

				series.setSelectedItem(INTERVAL_EVENT_NAME);
				resetYAxisValues(true);
				yaxisStat.setSelectedItem(MEAN);
				yaxisValue.setSelectedItem(EXCLUSIVE);

				refreshDynamicControls(false, true, false);
			}
		}else if(source == callPath){
			facade.setChartEventNoCallPath(callPath.isSelected()?0:1);

			refreshDynamicControls(false, true, false);


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

			/*
			 * There are two places to select xml metadata, one for the series and
			 * one for the x-axis.  Only enable/disable the correct one.
			 */
			if(source==series){
				if(tmp.equalsIgnoreCase("trial.xml_metadata")){
					this.seriesXmlNameLabel.setEnabled(true);
					this.seriesXmlName.setEnabled(true);
					refreshDynamicControls(false, false, true);
				}else{
					this.seriesXmlNameLabel.setEnabled(false);
					this.seriesXmlName.setEnabled(false);
				}
			}
			else if(source==xaxisValue){
				if(tmp2.equalsIgnoreCase("trial.xml_metadata")){
					this.xmlNameLabel.setEnabled(true);
					this.xmlName.setEnabled(true);
					refreshDynamicControls(false, false, true);
				}else{
					this.xmlNameLabel.setEnabled(false);
					this.xmlName.setEnabled(false);
				}
			}


			//			if (tmp.equalsIgnoreCase("trial.xml_metadata") ||
			//				tmp2.equalsIgnoreCase("trial.xml_metadata")) {
			//				if(source==xaxisValue){
			//					this.xmlNameLabel.setEnabled(true);
			//					this.xmlName.setEnabled(true);
			//				}else{
			//					this.seriesXmlNameLabel.setEnabled(true);
			//					this.seriesXmlName.setEnabled(true);
			//				}
			//				
			//				refreshDynamicControls(false, false, true);
			//			} else {
			//				if(source==xaxisValue){
			//					this.xmlNameLabel.setEnabled(false);
			//					this.xmlName.setEnabled(false);
			//				}else{
			//					this.seriesXmlNameLabel.setEnabled(false);
			//					this.seriesXmlName.setEnabled(false);
			//				}
			//			}
			if (tmp.equalsIgnoreCase(INTERVAL_EVENT_NAME) ||
					tmp.equalsIgnoreCase(INTERVAL_EVENT_GROUP_NAME)) {
				refreshDynamicControls(false, true, false);
			}
			if (tmp.equalsIgnoreCase(ATOMIC_EVENT_NAME)) {
				this.metricLabel.setEnabled(false);
				this.metric.setEnabled(false);
				this.dimensionLabel.setEnabled(false);
				this.dimension.setEnabled(false);
				refreshDynamicControls(false, true, false);
				this.units.setSelectedIndex(5);  // change to "units"
			} else {
				this.metricLabel.setEnabled(true);
				this.metric.setEnabled(true);
				this.dimensionLabel.setEnabled(true);
				this.dimension.setEnabled(true);
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
	@SuppressWarnings("unchecked")
	private PerfExplorerChart doGeneralChart () {

		// get the data
		PerfExplorerModel model = PerfExplorerModel.getModel();
		RMIGeneralChartData rawData = server.requestGeneralChartData(
				model, ChartDataType.PARAMETRIC_STUDY_DATA);

		PECategoryDataset dataset = new PECategoryDataset();

		DefaultTableXYDataset xydataset = new DefaultTableXYDataset();

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

		PlotOrientation orientation = PlotOrientation.VERTICAL;
		if (model.getChartHorizontal()) {
			orientation = PlotOrientation.HORIZONTAL;
		}

		JFreeChart chart = null;
		XYSeries ideal=null;
		Map<String,XYSeries> labelMap=new HashMap<String,XYSeries>();
		if (rawData.getCategoryType() == Integer.class) 
		{

			if (model.getChartScalability()) {

				// create an "ideal" line.
				//dataset.addValue(1.0, IDEAL, new Integer(rawData.getMinimum()));
				//				dataset.addValue(rawData.getMaximum()/rawData.getMinimum(),IDEAL, 
				//						new Integer(rawData.getMaximum()));

				// get the baseline values
				edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData.CategoryDataRow baseline = rawData.getRowData(0);

				// iterate through the values
				for (int i = 0 ; i < rawData.getRows() ; i++) {
					edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
					if (!shortName(row.series).equals(shortName(baseline.series))) {
						baseline = row;
					}


					XYSeries s = labelMap.get(row.series);
					if(s==null){
						s=new XYSeries(shortName(row.series), true, false);
						labelMap.put(row.series, s);

					}

					if (model.getConstantProblem().booleanValue()) {

						double ratio = baseline.categoryInteger.doubleValue() / 
						row.categoryInteger.doubleValue();
						double efficiency = baseline.value/row.value;
						dataset.addValue(efficiency / ratio, shortName(row.series), row.categoryInteger);
						s.add(row.categoryInteger.doubleValue(),efficiency / ratio);
					} else {
						dataset.addValue(baseline.value / row.value, shortName(row.series), row.categoryInteger);
						s.add(row.categoryInteger.doubleValue(),baseline.value / row.value);
					}
				}

				ideal=new XYSeries(IDEAL, true, false);

				// create an "ideal" line.
				List<Integer> keys = dataset.getColumnKeys();
				for (int i = 0 ; i < keys.size() ; i++) {
					Integer key = keys.get(i);
					dataset.addValue(key.doubleValue()/rawData.getMinimum(), IDEAL, key);
					ideal.add(key.doubleValue(),key.doubleValue()/rawData.getMinimum());
				}

			} else if (model.getChartEfficiency()) {

				// get the baseline values
				edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData.CategoryDataRow baseline = rawData.getRowData(0);

				// iterate through the values
				for (int i = 0 ; i < rawData.getRows() ; i++) {
					edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
					if (!shortName(row.series).equals(shortName(baseline.series))) {
						baseline = row;
					}

					XYSeries s = labelMap.get(row.series);
					if(s==null){
						s=new XYSeries(shortName(row.series), true, false);
						labelMap.put(row.series, s);

					}

					if (model.getConstantProblem().booleanValue()) {
						dataset.addValue(baseline.value / row.value, shortName(row.series), row.categoryInteger);
						s.add(row.categoryInteger.doubleValue(), baseline.value / row.value);
					} else {
						dataset.addValue((baseline.value * baseline.categoryInteger.doubleValue())/ (row.value * row.categoryInteger.doubleValue()), shortName(row.series), row.categoryInteger);
						s.add(row.categoryInteger.doubleValue(), (baseline.value * baseline.categoryInteger.doubleValue())/ (row.value * row.categoryInteger.doubleValue()));
					}
				}
				ideal=new XYSeries(IDEAL, true, false);
				//labelMap.put(IDEAL,s);

				// create an "ideal" line.
				List<Integer> keys = dataset.getColumnKeys();
				for (int i = 0 ; i < keys.size() ; i++) {
					Integer key = (Integer)keys.get(i);
					dataset.addValue(1.0, IDEAL, key);
					ideal.add(key.doubleValue(),1.0);
				}

			} else {

				for (int y = 0 ; y < rawData.getRows() ; y++) {
					CategoryDataRow row = rawData.getRowData(y);
					XYSeries s = labelMap.get(row.series);
					if(s==null){
						s=new XYSeries(shortName(row.series), true, false);
						labelMap.put(row.series, s);

					}
					dataset.addValue(row.value / conversion, shortName(row.series), row.categoryInteger);

					s.add(row.categoryInteger.doubleValue(), row.value/conversion);

				}

			}

			for(String key : labelMap.keySet()){
				XYSeries s = labelMap.get(key);
				xydataset.addSeries(s);
			}

			if(ideal!=null){
				xydataset.addSeries(ideal);
			}
			if(exportdata.isSelected()){
				printData(xydataset);
			}

			if(!this.alwaysCategory.isSelected())
			{	

				chart = ChartFactory.createXYLineChart(
						model.getChartTitle(),  // chart title
						model.getChartXAxisLabel(),  // domain axis label
						model.getChartYAxisLabel(),  // range axis label
						xydataset,                         // data
						PlotOrientation.VERTICAL,        // the plot orientation
						true,                            // legend
						true,                            // tooltips
						false                            // urls
				);

				// set the chart to a common style
				Utility.applyDefaultChartTheme(chart);
				customizeLineChart(model,rawData, chart);
				PerfExplorerChart.customizeChart(chart,labelMap.keySet().size(),(ideal!=null));

			} else{
				chart = ChartFactory.createLineChart(
						model.getChartTitle(),  // chart title
						model.getChartXAxisLabel(),  // domain axis label
						model.getChartYAxisLabel(),  // range axis label
						dataset,                         // data
						orientation,        // the plot orientation
						true,                            // legend
						true,                            // tooltips
						false                            // urls
				);

				// set the chart to a common style
				Utility.applyDefaultChartTheme(chart);

				customizeCategoryChart(model, rawData, chart);
			}
		}
		else {
			// iterate through the values
			for (int i = 0 ; i < rawData.getRows() ; i++) {
				edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData.CategoryDataRow row = rawData.getRowData(i);
				dataset.addValue(row.value / conversion, shortName(row.series), row.categoryString);
			}

			chart = ChartFactory.createLineChart(
					model.getChartTitle(),  // chart title
					model.getChartXAxisLabel(),  // domain axis label
					model.getChartYAxisLabel(),  // range axis label
					dataset,                         // data
					orientation,        // the plot orientation
					true,                            // legend
					true,                            // tooltips
					false                            // urls
			);

			// set the chart to a common style
			Utility.applyDefaultChartTheme(chart);

			customizeCategoryChart(model, rawData, chart);
		}

		PerfExplorerChart pec = new PerfExplorerChart(chart, model.getChartTitle());

		return pec;
	}

	private boolean getFiletoSave() {


		int returnVal = fc.showSaveDialog(this);
		if (returnVal == JFileChooser.APPROVE_OPTION) {
			File savefile =  fc.getSelectedFile();
			if(savefile.exists()){
				int result = 	JOptionPane.showOptionDialog(this, 
						"\""+savefile.getName()+"\" already exists. Do you want to replace it? ", "Replace File"
						,JOptionPane.YES_NO_OPTION,JOptionPane.WARNING_MESSAGE, null, null, null);
				if(result ==JOptionPane.NO_OPTION) 
					return getFiletoSave();
			}
			return true;
		}else 
			return false;//user hit cancel 
	}

	private void printData(DefaultTableXYDataset xydataset)   {
		if(getFiletoSave()){
			try {
				File savefile =  fc.getSelectedFile();
				//File savefile = new File("/Users/somillstein/Desktop/chartdata");
				FileOutputStream write;
				write = new FileOutputStream(savefile);
				PerfExplorerModel model = PerfExplorerModel.getModel();
				writeln(model.getChartTitle(),write);
				writeln(model.getChartXAxisLabel(),write);
				writeln(model.getChartYAxisLabel(),write);
				
				for(int seriesID=0; seriesID<xydataset.getSeriesCount();seriesID++){
					XYSeries series = xydataset.getSeries(seriesID);
					writeln(series.getKey(), write);
					for(int i=0;i<series.getItemCount();i++){
						write(series.getX(i) + "\t", write);
						writeln(series.getY(i)+ "\t", write);
					}

				}
			} catch (FileNotFoundException e) {
				StringBuilder sb = new StringBuilder();
				sb.append("File not found.\n");
				JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), sb.toString(),
						"Selection Warning", JOptionPane.ERROR_MESSAGE);
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

	}
	private void writeln(Object s, FileOutputStream write) throws IOException {
		write(s+"\n",write);
		
	}

	private void write(String s, FileOutputStream write) throws IOException{
		write.write(s.getBytes());
	}

	private void customizeLineChart(PerfExplorerModel model,RMIGeneralChartData rawData, JFreeChart chart) {
		// get a reference to the plot for further customisation...
		XYPlot plot = (XYPlot)chart.getPlot();

		//StandardXYItemRenderer renderer = (StandardXYItemRenderer) plot.getRenderer();
		XYLineAndShapeRenderer renderer = (XYLineAndShapeRenderer)plot.getRenderer();
		renderer.setBaseShapesFilled(true);
		renderer.setBaseShapesVisible(true);//Was drawshapes
		renderer.setDrawOutlines(true);//Was drawlines
		renderer.setBaseItemLabelsVisible(true);
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
	}

	private void customizeCategoryChart(PerfExplorerModel model,
			RMIGeneralChartData rawData, JFreeChart chart) {
		// get a reference to the plot for further customisation...
		CategoryPlot plot = (CategoryPlot)chart.getPlot();

		//StandardXYItemRenderer renderer = (StandardXYItemRenderer) plot.getRenderer();
		LineAndShapeRenderer renderer = (LineAndShapeRenderer)plot.getRenderer();
		renderer.setBaseShapesFilled(true);
		renderer.setBaseShapesVisible(true);//Was drawshapes
		renderer.setDrawOutlines(true);//Was drawlines
		renderer.setBaseItemLabelsVisible(true);
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

		if (angleXLabels.isSelected()){//angledXaxis && !scalingChart) {
			//CategoryPlot plot = chart.getCategoryPlot();
			CategoryAxis xAxis = (CategoryAxis) plot.getDomainAxis();
			xAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_45);
		}
	}


	private String shortName(String longName) {

		int codeDex = longName.indexOf("[{");
		//If this is somehow the first string, we don't want to make an empty string...
		if(codeDex<1){
			longName=longName.trim();
			return longName;
		}

		String shorter = longName.substring(0, codeDex).trim();
		return shorter;
	}

	class MyJTextField extends javax.swing.JTextField
	{   
		/**
		 * 
		 */
		private static final long serialVersionUID = -7156539927712296439L;

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

	class MyJComboBox extends javax.swing.JComboBox
	{   
		/**
		 * 
		 */
		private static final long serialVersionUID = -7805756500468380965L;

		public MyJComboBox(Object[] items) {
			super(items);
			setPrototypeDisplayValue("WWWWW");
		}

		public MyJComboBox() {
			super();
			setPrototypeDisplayValue("WWWWW");
		}

		public MyJComboBox(List<Object> items) {
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
				/**
				 * 
				 */
				private static final long serialVersionUID = -992135884016287671L;

				public void setVisible(boolean showIt) {
					if (showIt) {
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
					} else {
						super.setVisible(false);
					}
				}
			};
			popup.getAccessibleContext().setAccessibleParent(comboBox);
			return popup;
		}
	}

	/**
	 * from http://www.codeguru.com/java/articles/163.shtml
	 */
	class SteppedComboBox extends JComboBox {
		/**
		 * 
		 */
		private static final long serialVersionUID = -6511789381891153830L;
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

		public SteppedComboBox(List<Object> items) {
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

	//	private class ChartPanelException extends Exception {
	//		ChartPanelException (String message) {
	//			super(message);
	//		}
	//	}
}
