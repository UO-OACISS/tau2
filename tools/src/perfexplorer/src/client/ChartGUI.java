/**
 * 
 */
package edu.uoregon.tau.perfexplorer.client;

import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.GridBagConstraints;
import java.awt.HeadlessException;
import java.awt.Insets;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.GridBagLayout;

import java.net.URL;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import org.jfree.data.general.SeriesException;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.common.Console;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;
import edu.uoregon.tau.perfexplorer.common.RMIView;
import edu.uoregon.tau.perfexplorer.common.TransformationType;

/**
 * @author khuck
 *
 */
public class ChartGUI extends JFrame implements ActionListener {
	private ActionListener listener = null;
	private JPanel panel = null;
	private final int windowWidth = 480;
	private final int windowHeight = 400;
	private final JLabel chartLabel = new JLabel("Chart Type");
	private final JLabel metricLabel = new JLabel(PerfExplorerActionListener.SET_METRICNAME);
	private final JLabel groupLabel = new JLabel(PerfExplorerActionListener.SET_GROUPNAME);
	private final JLabel eventLabel = new JLabel(PerfExplorerActionListener.SET_EVENTNAME);
	private final JLabel scalingLabel = new JLabel(PerfExplorerActionListener.SET_PROBLEM_SIZE);
	private final JLabel timestepsLabel = new JLabel(PerfExplorerActionListener.SET_TIMESTEPS);
	private final JLabel reductionMethodLabel = new JLabel(PerfExplorerActionListener.DIMENSION_REDUCTION);
	private final JLabel reductionThresholdLabel = new JLabel("Reduction Threshold");
   	private JComboBox chart = new JComboBox();
   	private JComboBox metric = new JComboBox();
   	private JComboBox group = new JComboBox();
   	private JComboBox event = new JComboBox();
   	private JComboBox scaling = new JComboBox();
   	private JComboBox reductionMethod = new JComboBox();
   	private JTextField timesteps = new JTextField("100");
   	private JTextField reductionThreshold = new JTextField("2.0");
   	private JButton goButton = new JButton("Create Chart");
   	private PerfExplorerConnection server = null;
   	private PerfExplorerModel theModel = null;
   	private static ChartGUI theInstance = null;
   	private static boolean averageWarning = false;

	/**
	 * @param title
	 * @throws HeadlessException
	 */
	private ChartGUI(String title, ActionListener listener) throws HeadlessException {
		super(title);
		this.server = PerfExplorerConnection.getConnection();
		this.listener = listener;
		setListeners();
		this.panel = new JPanel(new GridBagLayout());
		
		getContentPane().add(panel);
		PerfExplorerWindowUtility.centerWindow(this, windowWidth, windowHeight, -200, -200, true);

    	URL url = Utility.getResource("tau32x32.gif");
    	if (url != null)
    		setIconImage(Toolkit.getDefaultToolkit().getImage(url));
    	
		// null the static reference if the user closes the window.
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				ChartGUI.theInstance = null;
			}
		});

    	addComponents();
    	refreshStatic();
		refreshDynamic();
		doEnableDisable();
		setResizable(false);
		pack();
		setVisible(true);
		toFront();
	}
	
	public static ChartGUI getInstance(boolean createIfNot) {
		if (theInstance == null && createIfNot) {
			theInstance = new ChartGUI("Scalabilty Chart Control Center", PerfExplorerClient.getMainFrame().getListener());
		}
		return theInstance;
	}

	private void setListeners() {
		this.chart.addActionListener(this);
		this.metric.addActionListener(this);
		this.group.addActionListener(this);
		this.scaling.addActionListener(this);
		this.timesteps.addActionListener(this);
		this.goButton.addActionListener(this);
	}

	private void doEnableDisable() {
		ChartType chartType = (ChartType)this.chart.getSelectedItem();
		this.eventLabel.setEnabled(false);
		this.groupLabel.setEnabled(false);
		this.timestepsLabel.setEnabled(false);
		this.scalingLabel.setEnabled(false);
		this.reductionMethodLabel.setEnabled(false);
		this.reductionThresholdLabel.setEnabled(false);
		this.event.setEnabled(false);
		this.group.setEnabled(false);
		this.timesteps.setEnabled(false);
		this.scaling.setEnabled(false);
		this.reductionMethod.setEnabled(false);
		this.reductionThreshold.setEnabled(false);
		switch (chartType.index) {
		case 0:  //timesteps per second
			this.timesteps.setEnabled(true);
			break;
		case 3:
		case 7:
			this.reductionMethodLabel.setEnabled(true);
			this.reductionMethod.setEnabled(true);
			this.reductionThresholdLabel.setEnabled(true);
			this.reductionThreshold.setEnabled(true);
			// don't break - fall through
		case 2:
		case 5:
		case 6:
		case 9:
			// all of the scaling/efficiency charts except "for one event"
			this.scalingLabel.setEnabled(true);
			this.scaling.setEnabled(true);
			break;
		case 4:
		case 8:
			// the scaling/efficiency charts "for one event"
			this.scalingLabel.setEnabled(true);
			this.scaling.setEnabled(true);
			this.eventLabel.setEnabled(true);
			this.event.setEnabled(true);
			break;
		case 10:  // group percentage of total
			this.groupLabel.setEnabled(true);
			this.group.setEnabled(true);
			break;
		case 12:
			this.scalingLabel.setEnabled(true);
			this.scaling.setEnabled(true);
		case 11:
			this.reductionMethodLabel.setEnabled(true);
			this.reductionMethod.setEnabled(true);
			this.reductionThresholdLabel.setEnabled(true);
			this.reductionThreshold.setEnabled(true);
			break;
		}
	}

	private void addComponents() {
		GridBagConstraints c = new GridBagConstraints();
		this.theModel = PerfExplorerModel.getModel();
		c.fill = GridBagConstraints.BOTH;
		c.anchor = GridBagConstraints.CENTER;
		c.weightx = 0.5;
		c.insets = new Insets(2,2,2,2);
		c.gridx = 0;
		c.gridy = 0;
		panel.add(this.chartLabel, c);
		c.gridy++;
		panel.add(this.chart, c);
		c.gridy++;
		panel.add(this.metricLabel, c);
		c.gridy++;
		panel.add(this.metric, c);
		c.gridy++;
		panel.add(this.groupLabel, c);
		c.gridy++;
		panel.add(this.group, c);
		c.gridy++;
		panel.add(this.eventLabel, c);
		c.gridy++;
		panel.add(this.event, c);
		c.gridy++;
		panel.add(this.reductionMethodLabel, c);
		c.gridy++;
		panel.add(this.reductionMethod, c);
		c.gridy++;
		panel.add(this.reductionThresholdLabel, c);
		c.gridy++;
		panel.add(this.reductionThreshold, c);
		c.gridy++;
		panel.add(this.timestepsLabel, c);
		c.gridy++;
		panel.add(this.timesteps, c);
		c.gridy++;
		panel.add(this.scalingLabel, c);
		c.gridy++;
		panel.add(this.scaling, c);
		c.gridy++;
		c.gridwidth = 2;
		c.fill = GridBagConstraints.NONE;
		panel.add(this.goButton, c);
	}
	
	private void refreshStatic() {
		this.chart.removeAllItems();
		this.chart.addItem(new ChartType(PerfExplorerActionListener.TOTAL_TIME_CHART,1));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.TIMESTEPS_CHART,0));		
		this.chart.addItem(new ChartType(PerfExplorerActionListener.EFFICIENCY_CHART,2));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.EFFICIENCY_EVENTS_CHART,3));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.EFFICIENCY_ONE_EVENT_CHART,4));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.EFFICIENCY_PHASE_CHART,5));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.SPEEDUP_CHART,6));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.SPEEDUP_EVENTS_CHART,7));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.SPEEDUP_ONE_EVENT_CHART,8));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.SPEEDUP_PHASE_CHART,9));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.COMMUNICATION_CHART,10));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.FRACTION_CHART,11));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.CORRELATION_CHART,12));
		this.chart.addItem(new ChartType(PerfExplorerActionListener.FRACTION_PHASE_CHART,13));
		
		this.scaling.removeAllItems();
		this.scaling.addItem("The problem size remains constant. (strong scaling)");
		this.scaling.addItem("The problem size increases as the processor count increases. (weak scaling)");
		
		this.reductionMethod.removeAllItems();
		this.reductionMethod.addItem("None");
		this.reductionMethod.addItem("Minimum Percentage");
		this.reductionMethod.setSelectedIndex(1);
	}
	
	public void refresh() {
		refreshDynamic();
		return;
	}
	
	private void refreshDynamic() {
		// if nothing selected, exit early
		if (theModel.getApplication() == null) {
			return;
		}
		if (this.metric.isEnabled())
			getMetrics();
		if (this.group.isEnabled())
			getGroups();
		if (this.event.isEnabled())
			getEvents();
	}

	private void getMetrics() {
		Object obj = this.metric.getSelectedItem();
		String oldMetric = "";
		
		// save the old metric selection
		if (obj != null)
			oldMetric = (String)obj;
		// clear the selection box
		this.metric.removeAllItems();

		List<String> metrics = server.getPotentialMetrics(theModel);
		boolean gotTime = false;
		for (Iterator<String> itr = metrics.iterator() ; itr.hasNext() ; ) {
			String next = itr.next();
			if (next.toUpperCase().indexOf("TIME") > 0) {
				gotTime = true;
			}
			this.metric.addItem(next);
			if (oldMetric.equals(next))
				this.metric.setSelectedItem(next);
		}
		if (gotTime) {
			this.metric.addItem("TIME");
			if (oldMetric.equals("TIME"))
				this.metric.setSelectedItem("TIME");
		}
	}
	
	private void getEvents() {
		Object obj = this.event.getSelectedItem();
		String oldEvent = "";
		if (obj != null)
			oldEvent = (String)obj;
		this.event.removeAllItems();

		List<String> events = server.getPotentialEvents(theModel);
		for (Iterator<String> itr = events.iterator() ; itr.hasNext() ; ) {
			String next = itr.next();
			this.event.addItem(next);
			if (oldEvent.equals(next))
				this.event.setSelectedItem(next);
		}
	}
	
	private void getGroups() {
		Object obj = this.group.getSelectedItem();
		String oldGroup = "";
		if (obj != null)
			oldGroup = (String)obj;
		this.group.removeAllItems();
		
		List<String> events = server.getPotentialGroups(theModel);
		for (Iterator<String> itr = events.iterator() ; itr.hasNext() ; ) {
			String next = itr.next();
			this.group.addItem(next);
			if (oldGroup.equals(next))
				this.group.setSelectedItem(next);
		}
	}
	
	private class ChartType {
		int index = 0;
		String name = "";
		
		public ChartType(String name, int index) {
			this.name = name;
			this.index = index;
		}
		
		public String toString() {
			return name;
		}
	}

	public void actionPerformed(ActionEvent actionEvent) {
		try {
			Object eventSrc = actionEvent.getSource();
			if (eventSrc.equals(this.chart)) {
				refreshDynamic();
				doEnableDisable();
			} else if (eventSrc.equals(this.goButton)) {
				PerfExplorerModel theModel = PerfExplorerModel.getModel();
				Object selection = theModel.getCurrentSelection();
				// allow Experiments or Trials or 1 view
				if (!(selection instanceof Experiment) && !(selection instanceof RMIView)) {
					JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), "Please select one or more Experiments or one View.",
						"Selection Error", JOptionPane.ERROR_MESSAGE);
					return;
				}
				// set the model, and request the chart.
				String metricName = (String)metric.getSelectedItem();
				String eventName = (String)event.getSelectedItem();
				String groupName = (String)group.getSelectedItem();
				String totalTimesteps = timesteps.getText();
				String scalingName = (String)scaling.getSelectedItem();
				String reductionMethodName = (String)this.reductionMethod.getSelectedItem();
				String reductionThresholdValue = reductionThreshold.getText();
				theModel.setMetricName(metricName);
				theModel.setEventName(eventName);
				theModel.setGroupName(groupName);
				theModel.setTotalTimesteps(totalTimesteps);
				theModel.setConstantProblem(scalingName.startsWith("The problem size remains") ? true : false);
				if (reductionMethodName.equals("None")) {
					theModel.setDimensionReduction(TransformationType.NONE);
				} else {
					theModel.setDimensionReduction(TransformationType.OVER_X_PERCENT);
					theModel.setXPercent(reductionThresholdValue);
				}
				
				ChartGUI.checkScaling();
				ChartType chartType = (ChartType)this.chart.getSelectedItem();
				switch (chartType.index) {
				case 0:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.TIMESTEPS_CHART,0));
					PerfExplorerChart.doTimestepsChart();
					break;
				case 1:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.TOTAL_TIME_CHART,1));
					PerfExplorerChart.doTotalTimeChart();
					break;
				case 2:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.EFFICIENCY_CHART,2));
					PerfExplorerChart.doEfficiencyChart();
					break;
				case 3:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.EFFICIENCY_EVENTS_CHART,3));
					PerfExplorerChart.doEfficiencyEventsChart();
					break;
				case 4:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.EFFICIENCY_ONE_EVENT_CHART,4));
					PerfExplorerChart.doEfficiencyOneEventChart();
					break;
				case 5:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.EFFICIENCY_PHASE_CHART,5));
					PerfExplorerChart.doEfficiencyPhasesChart();
					break;
				case 6:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.SPEEDUP_CHART,6));
					PerfExplorerChart.doSpeedupChart();
					break;
				case 7:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.SPEEDUP_EVENTS_CHART,7));
					PerfExplorerChart.doSpeedupEventsChart();
					break;
				case 8:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.SPEEDUP_ONE_EVENT_CHART,8));
					PerfExplorerChart.doSpeedupOneEventChart();
					break;
				case 9:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.SPEEDUP_PHASE_CHART,9));
					PerfExplorerChart.doSpeedupPhasesChart();
					break;
				case 10:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.COMMUNICATION_CHART,10));
					PerfExplorerChart.doCommunicationChart();
					break;
				case 11:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.FRACTION_CHART,11));
					PerfExplorerChart.doFractionChart();
					break;
				case 12:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.CORRELATION_CHART,12));
					PerfExplorerChart.doCorrelationChart();
					break;
				case 13:
//					this.chart.addItem(new ChartType(PerfExplorerActionListener.FRACTION_PHASE_CHART,13));
					PerfExplorerChart.doFractionPhasesChart();
					break;
				}
			} else {
				// not handled yet
			}
		} catch (SeriesException e) {
			// this shouldn't happen, but if it does, handle it gracefully.
			StringBuilder sb = new StringBuilder();
			sb.append("Two or more trials in this selection have the same total number of threads of execution, and an error occurred.\n");
			sb.append("To create a scalability chart, please ensure the trials selected have different numbers of threads.\n");
			sb.append("To create a different parametric chart, please use the custom chart interface.");			
			JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), sb.toString(),
					"Selection Warning", JOptionPane.ERROR_MESSAGE);
		} catch (Exception e) {
			System.err.println("actionPerformed Exception: " + e.getMessage());
			e.printStackTrace();
		} 
	}

	public static void checkScaling() {
		// check to make sure that there is only one value for each processor count
		PerfExplorerModel theModel = PerfExplorerModel.getModel();
		Map<Integer, Integer> counts = PerfExplorerConnection.getConnection().checkScalabilityChartData(theModel);
		for (Integer threads : counts.keySet()) {
			Integer count = counts.get(threads);
			if (count > 1 && !ChartGUI.averageWarning) {
				StringBuilder sb = new StringBuilder();
				sb.append("Two or more trials in this selection have the same total number of threads of execution.\n");
				sb.append("Trials with the same numbers of threads will have their measurements averaged.\n");
				sb.append("To create a different parametric chart, please use the custom chart interface.");			
				JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), sb.toString(),
						"Selection Warning", JOptionPane.WARNING_MESSAGE);
				ChartGUI.averageWarning = true;
			}
		}
	}
}
