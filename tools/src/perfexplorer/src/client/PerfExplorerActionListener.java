package client;

import javax.swing.*;
import java.awt.event.*;
import java.util.List;
import java.util.ListIterator;
import edu.uoregon.tau.dms.dss.*;
import common.RMIView;
import common.RMIPerfExplorerModel;

public class PerfExplorerActionListener implements ActionListener {

	public final static String QUIT = "Quit PerfExplorer";
	public final static String QUIT_SERVER = "Quit PerfExplorer (Shutdown Server)";
	public final static String ABOUT = "About PerfExplorer";
	public final static String SEARCH = "Search for help on...";
	// clustering menu items
	public final static String CLUSTERING_METHOD = "Select Clustering Method";
	public final static String DIMENSION_REDUCTION = "Select Dimension Reduction";
	public final static String NORMALIZATION = "Select Normalization Method";
	public final static String NUM_CLUSTERS = "Set Number of Clusters";
	public final static String DO_CLUSTERING = "Do Clustering";
	public final static String DO_CORRELATION_ANALYSIS = "Do Correlation Analysis";
	public final static String DO_CORRELATION_CUBE = "Do 3D Correlation Cube";
	public final static String DO_VARIATION_ANALYSIS = "Do Variation Analysis";
	// chart menu items
	public final static String SET_GROUPNAME = "Set Group Name";
	public final static String SET_METRICNAME = "Set Metric of Interest";
	public final static String SET_TIMESTEPS = "Set Total Number of Timesteps";
	public final static String SET_EVENTNAME = "Set Event of Interest";
	public final static String TIMESTEPS_CHART = "Timesteps Per Second";
	public final static String EFFICIENCY_CHART = "Relative Efficiency";
	public final static String EFFICIENCY_EVENTS_CHART = "Relative Efficiency by Event";
	public final static String EFFICIENCY_ONE_EVENT_CHART = "Relative Efficiency for One Event";
	public final static String SPEEDUP_CHART = "Relative Speedup";
	public final static String SPEEDUP_EVENTS_CHART = "Relative Speedup by Event";
	public final static String SPEEDUP_ONE_EVENT_CHART = "Relative Speedup for One Event";
	public final static String COMMUNICATION_CHART = "Group % of Total Runtime";
	public final static String FRACTION_CHART = "Runtime Breakdown";
	// phase chart menu items
	public final static String EFFICIENCY_PHASE_CHART = "Relative Efficiency per Phase";
	public final static String SPEEDUP_PHASE_CHART = "Relative Speedup per Phase";
	public final static String FRACTION_PHASE_CHART = "Phase Fraction of Total Runtime";
	// arbitrary view menu items
	public final static String CREATE_NEW_VIEW = "Create New View";
	public final static String CREATE_NEW_SUB_VIEW = "Create New Sub-view";

	private PerfExplorerClient mainFrame;

	public PerfExplorerActionListener (PerfExplorerClient mainFrame) {
		super();
		this.mainFrame = mainFrame;
	}

	public void actionPerformed (ActionEvent event) {
		try {
			Object EventSrc = event.getSource();
			if(EventSrc instanceof JMenuItem) {
				String arg = event.getActionCommand();
				if(arg.equals(QUIT)) {
					System.exit(0);
				} else if(arg.equals(QUIT_SERVER)) {
					//PerfExplorerConnection server = PerfExplorerConnection.getConnection();
					//server.stopServer();
					System.exit(0);
			// help menu items
				} else if (arg.equals(ABOUT)) {
					createAboutWindow();
				} else if (arg.equals(SEARCH)) {
					createHelpWindow();
			// clustering items
				} else if (arg.equals(CLUSTERING_METHOD)) {
					createMethodWindow();
				} else if (arg.equals(DIMENSION_REDUCTION)) {
					createDimensionWindow();
				} else if (arg.equals(NORMALIZATION)) {
					createNormalizationWindow();
				} else if (arg.equals(NUM_CLUSTERS)) {
					createClusterSizeWindow();
				} else if (arg.equals(DO_CLUSTERING)) {
					createDoClusteringWindow();
				} else if (arg.equals(DO_CORRELATION_ANALYSIS)) {
					createDoCorrelationWindow();
				} else if (arg.equals(DO_CORRELATION_CUBE)) {
					PerfExplorerCube.doCorrelationCube();
				} else if (arg.equals(DO_VARIATION_ANALYSIS)) {
					PerfExplorerVariation.doVariationAnalysis();
			// chart items
				} else if (arg.equals(SET_GROUPNAME)) {
					checkAndSetGroupName(true);
				} else if (arg.equals(SET_METRICNAME)) {
					checkAndSetMetricName(true);
				} else if (arg.equals(SET_EVENTNAME)) {
					checkAndSetEventName(true);
				} else if (arg.equals(SET_TIMESTEPS)) {
					checkAndSetTimesteps(true);
				} else if (arg.equals(TIMESTEPS_CHART)) {
					if (checkAndSetMetricName(false))
						if (checkAndSetTimesteps(false))
							PerfExplorerChart.doTimestepsChart();
				} else if (arg.equals(EFFICIENCY_CHART)) {
					if (checkAndSetMetricName(false))
						PerfExplorerChart.doEfficiencyChart();
				} else if (arg.equals(EFFICIENCY_EVENTS_CHART)) {
					if (checkAndSetMetricName(false))
						PerfExplorerChart.doEfficiencyEventsChart();
				} else if (arg.equals(EFFICIENCY_ONE_EVENT_CHART)) {
					if (checkAndSetMetricName(false))
						if (checkAndSetEventName(false))
							PerfExplorerChart.doEfficiencyOneEventChart();
				} else if (arg.equals(SPEEDUP_CHART)) {
					if (checkAndSetMetricName(false))
						PerfExplorerChart.doSpeedupChart();
				} else if (arg.equals(SPEEDUP_EVENTS_CHART)) {
					if (checkAndSetMetricName(false))
						PerfExplorerChart.doSpeedupEventsChart();
				} else if (arg.equals(SPEEDUP_ONE_EVENT_CHART)) {
					if (checkAndSetMetricName(false))
						if (checkAndSetEventName(false))
							PerfExplorerChart.doSpeedupOneEventChart();
				} else if (arg.equals(COMMUNICATION_CHART)) {
					if (checkAndSetMetricName(false))
						if (checkAndSetGroupName(false))
							PerfExplorerChart.doCommunicationChart();
				} else if (arg.equals(FRACTION_CHART)) {
					if (checkAndSetMetricName(false))
						PerfExplorerChart.doFractionChart();
				} else if (arg.equals(EFFICIENCY_PHASE_CHART)) {
					if (checkAndSetMetricName(false))
						PerfExplorerChart.doEfficiencyPhasesChart();
				} else if (arg.equals(SPEEDUP_PHASE_CHART)) {
					if (checkAndSetMetricName(false))
						PerfExplorerChart.doSpeedupPhasesChart();
				} else if (arg.equals(FRACTION_PHASE_CHART)) {
					if (checkAndSetMetricName(false))
						PerfExplorerChart.doFractionPhasesChart();
			// view items
				} else if (arg.equals(CREATE_NEW_VIEW)) {
					int parent = 0; // no parent
					PerfExplorerViews.createNewView(mainFrame, parent);
				} else if (arg.equals(CREATE_NEW_SUB_VIEW)) {
					PerfExplorerViews.createNewSubView(mainFrame);
				} else {
					System.out.println("unknown event! " + arg);
				}
			}
		} catch (Exception e) {
			System.err.println("actionPerformed Exception: " + e.getMessage());
			e.printStackTrace();
		} 
	}

	public void createAboutWindow() {
		JOptionPane.showMessageDialog(mainFrame, "PerfExplorer Client",
			"About PerfExplorer", JOptionPane.PLAIN_MESSAGE);
	}

	public void createHelpWindow() {
		JOptionPane.showMessageDialog(mainFrame, 
			"If this were a real application, you would get some help here...",
			"PerfExplorer Help", JOptionPane.PLAIN_MESSAGE);
	}

	public void createMethodWindow() {
		Object[] options = PerfExplorerModel.getClusterMethods();
		String reply = (String)JOptionPane.showInputDialog (mainFrame,
			"Select a cluster method:",
			"Cluster Method",
			JOptionPane.PLAIN_MESSAGE,
			null,
			options,
			PerfExplorerModel.K_MEANS);
		PerfExplorerModel.getModel().setClusterMethod(reply);
	}

	public void createDimensionWindow() {
		Object[] options = PerfExplorerModel.getDimensionReductions();
		String reply = (String)JOptionPane.showInputDialog (mainFrame,
			"Select a dimension reduction method:",
			"Dimension Reduction",
			JOptionPane.PLAIN_MESSAGE,
			null,
			options,
			PerfExplorerModel.NONE);
		PerfExplorerModel.getModel().setDimensionReduction(reply);
		if (PerfExplorerModel.getModel().getDimensionReduction().equals(reply)) {
			reply = (String)JOptionPane.showInputDialog (mainFrame,
				"Only select events with exclusive time % greater than X:\n(where 0 <= X < 100)",
				"Minimum Percentage", JOptionPane.PLAIN_MESSAGE);
			if (reply != null)
				PerfExplorerModel.getModel().setXPercent(reply);
		}
	}

	public void createNormalizationWindow() {
		Object[] options = PerfExplorerModel.getNormalizations();
		String reply = (String)JOptionPane.showInputDialog (mainFrame,
			"Select a normalization method:",
			"Normalization",
			JOptionPane.PLAIN_MESSAGE,
			null,
			options,
			PerfExplorerModel.NONE);
		PerfExplorerModel.getModel().setNormalization(reply);
	}

	public void createClusterSizeWindow() {
		String numClusters = (new Integer(PerfExplorerModel.getModel().getNumberOfClusters())).toString();
		String reply = (String)JOptionPane.showInputDialog (mainFrame,
			"Enter the max number of clusters (<= " + PerfExplorerModel.MAX_CLUSTERS + "):",
			"Max Clusters", JOptionPane.PLAIN_MESSAGE);
		if (reply != null)
			PerfExplorerModel.getModel().setNumberOfClusters(reply);
	}

	public void createDoClusteringWindow() {
		PerfExplorerModel theModel = PerfExplorerModel.getModel();
		Object selection = theModel.getCurrentSelection();
		String status = null;
		if (selection instanceof Application) {
			int reply = getConfirmation(theModel);
			if (reply == 1) {
				Application application = (Application)selection;
				PerfExplorerConnection server = PerfExplorerConnection.getConnection();
				ListIterator experiments = server.getExperimentList(application.getID());
				Experiment experiment = null;
				boolean failed = false;
				while (experiments.hasNext() && !failed) {
					experiment = (Experiment) experiments.next();
					theModel.setCurrentSelection(experiment);
					ListIterator trials = server.getTrialList(experiment.getID());
					Trial trial = null;
					while (trials.hasNext() && !failed) {
						trial = (Trial) trials.next();
						theModel.setCurrentSelection(trial);
						List metrics = trial.getMetrics();
						for (int i = 0; i < metrics.size(); i++) {
							Object metric = metrics.get(i);
							theModel.setCurrentSelection(metric);
							// request some analysis!
							status = server.requestAnalysis(theModel, true);
							if (!status.endsWith("Request accepted.")) {
								JOptionPane.showMessageDialog(mainFrame, "Request Status: \n" + status,
									"Request Status", JOptionPane.ERROR_MESSAGE);
								failed = true;
								break;
							}
						}
					}
					// set the selection back to experiment
					theModel.setCurrentSelection(experiment);
				}
				// set the selection back to application
				theModel.setCurrentSelection(application);
				if (status.endsWith("Request accepted.")) {
					JOptionPane.showMessageDialog(mainFrame, "Request Status: \n" + status,
						"Request Status", JOptionPane.PLAIN_MESSAGE);
				}
			}
		} else if (selection instanceof Experiment) {
			int reply = getConfirmation(theModel);
			if (reply == 1) {
				Experiment experiment = (Experiment)selection;
				PerfExplorerConnection server = PerfExplorerConnection.getConnection();
				ListIterator trials = server.getTrialList(experiment.getID());
				Trial trial = null;
				boolean failed = false;
				while (trials.hasNext() && !failed) {
					trial = (Trial) trials.next();
					theModel.setCurrentSelection(trial);
					List metrics = trial.getMetrics();
					for (int i = 0; i < metrics.size(); i++) {
						Object metric = metrics.get(i);
						theModel.setCurrentSelection(metric);
						// request some analysis!
						status = server.requestAnalysis(theModel, true);
						if (!status.endsWith("Request accepted.")) {
							JOptionPane.showMessageDialog(mainFrame, "Request Status: \n" + status,
								"Request Status", JOptionPane.ERROR_MESSAGE);
							failed = true;
							break;
						}
					}
				}
				// set the selection back to experiment
				theModel.setCurrentSelection(experiment);
				if (status.endsWith("Request accepted.")) {
					JOptionPane.showMessageDialog(mainFrame, "Request Status: \n" + status,
						"Request Status", JOptionPane.PLAIN_MESSAGE);
				}
			}
		} else if (selection instanceof Trial) {
		/*
			JOptionPane.showMessageDialog(mainFrame, "Please select a Metric.",
				"Selection Error", JOptionPane.ERROR_MESSAGE);
				*/
			int reply = getConfirmation(theModel);
			if (reply == 1) {
				PerfExplorerConnection server = PerfExplorerConnection.getConnection();
				RMIPerfExplorerModel modelCopy = theModel.copy();
				status = server.requestAnalysis(modelCopy, true);
				Trial trial = (Trial)selection;
				List metrics = trial.getMetrics();
				for (int i = 0; i < metrics.size(); i++) {
					modelCopy = theModel.copy();
					Object metric = metrics.get(i);
					modelCopy.setCurrentSelection(metric);
					// request some analysis!
					status = server.requestAnalysis(modelCopy, true);
					if (!status.endsWith("Request accepted.")) {
						JOptionPane.showMessageDialog(mainFrame, "Request Status: \n" + status,
							"Request Status", JOptionPane.ERROR_MESSAGE);
						break;
					}
				}
				if (status.endsWith("Request accepted.")) {
					JOptionPane.showMessageDialog(mainFrame, "Request Status: \n" + status,
						"Request Status", JOptionPane.PLAIN_MESSAGE);
				}
			}
		} else if (selection instanceof Metric) {
			int reply = getConfirmation(theModel);
			if (reply == 1) {
				// request some analysis!
				PerfExplorerConnection server = PerfExplorerConnection.getConnection();
				status = server.requestAnalysis(PerfExplorerModel.getModel(), true);
				if (status.endsWith("Request accepted.")) {
					JOptionPane.showMessageDialog(mainFrame, "Request Status: \n" + status,
						"Request Status", JOptionPane.PLAIN_MESSAGE);
				} else {
					JOptionPane.showMessageDialog(mainFrame, "Request Status: \n" + status,
						"Request Status", JOptionPane.ERROR_MESSAGE);
				}
			}
		}
		return;
	}

	public void createDoCorrelationWindow() {
		PerfExplorerModel theModel = PerfExplorerModel.getModel();
		Object selection = theModel.getCurrentSelection();
		String status = null;
		if ((selection instanceof Trial) || (selection instanceof Metric)) {
			String tmp = theModel.getClusterMethod();
			RMIPerfExplorerModel modelCopy = theModel.copy();
			modelCopy.setClusterMethod(PerfExplorerModel.CORRELATION_ANALYSIS);
			int reply = getConfirmation(modelCopy);
			if (reply == 1) {
				// request some analysis!
				PerfExplorerConnection server = PerfExplorerConnection.getConnection();
				status = server.requestAnalysis(modelCopy, true);
				if (status.endsWith("Request accepted.")) {
					JOptionPane.showMessageDialog(mainFrame, "Request Status: \n" + status,
						"Request Status", JOptionPane.PLAIN_MESSAGE);
				} else {
					JOptionPane.showMessageDialog(mainFrame, "Request Status: \n" + status,
						"Request Status", JOptionPane.ERROR_MESSAGE);
				}
			}
		}
		return;
	}

	private int getConfirmation(RMIPerfExplorerModel theModel) {
		Object [] options = { "No, not yet" , "Yes, do analysis" };
		StringBuffer buf = new StringBuffer();
		buf.append("Analysis method: " + theModel.getClusterMethod());
		buf.append("\nDimension Reduction: " + theModel.getDimensionReduction());
		if (theModel.getDimensionReduction().equals(PerfExplorerModel.OVER_X_PERCENT)) 
			buf.append("\n\t\t Minimum percentage: " + theModel.getXPercent());
		buf.append("\nNormalization: " + theModel.getNormalization());
		if (!theModel.getClusterMethod().equals(PerfExplorerModel.CORRELATION_ANALYSIS))
			buf.append("\nMax Clusters: " + theModel.getNumberOfClusters());
		buf.append("\nTrial: " + theModel.toString());
		buf.append("\n\nPerform clustering with the these options?");
		int reply = JOptionPane.showOptionDialog(mainFrame, buf.toString(),
			"Confirm Analysis",
			JOptionPane.YES_NO_OPTION, 
			JOptionPane.PLAIN_MESSAGE,
			null, 
			options, 
			options[1]);
		return reply;
	}

	private boolean validSelection (PerfExplorerModel theModel) {
		Object selection = theModel.getCurrentSelection();
		// allow Experiments or Trials or 1 view
		if (!(selection instanceof Experiment) && !(selection instanceof Trial) && !(selection instanceof RMIView)) {
			JOptionPane.showMessageDialog(mainFrame, "Please select one or more Experiments or Trials.",
				"Selection Error", JOptionPane.ERROR_MESSAGE);
			return false;
		}
		// check multi-selections, to make sure they are homogeneous
		return true;
	}

	private boolean checkAndSetMetricName (boolean forceIt) {
		//TODO - MAKE SURE THE METRIC EXISTS IN THE SELECTED TRIALS!
		PerfExplorerModel theModel = PerfExplorerModel.getModel();
		if (!validSelection(theModel))
			return false;
		String metric = theModel.getMetricName();
		if (forceIt || (metric == null)) {
			PerfExplorerConnection server = PerfExplorerConnection.getConnection();
			List metrics = server.getPotentialMetrics(theModel);
			Object[] options = metrics.toArray();
			if (options.length > 0) {
				metric = (String)JOptionPane.showInputDialog (mainFrame,
					"Please enter the metric of interest",
					"Metric of interest", JOptionPane.PLAIN_MESSAGE,
					null, options,options[0]);
				theModel.setMetricName(metric);
			}
		}
		return (!forceIt && metric == null) ? false : true;
	}

	private boolean checkAndSetGroupName (boolean forceIt) {
		//TODO - MAKE SURE THE GROUP EXISTS IN THE SELECTED TRIALS!
		PerfExplorerModel theModel = PerfExplorerModel.getModel();
		String group = theModel.getGroupName();
		if (forceIt || group == null) {
			PerfExplorerConnection server = PerfExplorerConnection.getConnection();
			List metrics = server.getPotentialGroups(theModel);
			Object[] options = metrics.toArray();
			if (options.length > 0) {
				group = (String)JOptionPane.showInputDialog (mainFrame,
					"Please enter the group of interest",
					"Group of interest", JOptionPane.PLAIN_MESSAGE,
					null, options, options[0]);
				theModel.setGroupName(group);
			}
		}
		return (!forceIt && group == null) ? false : true;
	}

	private boolean checkAndSetEventName (boolean forceIt) {
		//TODO - MAKE SURE THE EVENT EXISTS IN THE SELECTED TRIALS!
		PerfExplorerModel theModel = PerfExplorerModel.getModel();
		String event = theModel.getEventName();
		if (forceIt || event == null) {
			PerfExplorerConnection server = PerfExplorerConnection.getConnection();
			List metrics = server.getPotentialEvents(theModel);
			Object[] options = metrics.toArray();
			if (options.length > 0) {
				event = (String)JOptionPane.showInputDialog (mainFrame,
					"Please enter the event of interest",
					"Event of interest", JOptionPane.PLAIN_MESSAGE,
					null, options, options[0]);
				theModel.setEventName(event);
			}
		}
		return (!forceIt && event == null) ? false : true;
	}

	private boolean checkAndSetTimesteps (boolean forceIt) {
		PerfExplorerModel theModel = PerfExplorerModel.getModel();
		String timesteps = theModel.getTotalTimesteps();
		if (forceIt || timesteps == null) {
			timesteps = (String)JOptionPane.showInputDialog (mainFrame,
				"Please enter the total number of timesteps for the experiment",
				"Total Timesteps", JOptionPane.PLAIN_MESSAGE);
			theModel.setTotalTimesteps(timesteps);
		}
		return (!forceIt && timesteps == null) ? false : true;
	}
}
