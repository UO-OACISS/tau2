package client;

import common.*;
import server.AnalysisTaskWrapper;

import jargs.gnu.CmdLineParser;
import edu.uoregon.tau.perfdmf.*;

import javax.swing.*;

import java.awt.*;
import java.awt.event.*;
import java.util.*;

public class TestHarness {
	private static String USAGE = "Usage: TestHarness [{-h,--help}] {-c,--configfile}=<config_file> [{-s,--standalone}] [{-e,--engine}=<analysis_engine>]\n  where analysis_engine = R or Weka";

	private TestHarness (boolean standalone, String configFile,
		int analysisEngine, boolean quiet) {

		JFrame frame = new PerfExplorerClient(standalone, configFile, 
			analysisEngine, quiet);
		frame.pack();
		frame.setVisible(true);

		PerfExplorerConnection connection = PerfExplorerConnection.getConnection();
		try {
			// test the connection
			System.out.println(connection.sayHello());

			
			// get lists of things. And spit them out.
			Application app = null;
			System.out.println("******** Applications *********");
			for (ListIterator apps = connection.getApplicationList(); 
				apps.hasNext() ; ) {
				app = (Application)apps.next();
				System.out.println(app.getName());
				//if (app.getName().equalsIgnoreCase("sweep3d")) {
				if (app.getName().equalsIgnoreCase("gyro-b1")) {
					break;
				}
			}

			Experiment exp = null;
			System.out.println("******** Experiments *********");
			for (ListIterator exps = connection.getExperimentList(app.getID()); 
				exps.hasNext() ; ) {
				exp = (Experiment)exps.next();
				System.out.println(exp.getName());
				//if (exp.getName().equalsIgnoreCase("150.1 Strong Scaling 1")) {
				if (exp.getName().equalsIgnoreCase("tg")) {
					break;
				}
			}

/*
			Trial trial = null;
			System.out.println("******** Trials *********");
			for (ListIterator trials = connection.getTrialList(2); 
				trials.hasNext() ; ) {
				trial = (Trial)trials.next();
				if (trial.getName().equalsIgnoreCase("32")) {
					System.out.println(trial.getName());
					break;
				}
			}

			Metric metric = null;
			Vector metrics = trial.getMetrics();
			for (int i = 0 ; i < metrics.size() ; i++) {
				metric = (Metric)metrics.elementAt(i);
				if (metric.getName().equalsIgnoreCase("time")) {
					System.out.println(metric.getName());
					break;
				}
			}
*/

			RMIPerfExplorerModel model = PerfExplorerModel.getModel();

			// select the application, experiment, trial, metric we want
			Object[] objects = new Object[4];
			objects[0] = app;
			objects[1] = exp;
			//objects[2] = trial;
			//objects[3] = metric;
			model.setCurrentSelection(objects);
			//model.setMetricName("Time");
			//model.setEventName("MPI_Recv()");
			//model.setGroupName("MPI");
			//model.setTotalTimesteps("12");
			//model.setConstantProblem(true);
			model.setMetricName("Time");
			model.setEventName("field");
			model.setGroupName("TRANSPOSE");
			model.setTotalTimesteps("100");
			model.setConstantProblem(true);

			PerfExplorerChart.doTotalTimeChart();
			PerfExplorerChart.doTimestepsChart();
			PerfExplorerChart.doEfficiencyChart();
			PerfExplorerChart.doSpeedupChart();
			PerfExplorerChart.doEfficiencyOneEventChart();
			PerfExplorerChart.doSpeedupOneEventChart();
			PerfExplorerChart.doEfficiencyEventsChart();
			PerfExplorerChart.doSpeedupEventsChart();
			PerfExplorerChart.doCommunicationChart();
			PerfExplorerChart.doFractionChart();
			PerfExplorerChart.doCorrelationChart();
			PerfExplorerChart.doEfficiencyPhasesChart();
			PerfExplorerChart.doSpeedupPhasesChart();
			PerfExplorerChart.doFractionPhasesChart();

/*
			//model.setClusterMethod(RMIPerfExplorerModel.CORRELATION_ANALYSIS);
			model.setDimensionReduction(RMIPerfExplorerModel.OVER_X_PERCENT);
			model.setNumberOfClusters("10");
			model.setXPercent("2");
			String status = connection.requestAnalysis(model, true);
			System.out.println(status);
			if (status.equalsIgnoreCase("Request already exists"))
				System.out.println(connection.requestAnalysis(model, true));
			java.lang.Thread.sleep(100000);
*/
			/*
	public RMIPerformanceResults getPerformanceResults(RMIPerfExplorerModel model) throws RemoteException;
	public List getPotentialGroups(RMIPerfExplorerModel model) throws RemoteException;
	public List getPotentialMetrics(RMIPerfExplorerModel model) throws RemoteException;
	public List getPotentialEvents(RMIPerfExplorerModel model) throws RemoteException;
	public String[] getMetaData(String tableName) throws RemoteException;
	public List getPossibleValues(String tableName, String columnName) throws RemoteException;
	public int createNewView(String name, int parent, String tableName, String columnName, String oper, String value) throws RemoteException;
	public List getViews(int parent) throws RemoteException;
	public List getTrialsForView(List views) throws RemoteException;
	public RMIPerformanceResults getCorrelationResults(RMIPerfExplorerModel model) throws RemoteException;
	public RMIVarianceData getVariationAnalysis(RMIPerfExplorerModel model) throws RemoteException;
	public RMICubeData getCubeData(RMIPerfExplorerModel model) throws RemoteException;
	*/
			//java.lang.Thread.sleep(1000);
		} catch (Exception e) {
			System.err.println("TestHarness exception: " + e.getMessage());
			e.printStackTrace();
		}
		/*
		finally {
			System.out.println("Shutting down server...");
			connection.stopServer();
			try {
				java.lang.Thread.sleep(1000);
			} catch (Exception e2) {
				System.err.println("TestHarness exception: " + e2.getMessage());
				e2.printStackTrace();
			}
			System.out.println("Exiting...");
			System.exit(0);
		}
		*/
	}

	public static void main (String[] args) {

		// set the tooltip delay to 20 seconds
		ToolTipManager.sharedInstance().setDismissDelay(20000);

		// Process the command line
		CmdLineParser parser = new CmdLineParser();
		CmdLineParser.Option helpOpt = parser.addBooleanOption('h',"help");
		CmdLineParser.Option standaloneOpt = parser.addBooleanOption('s',"standalone");
		CmdLineParser.Option configfileOpt = parser.addStringOption('c',"configfile");
		CmdLineParser.Option engineOpt = parser.addStringOption('e',"engine");
		CmdLineParser.Option quietOpt = parser.addBooleanOption('q',"quiet");

		try {
			parser.parse(args);
		} catch (CmdLineParser.OptionException e) {
			System.err.println(e.getMessage());
			System.err.println(USAGE);
			System.exit(-1);
		}   
		
		Boolean help = (Boolean) parser.getOptionValue(helpOpt);
		Boolean standalone = (Boolean) parser.getOptionValue(standaloneOpt);
		String configFile = (String) parser.getOptionValue(configfileOpt);
		String engine = (String) parser.getOptionValue(engineOpt);
		Boolean quiet = (Boolean) parser.getOptionValue(quietOpt);

		int analysisEngine = AnalysisTaskWrapper.WEKA_ENGINE;

		if (help != null && help.booleanValue()) {
			System.err.println(USAGE);
			System.exit(-1);
		}

		if (quiet == null) 
			quiet = new Boolean(false);

		if (standalone == null) 
			standalone = new Boolean(false);

		if (standalone.booleanValue()) {
			if (configFile == null) {
				System.err.println("Please enter a valid config file.");
				System.err.println(USAGE);
				System.exit(-1);
			}
			if (engine == null) {
				System.err.println("Please enter a valid engine type.");
				System.err.println(USAGE);
				System.exit(-1);
			} else if (engine.equalsIgnoreCase("R")) {
				analysisEngine = AnalysisTaskWrapper.RPROJECT_ENGINE;
			} else if (engine.equalsIgnoreCase("weka")) {
				analysisEngine = AnalysisTaskWrapper.WEKA_ENGINE;
			} else {
				System.err.println(USAGE);
				System.exit(-1);
			}
		}


	/*
		try {
			UIManager.setLookAndFeel(
				UIManager.getCrossPlatformLookAndFeelClassName());
		} catch (Exception e) { }
	*/

		TestHarness harness = new TestHarness(standalone.booleanValue(),
			configFile, analysisEngine, quiet.booleanValue());
	}

}
