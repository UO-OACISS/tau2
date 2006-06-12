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
	private static String USAGE = "Usage: TestHarness [{-h,--help}] {-c,--configfile}=<config_file> [{-s,--standalone}] [{-e,--engine}=<analysis_engine>] [{-t,--test}=<test_type>]\n  where analysis_engine = R or Weka and test_type = charts, cluster, correlation, viz or all ";
	private PerfExplorerConnection connection = null;
	private RMIPerfExplorerModel model = null;
	private Object[] viewList = null;
	private boolean foundView = false;

	private TestHarness (boolean standalone, String configFile,
		int analysisEngine, boolean quiet, String test) {

		JFrame frame = new PerfExplorerClient(standalone, configFile, 
			analysisEngine, quiet);
		frame.pack();
		frame.setVisible(true);

		// get the connection to the server object
		connection = PerfExplorerConnection.getConnection();
		// get the tree model
		model = PerfExplorerModel.getModel();

		try {
			// test the connection
			System.out.println(connection.sayHello());

			if (test.equals("chart") || test.equals("all")) {
				// derby settings
				//setSelection("gyro-b1", "tg", null, null);
				//model.setMetricName("Time");
		
				// postgres settings
				//setSelection("gyro.B1-std", "B1-std.tg", null, null);
				//model.setMetricName("WALL_CLOCK_TIME");
				//model.setEventName("field");
				//model.setGroupName("TRANSPOSE");
				//model.setTotalTimesteps("100");
				//model.setConstantProblem(true);

				// DB2 settings
				setSelection("FLASH", "hydro radiation scaling on BG/L", null, null);
				model.setMetricName("Time");
				model.setEventName("MPI_Barrier()");
				model.setGroupName("MPI");
				model.setTotalTimesteps("100");
				model.setConstantProblem(false);

				System.out.println("Testing charts...");
				testCharts();
			}
			if (test.equals("viz") || test.equals("all")) {
				System.out.println("Testing visualization...");
				// derby/postgres
				//setSelection("sweep3d", "150.1 Strong Scaling 1", "128", "time");
				setSelection("sweep3d", "150.1 Strong Scaling 2", "128", "time");

				// DB2
				//setSelection("FLASH", "hydro radiation scaling on BG/L", "tau64p.ppk/64p/scaling/hydro-radiation-scaling/flash/flash/taudata/packages/disk2/", "Time");

				testVisualization();
			}
			if (test.equals("views") || test.equals("all")) {
				System.out.println("Testing views...");
				viewList = null;
				foundView = false;
				getViews("0", new ArrayList());
				System.out.print("Selecting: ");
				for(int i = 0 ; i < viewList.length ; i++) 
					System.out.print(viewList[i] + ": ");
				System.out.println("");
				model.setCurrentSelection(viewList);
				model.setMetricName("Time");
				model.setEventName("field");
				model.setGroupName("TRANSPOSE");
				model.setTotalTimesteps("100");
				model.setConstantProblem(true);
				testViews();
			}
			if (test.equals("cluster") || test.equals("all")) {
				System.out.println("Testing clustering...");
				//setSelection("sweep3d", "150.1 Strong Scaling 2", "32", "time");
				setSelection("FLASH", "hydro radiation scaling on BG/L", "tau64p.ppk/64p/scaling/hydro-radiation-scaling/flash/flash/taudata/packages/disk2/", "Time");

				testCluster();
			}
			if (test.equals("correlation") || test.equals("all")) {
				System.out.println("Testing correlation...");
				//setSelection("sweep3d", "150.1 Strong Scaling 2", "32", "time");
				setSelection("FLASH", "hydro radiation scaling on BG/L", "tau64p.ppk/64p/scaling/hydro-radiation-scaling/flash/flash/taudata/packages/disk2/", "Time");

				testCorrelation();
			}
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

	private Application setApplication(String name) {
		Application app = null;
		System.out.println("******** Applications *********");
		for (ListIterator apps = connection.getApplicationList(); 
			apps.hasNext() ; ) {
			app = (Application)apps.next();
			System.out.println("\t" + app.getName());
			if (app.getName().equalsIgnoreCase(name)) {
				break;
			}
		}
		return app;
	}

	private Experiment setExperiment(Application app, String name) {
		Experiment exp = null;
		System.out.println("******** Experiments *********");
		for (ListIterator exps = connection.getExperimentList(app.getID()); 
			exps.hasNext() ; ) {
			exp = (Experiment)exps.next();
			System.out.println("\t" + exp.getName());
			if (exp.getName().equalsIgnoreCase(name)) {
				break;
			}
		}
		return exp;
	}

	private Trial setTrial(Experiment exp, String name) {
		Trial trial = null;
		System.out.println("******** Trials *********");
		for (ListIterator trials = connection.getTrialList(exp.getID()); 
			trials.hasNext() ; ) {
			trial = (Trial)trials.next();
			System.out.println("\t" + trial.getName());
			if (trial.getName().trim().equalsIgnoreCase(name)) {
				break;
			}
		}
		return trial;
	}

	private Metric setMetric(Trial trial, String name) {
		Metric metric = null;
		System.out.println("******** Metrics *********");
		Vector metrics = trial.getMetrics();
		for (int i = 0 ; i < metrics.size() ; i++) {
			metric = (Metric)metrics.elementAt(i);
			System.out.println("\t" + metric.getName());
			if (metric.getName().equalsIgnoreCase(name)) {
				break;
			}
		}
		return metric;
	}

	public void setSelection(String appName, String expName, String trialName, String metricName) {
		Object[] objects = new Object[4];

		Application app = setApplication(appName);
		objects[0] = app;

		if (expName != null) {
			Experiment exp = setExperiment(app, expName);
			objects[1] = exp;

			if (trialName != null) {
				Trial trial = setTrial(exp, trialName);
				objects[2] = trial;

				if (metricName != null) {
					Metric metric = setMetric(trial, metricName);
					objects[3] = metric;
				}
			}
		}

		// select the application, experiment, trial, metric we want
		model.setCurrentSelection(objects);
	}

	public void testCharts() throws Exception {
		// do the tests
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
	}

	public void testVisualization() throws Exception {
	/*
		PerfExplorerCube.doCorrelationCube();
		PerfExplorerVariation.doVariationAnalysis();
		PerfExplorerBoxChart.doIQRBoxChart();
		PerfExplorerHistogramChart.doHistogram();
		*/
		PerfExplorerProbabilityPlot.doProbabilityPlot();
	}

	public void testViews() throws Exception {
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
//public int createNewView(String name, int parent, String tableName, String columnName, String oper, String value) throws RemoteException;
	}

	public void getViews(String parent, ArrayList list) throws Exception {
        java.util.List viewVector = connection.getViews(Integer.parseInt(parent));
        Iterator views = viewVector.iterator();
        while (views.hasNext()) {
            RMIView view = (RMIView) views.next();
			System.out.println("VIEW: " + view.getField("NAME"));
			//if (view.getField("VALUE").equals("gyro-b1"))
			//if (view.getField("VALUE").equals("B1-std.tg"))
			if (view.getField("VALUE").equals("gyro.B1-std"))
				foundView = true;
			ArrayList newList = new ArrayList(list);
			newList.add(view);
            getViews(view.getField("ID"), newList);
        }
        if (viewVector.size() == 0) {
			if (viewList == null && foundView) {
				viewList = list.toArray();
			}
            ListIterator trials = connection.getTrialsForView(list);
            Trial trial = null;
            while(trials.hasNext())
            {
                trial = (Trial) trials.next();
				System.out.println("TRIAL: " + trial.getName());
            }
        }
	}

	public void testCluster() throws Exception {
		model.setDimensionReduction(RMIPerfExplorerModel.OVER_X_PERCENT);
		model.setNumberOfClusters("10");
		model.setXPercent("2");
		String status = connection.requestAnalysis(model, true);
		System.out.println(status);
		if (status.equalsIgnoreCase("Request already exists"))
			System.out.println(connection.requestAnalysis(model, true));
		java.lang.Thread.sleep(5000);
	}

	public void testCorrelation() throws Exception {
		model.setClusterMethod(RMIPerfExplorerModel.CORRELATION_ANALYSIS);
		model.setDimensionReduction(RMIPerfExplorerModel.OVER_X_PERCENT);
		model.setXPercent("2");
		String status = connection.requestAnalysis(model, true);
		System.out.println(status);
		if (status.equalsIgnoreCase("Request already exists"))
			System.out.println(connection.requestAnalysis(model, true));
		java.lang.Thread.sleep(5000);
	}

/*
public RMIPerformanceResults getPerformanceResults(RMIPerfExplorerModel model) throws RemoteException;
public List getPotentialGroups(RMIPerfExplorerModel model) throws RemoteException;
public List getPotentialMetrics(RMIPerfExplorerModel model) throws RemoteException;
public List getPotentialEvents(RMIPerfExplorerModel model) throws RemoteException;
public String[] getMetaData(String tableName) throws RemoteException;
public List getPossibleValues(String tableName, String columnName) throws RemoteException;
public RMIPerformanceResults getCorrelationResults(RMIPerfExplorerModel model) throws RemoteException;
public RMIVarianceData getVariationAnalysis(RMIPerfExplorerModel model) throws RemoteException;
public RMICubeData getCubeData(RMIPerfExplorerModel model) throws RemoteException;
*/

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
		CmdLineParser.Option testOpt = parser.addStringOption('t',"test");

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
		String test = (String) parser.getOptionValue(testOpt);

		int analysisEngine = AnalysisTaskWrapper.WEKA_ENGINE;

		if (help != null && help.booleanValue()) {
			System.err.println(USAGE);
			System.exit(-1);
		}

		if (quiet == null) 
			quiet = new Boolean(false);

		if (standalone == null) 
			standalone = new Boolean(false);

		if (test == null)
			test = new String("all");
		else
			test = test.toLowerCase();

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

		if (!test.equals("chart") && !test.equals("cluster") &&
			!test.equals("viz") && !test.equals("correlation") &&
			!test.equals("views") && !test.equals("all")) {
			System.err.println("Please enter a valid test.");
			System.err.println(USAGE);
			System.exit(-1);
		}


	/*
		try {
			UIManager.setLookAndFeel(
				UIManager.getCrossPlatformLookAndFeelClassName());
		} catch (Exception e) { }
	*/

		TestHarness harness = new TestHarness(standalone.booleanValue(),
			configFile, analysisEngine, quiet.booleanValue(), test);
	}

}
