package edu.uoregon.tau.perfexplorer.server;


import jargs.gnu.CmdLineParser;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.rmi.ConnectException;
import java.rmi.Naming;
import java.rmi.RMISecurityManager;
import java.rmi.RemoteException;
import java.rmi.ServerError;
import java.rmi.ServerException;
import java.rmi.StubNotFoundException;
import java.rmi.server.UnicastRemoteObject;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.DBDataSource;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.Database;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.taudb.TAUdbDataSource;
import edu.uoregon.tau.perfdmf.taudb.TAUdbDatabaseAPI;
import edu.uoregon.tau.perfexplorer.common.AnalysisType;
import edu.uoregon.tau.perfexplorer.common.ChartDataType;
import edu.uoregon.tau.perfexplorer.common.ChartType;
import edu.uoregon.tau.perfexplorer.common.ConfigureFiles;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerException;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;
import edu.uoregon.tau.perfexplorer.common.RMIChartData;
import edu.uoregon.tau.perfexplorer.common.RMICubeData;
import edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData;
import edu.uoregon.tau.perfexplorer.common.RMIPerfExplorer;
import edu.uoregon.tau.perfexplorer.common.RMIPerfExplorerModel;
import edu.uoregon.tau.perfexplorer.common.RMIPerformanceResults;
import edu.uoregon.tau.perfexplorer.common.RMISortableIntervalEvent;
import edu.uoregon.tau.perfexplorer.common.RMIVarianceData;
import edu.uoregon.tau.perfexplorer.constants.Constants;

/**
 * The main PerfExplorer Server class.  This class is defined as a singleton,
 * with multiple threads.  The main thread processes interactive requests, 
 * and the background thread processes long-running analysis requests.
 * This server is accessed through RMI, and objects are passed back and forth
 * over the RMI link to the client.
 *
 * <P>CVS $Id: PerfExplorerServer.java,v 1.92 2009/11/19 15:53:33 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class PerfExplorerServer extends UnicastRemoteObject implements RMIPerfExplorer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6257362740066518307L;
	private static String USAGE = "Usage: PerfExplorerClient [{-h,--help}] {-c,--configfile}=<config_file> [{-p,--port}=<port_number>]\n ";
	private DatabaseAPI session = null;
	private List<DatabaseAPI> sessions = new ArrayList<DatabaseAPI>();
	private List<String> configNames = new ArrayList<String>();
	private List<String> sessionStrings = new ArrayList<String>();
	private List<Integer> schemaVersions = new ArrayList<Integer>();
	private List<Boolean> connected = new ArrayList<Boolean>();
	private List<String> actualConfigFiles = new ArrayList<String>();
	private List<Queue<RMIPerfExplorerModel>> requestQueues = new ArrayList<Queue<RMIPerfExplorerModel>>();
	private List<Thread> timerThreads = new ArrayList<Thread>();
	private List<TimerThread> timers = new ArrayList<TimerThread>();
	private static PerfExplorerServer theServer = null;
	private String configFile;
	private String tauHome = "";
	private String tauArch = "";

	/**
	 * Static method to get the server instance reference.
	 * Note: This method assumes the server has been instantiated.
	 * 
	 * @return
	 */
	public static PerfExplorerServer getServer () {
		return theServer;
	}
	
	public DatabaseAPI getSession() {
		return session;
	}
	
	public DatabaseAPI getSession(Database database){
		if(database==null)
			return null;
		
		Iterator<DatabaseAPI> sit = sessions.iterator();
		while(sit.hasNext()){
			DatabaseAPI next = sit.next();
			if (next != null) {
				if(next.getDb().getDatabase().getID()==database.getID()) {
					return next;
				}
			}
		}
		return null;
	}

	/**
	 * Static method to return the server instance reference.
	 * If the server has not yet been created, it is created.
	 * The passed in configFile is a perfdmf configuration file, 
	 * specifiying the database parameters.
	 * 
	 * @param configFile
	 * @return
	 */
	public static PerfExplorerServer getServer (String configFile, String tauRoot, String tauArch) {
		try {
			if (theServer == null)
				theServer = new PerfExplorerServer (configFile, 0, false, tauRoot, tauArch);
        	//System.out.println("Checking for " + Constants.TMPDIR);

        	File dir = new File(Constants.TMPDIR);
        	if (!dir.exists()) {
            	System.out.println("Temporary directory not found, creating " + Constants.TMPDIR);
            	boolean success = dir.mkdirs();
				if (success) {
            		System.out.println("Created " + Constants.TMPDIR);
				} else {
            		System.err.println("Failed to create " + Constants.TMPDIR);
				}
        	}

		} catch (Exception e) {
			System.err.println("getServer exception: " + e.getMessage());
			e.printStackTrace();
		}
		return theServer;
	}

	public void resetServer () {
		String configFile = theServer.configFile;
		String tauHome = theServer.tauHome;
		String tauArch = theServer.tauArch;
		PerfExplorerServer.theServer = null;
		getServer(configFile, tauHome, tauArch);
	}
	
	public static void hardResetServer(){
		PerfExplorerServer.theServer=null;
	}
	
	/**
	 * Private constructor, which uses the PerfDMF configuration file to
	 * connect to the PerfDMF database.  A semaphore is also created to 
	 * control access to the database connection.
	 * 
	 * @param configFile
	 * @throws RemoteException
	 */
	private PerfExplorerServer(String configFile,
	int port, boolean quiet, String tauHome, String tauArch) throws RemoteException {
		super(port);
		this.configFile = configFile;
		this.tauHome = tauHome;
		this.tauArch = tauArch;
		PerfExplorerOutput.setQuiet(quiet);
		theServer = this;
		boolean makeConnection = false;
		List<String> configFiles = ConfigureFiles.getConfigurationNames();
		if (configFile != null && configFile.equals("NONE")) {
			configFiles = new ArrayList<String>();
			makeConnection = true;
		} else if (configFile != null && configFile.length() > 0) {
			// if the user supplied a config file, use just that one
			configFiles = new ArrayList<String>();
			configFiles.add(configFile);
			addWorkingDatabase(configFiles);
		} else {
			addWorkingDatabase(configFiles);
		}
		for (Iterator<String> iter = configFiles.iterator() ; iter.hasNext() ; ) {
			String tmpFile = iter.next();
			if (tmpFile.contains("NO_DATABASE")) {
				break;
			}
			connectToDatabase(tmpFile, makeConnection, -1);
		}
	}

	/**
	 * @param i
	 * @param api
	 * @param tmpFile
	 * @param index 
	 * @param configName
	 * @return
	 * @throws SQLException
	 */
	private void connectToDatabase(String tmpFile, boolean makeConnection, int index) {
        String home = System.getProperty("user.home");
        String slash = System.getProperty("file.separator");
        String prefix = home + slash + ".ParaProf" + slash + "perfdmf.cfg.";
		// this is how you replace 1 backslash with 2 backslashes for use later 
		// as a regular expression...
		prefix = prefix.replaceAll("\\\\", "\\\\\\\\");
		//System.out.println(prefix);

		try {
			DatabaseAPI api = new DatabaseAPI();
			String configName = tmpFile.replaceAll(prefix, "");
			
			//This isn't needed for the gui-enabled version. If the tmpFile is just the database name adding the config file prefix will make it a valid configuration file.
			if(!new java.io.File(tmpFile).exists()) {
				tmpFile=prefix+tmpFile;
			}
			
			if (makeConnection) {
				PerfExplorerOutput.print("Connecting...");
				api.initialize(tmpFile, false);
				if (api.db().getSchemaVersion() > 0) {
					// copy the DatabaseAPI object data into a new TAUdbDatabaseAPI object
					api = new TAUdbDatabaseAPI(api);
				}
				PerfExplorerOutput.println(" Connected to " + api.db().getConnectString() + ".");
				Queue<RMIPerfExplorerModel> requestQueue = new LinkedList<RMIPerfExplorerModel>();
				TimerThread timer = new TimerThread(this, api, ((index<0)?this.sessions.size():index));
				java.lang.Thread timerThread = new java.lang.Thread(timer);
				if (index < 0) {
					this.sessions.add(api);
					this.connected.add(true);
					this.configNames.add(configName);
					this.sessionStrings.add(api.db().getConnectString());
					this.schemaVersions.add(api.db().getSchemaVersion());
					this.session = api;
					this.actualConfigFiles.add(tmpFile);
					this.requestQueues.add(requestQueue);
					this.timers.add(timer);
					this.timerThreads.add(timerThread);
				} else {
					this.sessions.set(index, api);
					this.connected.set(index, true);
					this.configNames.set(index, configName);
					this.sessionStrings.set(index, api.db().getConnectString());
					this.schemaVersions.set(index, api.db().getSchemaVersion());
					this.session = api;
					this.requestQueues.set(index,requestQueue);
					this.timers.set(index,timer);
					this.timerThreads.set(index,timerThread);
				}
				timerThread.start();
			} else {
				this.sessions.add(null);
				this.connected.add(false);
				this.configNames.add(configName);
				this.sessionStrings.add("jdbc:"+configName);
				this.schemaVersions.add(0);
				this.actualConfigFiles.add(tmpFile);
				this.requestQueues.add(null);
				this.timers.add(null);
				this.timerThreads.add(null);
			}
		} catch (Exception e) {
			if (e instanceof FileNotFoundException) {
				System.err.println(e.getMessage());
			} else {
				System.err.println("Error connecting to " + tmpFile + "!");
				System.err.println(e.getMessage());
        		StringBuilder buf = new StringBuilder();
        		buf.append("\nPlease make sure that your DBMS is ");
        		buf.append("configured correctly, and the database ");
        		buf.append("has been created.");
        		buf.append("\nSee the PerfExplorer and/or PerfDMF");
        		buf.append("configuration utilities for details.\n");
        		System.err.println(buf.toString());
				e.printStackTrace();
				//System.exit(1);
			}
    	}
	}

//	/**
//	 * Return the constructed analysisfactory.
//	 * @return
//	 */
//	public AnalysisFactory getAnalysisFactory() {
//		//System.out.println("getting factory");
//		if (factory == null) {
//        	factory = new AnalysisFactory();
//		}
//		return factory;
//	}
	
	/**
	 * Locate / Add working database.  If there is no working database, create
	 * one, and add its location to the list of databases.
	 * 
	 */
	private void addWorkingDatabase(List<String> configFiles) {
		String newConfig = System.getProperty("user.home") + 
		  File.separator + ".ParaProf" + 
		  File.separator + "perfdmf.cfg." + Constants.PERFEXPLORER_WORKING_CONFIG;
		// make sure the working configuration is the last one in the list
		// by removing it if it exists
		configFiles.remove(newConfig);
//		for (String cf : configFiles) {
//			if (cf.equals(newConfig));
//				configFiles.remove(cf);
//				break;
//		}
		File defaultConfig = new File(newConfig);
		if (!defaultConfig.exists() 
				&& tauHome!=null 
				&& tauArch!=null 
				&& tauHome.length() > 0 
				&& tauArch.length() > 0) {
			try {
				System.out.println("Default working database does not exist, creating...");
				String dbName = System.getProperty("user.home") + 
				  File.separator + ".ParaProf" + 
				  File.separator + Constants.PERFEXPLORER_WORKING_CONFIG;
				edu.uoregon.tau.perfdmf.loader.Configure.createDefault(newConfig, tauHome, tauArch, dbName);
				System.out.println("Configuration file " + newConfig + " created.");
				edu.uoregon.tau.perfexplorer.common.Configure.loadDefaultSchema(newConfig, tauHome, tauArch);
			} catch (Exception perfDMFException) {
				System.err.println(perfDMFException.getMessage());
			}
		}
		configFiles.add(newConfig);
	}

	/**
	 * Test method.
	 * 
	 */
	public String sayHello() {
		PerfExplorerOutput.println("sayHello()...");
		String howdy = new String("Hello, client - this is server!");
		return howdy;
	}

	/**
	 * Requests an ApplicationList object from the PerfDMF database.
	 * 
	 * @return List of PerfDMF Application objects.
	 */
	public List<Application> getApplicationList() {
		//PerfExplorerOutput.println("getApplicationList()...");
		List<Application> applications = this.session.getApplicationList();
		return applications;
	}

	/**
	 * Requests an ExperimentList object from the PerfDMF database,
	 * based on the application id which is passed in.
	 * 
	 * @param applicationID
	 * @return List of PerfDMF Experiment objects.
	 */
	public List<Experiment> getExperimentList(int applicationID) {
		//PerfExplorerOutput.println("getExperimentList(" + applicationID + ")...");
		this.session.setApplication(applicationID);
		List<Experiment> experiments = null;
		try {
			experiments = this.session.getExperimentList();
		} catch (DatabaseException e) {}

		return experiments;
	}

	/**
	 * Requests an ExperimentList object from the PerfDMF database,
	 * based on the application id which is passed in.
	 * 
	 * @param experimentID
	 * @return List of PerfDMF Trial objects.
	 */
	public List<Trial> getTrialList(int experimentID, boolean getXMLMetadata) {
		//PerfExplorerOutput.println("getTrialList(" + experimentID + ")...");
		try {
			this.session.setExperiment(experimentID);
		} catch (DatabaseException e) {}
		List<Trial> trials = this.session.getTrialList(getXMLMetadata);
		return trials;
	}

	public void stopServer() {
		PerfExplorerOutput.println("stopServer()...");
		for (int i = 0 ; i < timers.size(); i++ ) {
			TimerThread timer = timers.get(i);
			timer.cancel();
		}
		try{
			java.lang.Thread.sleep(1000);
		} catch (Exception e) {/* nothing to do */}
		System.exit(0);
	}

	/**
	 * Based on the model passed in over RMI, request some type of analysis
	 * on the settings.  The analysis is a long-running analysis, so it will
	 * be placed in the request queue, and the background thread will process
	 * the request.  If the request already exists, the "force" parameter will
	 * replace the previous analysis request.  The String returned contains
	 * the status of having queued the request.
	 * 
	 * @param model
	 * @param force
	 * @return String 
	 */
	public String requestAnalysis(RMIPerfExplorerModel model, boolean force) {
		StringBuilder status = new StringBuilder();
		//PerfExplorerOutput.println("requestAnalysis(" + model.toString() + ")...");
		try {
			if (!force && checkForRequest(model) != 0) {
				throw new PerfExplorerException("Request already exists");
			}
			int analysisID = insertRequest(model);
			status.append("Request " + analysisID + " queued.");
			model.setAnalysisID(analysisID);
			status.append("\nRequest accepted.");
			Queue<RMIPerfExplorerModel> requestQueue = requestQueues.get(model.getConnectionIndex());
			requestQueue.offer(model);
		} catch (PerfExplorerException e) {
			String tmp = e.getMessage();
			Throwable exec = e.getCause();
			if (exec != null)
				tmp += "\n" + exec.getMessage();
			return tmp;
		}
		return status.toString();
	}

	/**
	 * After Correlation analysis has been requested, the results can be
	 * requested.  Based on the model passed in, if the configuration has been
	 * analyzed, the results are retrieved from the database.  The results
	 * are serialized over the RMI link back to the client application.
	 * 
	 * @param model
	 * @return RMIPerformanceResults
	 */
	public RMIPerformanceResults getCorrelationResults(RMIPerfExplorerModel model) {
		//PerfExplorerOutput.print("getCorrelationResults(" + model.toString() + ")... ");
		RMIPerformanceResults analysisResults = new RMIPerformanceResults();
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			if (model.getCurrentSelection() instanceof Metric) {
				if (db.getSchemaVersion() == 0) {
					statement = db.prepareStatement("select id from analysis_settings where application = ? and experiment = ? and trial = ? and metric = ? and method = ? order by id desc");
					statement.setInt(1, model.getApplication().getID());
					statement.setInt(2, model.getExperiment().getID());
					statement.setInt(3, model.getTrial().getID());
					statement.setInt(4, ((Metric)(model.getCurrentSelection())).getID());
					statement.setString(5, AnalysisType.CORRELATION_ANALYSIS.toString());
				} else {
					statement = db.prepareStatement("select id from analysis_settings where trial = ? and metric = ? and method = ? order by id desc");
					statement.setInt(1, model.getTrial().getID());
					statement.setInt(2, ((Metric)(model.getCurrentSelection())).getID());
					statement.setString(3, AnalysisType.CORRELATION_ANALYSIS.toString());
				}
			}else {
				if (db.getSchemaVersion() == 0) {
					statement = db.prepareStatement("select id from analysis_settings where application = ? and experiment = ? and trial = ? and metric is null and method = ? order by id desc");
					statement.setInt(1, model.getApplication().getID());
					statement.setInt(2, model.getExperiment().getID());
					statement.setInt(3, model.getTrial().getID());
					statement.setString(4, AnalysisType.CORRELATION_ANALYSIS.toString());
				} else {
					statement = db.prepareStatement("select id from analysis_settings where trial = ? and metric is null and method = ? order by id desc");
					statement.setInt(1, model.getTrial().getID());
					statement.setString(2, AnalysisType.CORRELATION_ANALYSIS.toString());			
				}
			}
			//PerfExplorerOutput.println(statement.toString());
			ResultSet results = statement.executeQuery();
			int analysisID = 0;
			if (results.next() != false) {
				analysisID = results.getInt(1);
			} else {
				// no results for this combination
				statement.close();
				return analysisResults;
			}
			statement.close();

			// get the images and thumbnails
			statement = db.prepareStatement("select id, description, thumbnail_size, " + 
					"thumbnail, image_size, image, result_type from analysis_result where " + 
					"analysis_settings = ? and result_type = ? order by id asc");
			statement.setInt(1, analysisID);
			statement.setInt(2,Integer.valueOf(ChartType.CORRELATION_SCATTERPLOT.toString()));
			//statement.setString(2, ChartType.CORRELATION_SCATTERPLOT.toString());
			//PerfExplorerOutput.println(statement.toString());
			results = statement.executeQuery();
			while (results.next() != false) {
				String id = results.getString(1);
				String description = results.getString(2);
				int thumbSize = results.getInt(3);
				InputStream thumbStream = results.getBinaryStream(4);
				byte[] thumbData = new byte[thumbSize];
				//int bytesRead = 
				thumbStream.read(thumbData);
				int imageSize = results.getInt(5);
				InputStream imageStream = results.getBinaryStream(6);
				byte[] imageData = new byte[imageSize];
				//bytesRead = 
				imageStream.read(imageData);
				//String k = results.getString(8);
				analysisResults.getDescriptions().add(description);
				analysisResults.getIDs().add(id);
				//analysisResults.getKs().add(k);
				analysisResults.getThumbnails().add(thumbData);
				analysisResults.getImages().add(imageData);
			}


			/* The following code was commented out for performance reasons.
			 * If the raw performance results are desired, then uncomment
			 * out the following code.  Otherwise, this code will only return
			 * the graphs and charts.
			 */
			
/*			Enumeration e = analysisResults.getIDs().elements();
			while (e.hasMoreElements()) {
				String id = (String) e.nextElement();
				StringBuilder buf = new StringBuilder();
				buf.append("select ie.name, m.name, ard.value, ard.data_type, ard.cluster_index ");
				buf.append("from analysis_result_data ard ");
				buf.append("inner join interval_event ie on ard.interval_event = ie.id ");
				buf.append("inner join metric m on ard.metric = m.id ");
				buf.append("where ard.analysis_result = ? ");
				buf.append("order by ard.cluster_index, ie.name, ard.data_type ");
				statement = db.prepareStatement(buf.toString());
				statement.setString(1, id);
				//PerfExplorerOutput.println(statement.toString());
				results = statement.executeQuery();
				String clusterID = new String("");
				List centroids = null;
				List deviations = null;
				while (results.next() != false) {
					String intervalEvent = results.getString(1);
					String metric = results.getString(2);
					String value = results.getString(3);
					int dataType = results.getInt(4);
					String clusterIndex = results.getString(5);
					// for each cluster index, create a vector(s) and add the values
					if (!clusterID.equals(clusterIndex)) {
						clusterID = clusterIndex;
						centroids = new ArrayList();
						analysisResults.getClusterCentroids().add(centroids);
						deviations = new ArrayList();
						analysisResults.getClusterDeviations().add(deviations);
					}
					if (dataType == 0)
						centroids.add(value);
					else if (dataType == 1)
						deviations.add(value);
				}
				statement.close();
			}
*/
		} catch (Exception e) {
			String error = "\nERROR: Couldn't select the analysis settings from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		//PerfExplorerOutput.println(" Done!");
		return analysisResults;
	}

	/**
	 * After Cluster analysis has been requested, the results can be
	 * requested.  Based on the model passed in, if the configuration has been
	 * analyzed, the results are retrieved from the database.  The results
	 * are serialized over the RMI link back to the client application.
	 * 
	 * @param model
	 * @return RMIPerformanceResults
	 */
	public RMIPerformanceResults getPerformanceResults(RMIPerfExplorerModel model) {
		//PerfExplorerOutput.print("getPerformanceResults(" + model.toString() + ")... ");
		RMIPerformanceResults analysisResults = new RMIPerformanceResults();
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			Object[] methods = AnalysisType.getClusterMethods();
			StringBuilder qmarks = new StringBuilder();
			for (int i = 0 ; i < methods.length ; i++) {
				if (i > 0)
					qmarks.append(",");
				qmarks.append("?");
			}
			if (model.getCurrentSelection() instanceof Metric) {
				if (db.getSchemaVersion() == 0) {
					statement = db.prepareStatement("select id from analysis_settings where application = ? and experiment = ? and trial = ? and metric = ? and method in (" + qmarks.toString() + ") order by id desc");
					statement.setInt(1, model.getApplication().getID());
					statement.setInt(2, model.getExperiment().getID());
					statement.setInt(3, model.getTrial().getID());
					statement.setInt(4, ((Metric)(model.getCurrentSelection())).getID());
					for (int i = 0 ; i < methods.length ; i++) {
						statement.setString(5+i, methods[i].toString());
					}
				} else {
					statement = db.prepareStatement("select id from analysis_settings where trial = ? and metric = ? and method in (" + qmarks.toString() + ") order by id desc");
					statement.setInt(1, model.getTrial().getID());
					statement.setInt(2, ((Metric)(model.getCurrentSelection())).getID());
					for (int i = 0 ; i < methods.length ; i++) {
						statement.setString(3+i, methods[i].toString());
					}
				}
			} else {
				if (db.getSchemaVersion() == 0) {
					statement = db.prepareStatement("select id from analysis_settings where application = ? and experiment = ? and trial = ? and metric is null and method in (" + qmarks.toString() + ") order by id desc");
					statement.setInt(1, model.getApplication().getID());
					statement.setInt(2, model.getExperiment().getID());
					statement.setInt(3, model.getTrial().getID());
					for (int i = 0 ; i < methods.length ; i++) {
						statement.setString(4+i, methods[i].toString());
					}
				} else {
					statement = db.prepareStatement("select id from analysis_settings where trial = ? and metric is null and method in (" + qmarks.toString() + ") order by id desc");
					statement.setInt(1, model.getTrial().getID());
					for (int i = 0 ; i < methods.length ; i++) {
						statement.setString(2+i, methods[i].toString());
					}
				}
			}
			//PerfExplorerOutput.println(statement.toString());
			ResultSet results = statement.executeQuery();
			int analysisID = 0;
			if (results.next() != false) {
				analysisID = results.getInt(1);
			} else {
				// no results for this combination
				statement.close();
				return analysisResults;
			}
			statement.close();
			statement = db.prepareStatement("select id, description, thumbnail_size, " + 
					"thumbnail, image_size, image, result_type from analysis_result where " + 
					"analysis_settings = ? order by id asc");
			statement.setInt(1, analysisID);
			//PerfExplorerOutput.println(statement.toString());
			results = statement.executeQuery();
			while (results.next() != false) {
				String id = results.getString(1);
				String description = results.getString(2);
				int thumbSize = results.getInt(3);
				InputStream thumbStream = results.getBinaryStream(4);
				byte[] thumbData = new byte[thumbSize];
				//int bytesRead = 
					thumbStream.read(thumbData);
				int imageSize = results.getInt(5);
				InputStream imageStream = results.getBinaryStream(6);
				byte[] imageData = new byte[imageSize];
				//bytesRead = 
					imageStream.read(imageData);
				//String k = results.getString(8);
				analysisResults.getDescriptions().add(description);
				analysisResults.getIDs().add(id);
				//analysisResults.getKs().add(k);
				analysisResults.getThumbnails().add(thumbData);
				analysisResults.getImages().add(imageData);
			}
			statement.close();

/*
			Iterator e = analysisResults.getIDs().iterator();
			while (e.hasNext()) {
				String id = (String) e.next();
				StringBuilder buf = new StringBuilder();
				buf.append("select ie.name, m.name, ard.value, ard.data_type, ard.cluster_index ");
				buf.append("from analysis_result_data ard ");
				buf.append("inner join interval_event ie on ard.interval_event = ie.id ");
				buf.append("inner join metric m on ard.metric = m.id ");
				buf.append("where ard.analysis_result = ? ");
				buf.append("order by ard.cluster_index, ie.name, ard.data_type ");
				statement = db.prepareStatement(buf.toString());
				statement.setString(1, id);
				//PerfExplorerOutput.println(statement.toString());
				//results = statement.executeQuery();
				String clusterID = new String("");
				List centroids = null;
				List deviations = null;
				while (results.next() != false) {
					String intervalEvent = results.getString(1);
					String metric = results.getString(2);
					String value = results.getString(3);
					int dataType = results.getInt(4);
					String clusterIndex = results.getString(5);
					// for each cluster index, create a vector(s) and add the values
					if (!clusterID.equals(clusterIndex)) {
						clusterID = clusterIndex;
						centroids = new ArrayList();
						analysisResults.getClusterCentroids().add(centroids);
						deviations = new ArrayList();
						analysisResults.getClusterDeviations().add(deviations);
					}
					if (dataType == 0)
						centroids.add(value);
					else if (dataType == 1)
						deviations.add(value);
				}
				statement.close();
			}
*/
		} catch (Exception e) {
			String error = "\nERROR: Couldn't select the analysis settings from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		//PerfExplorerOutput.println(" Done!");
		return analysisResults;
	}

	/**
	 * When the background analysis task is completed, the request is popped
	 * of the front of the queue.
	 * 
	 */
	public void taskFinished (int connectionIndex) {
		Queue<RMIPerfExplorerModel> requestQueue = requestQueues.get(connectionIndex);
		//RMIPerfExplorerModel model = 
		requestQueue.poll();
		//PerfExplorerOutput.println(model.toString() + " finished!");
		// endRSession();
	}

	/**
	 * Get the next request from the front of the request queue.  Do not remove
	 * it from the queue, in case something happens, so we can re-process the
	 * request (NOT YET IMPLEMENTED - TODO).
	 * @return
	 */
	public RMIPerfExplorerModel getNextRequest (int connectionIndex) {
		Queue<RMIPerfExplorerModel> requestQueue = requestQueues.get(connectionIndex);
		RMIPerfExplorerModel model = requestQueue.peek();
		return model;
	}

	/**
	 * Convenience method to return a pointer to the datbase connection.
	 * @return
	 */
	public DB getDB (){
		if (session != null)
			return session.db();
		return null;
	}

	/**
	 * Check to see if the current request has been requested before.
	 * 
	 * @param modelData
	 * @return
	 * @throws PerfExplorerException
	 */
	public int checkForRequest (RMIPerfExplorerModel modelData) throws PerfExplorerException {
		int analysisID = 0;
		// check to see if this request has already been done
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			int index = 1;
			if (db.getSchemaVersion() == 0) {
				statement = db.prepareStatement("select id from analysis_settings where application = ? and experiment = ? and trial = ? and metric = ? and method = ? and dimension_reduction = ? and normalization = ?");
				statement.setInt(index++, modelData.getApplication().getID());
				statement.setInt(index++, modelData.getExperiment().getID());
			} else {
				statement = db.prepareStatement("select id from analysis_settings where trial = ? and metric = ? and method = ? and dimension_reduction = ? and normalization = ?");			
			}
			statement.setInt(index++, modelData.getTrial().getID());
			statement.setInt(index++, ((Metric)(modelData.getCurrentSelection())).getID());
			statement.setString(index++, modelData.getClusterMethod().toString());
			statement.setString(index++, modelData.getDimensionReduction().toString());
			statement.setString(index++, modelData.getNormalization().toString());
			ResultSet results = statement.executeQuery();
			if (results.next() != false) {
				analysisID = results.getInt(1);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			String error = "ERROR: Couldn't select the analysis settings from the database!";
			System.err.println(error);
			e.printStackTrace();
			throw new PerfExplorerException(error, e);
		}
		return analysisID;
	}

	/**
	 * Insert a new processing request into the database.
	 * 
	 * @param modelData
	 * @return
	 * @throws PerfExplorerException
	 */
	public int insertRequest (RMIPerfExplorerModel modelData) throws PerfExplorerException {
		int analysisID = 0;
		// insert a record into the database for this analysis
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			int index = 1;
			if (db.getSchemaVersion() == 0) {
				statement = db.prepareStatement("insert into analysis_settings (application, experiment, trial, metric, method, dimension_reduction, normalization) values (?, ?, ?, ?, ?, ?, ?)");
				statement.setInt(index++, modelData.getApplication().getID());
				statement.setInt(index++, modelData.getExperiment().getID());
			} else {
				statement = db.prepareStatement("insert into analysis_settings (application, experiment, trial, metric, method, dimension_reduction, normalization) values (null, null, ?, ?, ?, ?, ?)");
			}
			statement.setInt(index++, modelData.getTrial().getID());
			if (modelData.getCurrentSelection() instanceof Metric) 
				statement.setInt(index++, ((Metric)(modelData.getCurrentSelection())).getID());
			else
				statement.setNull(index++, java.sql.Types.INTEGER);
			statement.setString(index++, modelData.getClusterMethod().toString());
			statement.setString(index++, modelData.getDimensionReduction().toString());
			statement.setString(index++, modelData.getNormalization().toString());
			statement.execute();
			statement.close();
			String tmpStr = new String();
			if (db.getDBType().compareTo("mysql") == 0) {
		   	tmpStr = "select LAST_INSERT_ID();";
			} else if (db.getDBType().compareTo("db2") == 0) {
		   	tmpStr = "select IDENTITY_VAL_LOCAL() FROM analysis_settings";
			} else if (db.getDBType().compareTo("derby") == 0) {
		   	tmpStr = "select IDENTITY_VAL_LOCAL() FROM analysis_settings";
			} else if (db.getDBType().compareTo("h2") == 0) {
		   	tmpStr = "select IDENTITY_VAL_LOCAL() FROM analysis_settings";
			} else if (db.getDBType().compareTo("oracle") == 0) {
		   	tmpStr = "SELECT as_id_seq.currval FROM DUAL";
			} else { // postgresql 
		   	tmpStr = "select currval('analysis_settings_id_seq');";
			}
			analysisID = Integer.parseInt(db.getDataItem(tmpStr));
		} catch (SQLException e) {
			String error = "ERROR: Couldn't insert the analysis settings into the database!\nPlease make sure that the analysis_settings and analysis_results tables\nhave been created in the database.";
			System.err.println(error);
			e.printStackTrace();
			throw new PerfExplorerException(error, e);
		}
		return analysisID;
	}

	/**
	 * Make an interactive request for Chart data.
	 * 
	 * @param modelData
	 * @param dataType
	 * @return RMIChartData
	 */
	public RMIChartData requestChartData(RMIPerfExplorerModel modelData, ChartDataType dataType) {
		//PerfExplorerOutput.println("requestChartData(" + modelData.toString() + ")...");
		ChartData chartData = ChartData.getChartData(modelData, dataType);
		return chartData;
	}

	/**
	 * Make an interactive request for Chart data.
	 * 
	 * @param modelData
	 * @param dataType
	 * @return RMIChartData
	 */
	public RMIGeneralChartData requestGeneralChartData(RMIPerfExplorerModel modelData, ChartDataType dataType) {
		//PerfExplorerOutput.println("requestChartData(" + modelData.toString() + ")...");
		GeneralChartData chartData = GeneralChartData.getChartData(modelData, dataType);
		return chartData;
	}

	/**
	 * Make an interactive request for XML fields.
	 * 
	 * @param modelData
	 * @return List
	 */
	public List<String> getXMLFields(RMIPerfExplorerModel modelData) {
		List<String> listOfMetadata = null;
		if (this.getDB().getSchemaVersion() == 0) {
			listOfMetadata = PerfDMFGeneralChartData.getXMLFields(modelData);
		} else {
			listOfMetadata = this.getPotentialPrimaryMetadata(modelData);
		}
		return listOfMetadata;
	}

	/**
	 * Get the groups defined in these profiles.  The client passes in a model
	 * with one or more experiments selected, and the code will get all the
	 * groups which are common among all trials for those experiemnts.
	 * 
	 * @param modelData
	 * @return List
	 */
	public List<String> getPotentialGroups(RMIPerfExplorerModel modelData) {
		//PerfExplorerOutput.println("getPotentialGroups()...");
		List<String> groups = new ArrayList<String>();
		try {
			DB db = this.getDB();
			StringBuilder buf;
			if (db.getSchemaVersion() == 0) {
				buf = new StringBuilder("select distinct ie.group_name ");
				buf.append(" from interval_event ie inner join trial t on ie.trial = t.id ");
			} else {
				buf = new StringBuilder("select distinct tg.group_name ");
				buf.append(" from timer_group tg inner join timer on tg.timer = timer.id inner join trial t on timer.trial = t.id ");
			}
			Object object = modelData.getCurrentSelection();
			if (object instanceof View) {
				buf.append(modelData.getViewSelectionPath(true, true, db.getDBType(), db.getSchemaVersion()));
			} else {
				buf.append(" inner join experiment e on t.experiment = e.id ");
				List<Object> selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application = ");
						buf.append(modelData.getApplication().getID());
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment = ");
						buf.append(modelData.getExperiment().getID());
					} else if (selection instanceof Trial) {
						buf.append(" where t.id = ");
						buf.append(modelData.getTrial().getID());
					}
				} else {
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Application app = (Application)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(app.getID());
						}
						buf.append(")");
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Experiment exp = (Experiment)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(exp.getID());
						}
						buf.append(")");
					} else if (selection instanceof Trial) {
						buf.append(" where t.id in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Trial trial = (Trial)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(trial.getID());
						}
						buf.append(")");
					}
				}
			}
			PreparedStatement statement = db.prepareStatement(buf.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				groups.add(results.getString(1));
			}
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the groups from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		return groups;
	}

	
	/**
	 * Get the metrics defined in these profiles.  The client passes in a model
	 * with one or more experiments selected, and the code will get all the
	 * metrics which are common among all trials for those experiemnts.
	 * 
	 * @param modelData
	 * @return List
	 */
	public List<String> getPotentialMetrics(RMIPerfExplorerModel modelData) {
		//PerfExplorerOutput.println("getPotentialMetrics()...");
		List<String> metrics = new ArrayList<String>();
		StringBuilder buf = new StringBuilder();
		try {
			DB db = this.getDB();
			
			if(db.getSchemaVersion()>0){
				db.getDatabase().setTAUdb(true);
			}else{db.getDatabase().setTAUdb(false);}
			if(db.getDatabase().isTAUdb()) return getPotentialMetricsTAUdb(modelData);
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("select distinct count(cast (m.name as VARCHAR(256))), cast (m.name as VARCHAR(256)) ");
			} else {
				buf.append("select distinct count(m.name), m.name ");
			}
			buf.append(" from metric m inner join trial t on m.trial = t.id ");
			Object object = modelData.getCurrentSelection();
			if (object instanceof View) {
				buf.append(modelData.getViewSelectionPath(true, true, db.getDBType(), db.getSchemaVersion()));
			} else {
				if (db.getSchemaVersion() == 0) {
					buf.append(" inner join experiment e on t.experiment = e.id ");
				}
				List<Object> selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application = ");
						buf.append(modelData.getApplication().getID());
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment = ");
						buf.append(modelData.getExperiment().getID());
					} else if (selection instanceof Trial) {
						buf.append(" where t.id = ");
						buf.append(modelData.getTrial().getID());
					}
				} else {
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Application app = (Application)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(app.getID());
						}
						buf.append(")");
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Experiment exp = (Experiment)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(exp.getID());
						}
						buf.append(")");
					} else if (selection instanceof Trial) {
						buf.append(" where t.id in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Trial trial = (Trial)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(trial.getID());
						}
						buf.append(")");
					}
				}
			}
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" group by cast (m.name as VARCHAR(256)) order by 1 desc");
			} else if (db.getDBType().compareTo("mysql") == 0) {
				buf.append(" group by 2 order by 1 desc");
			//} else if (db.getDBType().compareTo("mysql") == 0) {
					//buf.append(" group by m.name order by 1 desc");
			} else {
				buf.append(" group by m.name order by count(m.name) desc");
			}
//			PerfExplorerOutput.println(buf.toString());
			PreparedStatement statement = db.prepareStatement(buf.toString());
			ResultSet results = statement.executeQuery();
			// only get the metrics that are in all trials.
			// KAH - changed - get all metrics from all trials.
//			int trialCount = 0;
			while (results.next() != false) {
/*				if (trialCount == 0)
					trialCount = results.getInt(1);
				if (results.getInt(1) == trialCount)
*/					metrics.add(results.getString(2));
			}
			results.close();
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the metrics from the database!";
			System.err.println(error);
			System.err.println(buf.toString());
			e.printStackTrace();
		}
		return metrics;
	}
	/**
	 * Get the metrics defined in these profiles.  The client passes in a model
	 * with one or more experiments selected, and the code will get all the
	 * metrics which are common among all trials for those experiemnts.
	 * 
	 * @param modelData
	 * @return List
	 */
	private List<String> getPotentialMetricsTAUdb(RMIPerfExplorerModel modelData) {
		//PerfExplorerOutput.println("getPotentialMetrics()...");
		List<String> metrics = new ArrayList<String>();
		StringBuilder buf = new StringBuilder();
		try {
			DB db = this.getDB();
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("select distinct count(cast (m.name as VARCHAR(256))), cast (m.name as VARCHAR(256)) ");
			} else {
				buf.append("select distinct count(m.name), m.name ");
			}
			buf.append(" from metric m inner join trial t on m.trial = t.id ");
			Object object = modelData.getCurrentSelection();
			if (object instanceof View) {
				buf.append(modelData.getViewSelectionPath(false, false, db.getDBType(), db.getSchemaVersion()));
			} else {
				List<Object> selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					Object selection = modelData.getCurrentSelection();
	
					if (selection instanceof Trial) {
						buf.append(" where t.id = ");
						buf.append(modelData.getTrial().getID());
					}
				} else {
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Trial) {
						buf.append(" where t.id in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Trial trial = (Trial)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(trial.getID());
						}
						buf.append(")");
					}
				}
			}
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" group by cast (m.name as VARCHAR(256)) order by 1 desc");
			} else if (db.getDBType().compareTo("mysql") == 0) {
				buf.append(" group by 2 order by 1 desc");
			//} else if (db.getDBType().compareTo("mysql") == 0) {
					//buf.append(" group by m.name order by 1 desc");
			} else {
				buf.append(" group by m.name order by count(m.name) desc");
			}
//			PerfExplorerOutput.println(buf.toString());
			PreparedStatement statement = db.prepareStatement(buf.toString());
			ResultSet results = statement.executeQuery();
			// only get the metrics that are in all trials.
			// KAH - changed - get all metrics from all trials.
//			int trialCount = 0;
			while (results.next() != false) {
/*				if (trialCount == 0)
					trialCount = results.getInt(1);
				if (results.getInt(1) == trialCount)
*/					metrics.add(results.getString(2));
			}
			results.close();
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the metrics from the database!";
			System.err.println(error);
			System.err.println(buf.toString());
			e.printStackTrace();
		}
		return metrics;
	}
	/**
	 * Get the events defined in these profiles.  The client passes in a model
	 * with one or more experiments selected, and the code will get all the
	 * events which are common among all trials for those experiemnts.
	 * 
	 * @param modelData
	 * @return List
	 */
	public List<String> getPotentialCallPathEvents(RMIPerfExplorerModel modelData) {
		//PerfExplorerOutput.println("getPotentialEvents()...");
		
		List<String> events = new ArrayList<String>();
		try {
			DB db = this.getDB();
			if(db.getSchemaVersion()>0) return getPotentialCallPathEventsTAUdb(modelData);
			StringBuilder buf = new StringBuilder();
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("select distinct cast (m.name as VARCHAR(256))");
			} else {
				buf.append("select distinct ie.name ");
			}
			if (db.getSchemaVersion() == 0) {
				buf.append(" from interval_event ie inner join trial t on ie.trial = t.id ");
			} else {
				buf.append(" from timer ie inner join trial t on ie.trial = t.id ");
				buf.append("inner join timer_callpath tcp on tcp.timer = ie.id and tcp.parent is not null ");
			}
			Object object = modelData.getCurrentSelection();
			if (object instanceof View) {
				buf.append(modelData.getViewSelectionPath(true, true, db.getDBType(), db.getSchemaVersion()));
			} else {
				if(db.getSchemaVersion()==0)
				buf.append(" inner join experiment e on t.experiment = e.id ");
				List<Object> selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application = ");
						buf.append(modelData.getApplication().getID());
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment = ");
						buf.append(modelData.getExperiment().getID());
					} else if (selection instanceof Trial) {
						buf.append(" where t.id = ");
						buf.append(modelData.getTrial().getID());
					}
				} else {
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Application app = (Application)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(app.getID());
						}
						buf.append(")");
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Experiment exp = (Experiment)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(exp.getID());
						}
						buf.append(")");
					} else if (selection instanceof Trial) {
						buf.append(" where t.id in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Trial trial = (Trial)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(trial.getID());
						}
						buf.append(")");
					}
				}
			}
			if (db.getSchemaVersion() == 0) {
				buf.append(" and (group_name is null or group_name like '%TAU_CALLPATH%') ");
			}
			PreparedStatement statement = db.prepareStatement(buf.toString());
			//PerfExplorerOutput.println(statement.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				events.add(results.getString(1));
			}
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the events from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		return events;
	}
	/**
	 * Get the events defined in these profiles.  The client passes in a model
	 * with one or more experiments selected, and the code will get all the
	 * events which are common among all trials for those experiemnts.
	 * 
	 * @param modelData
	 * @return List
	 */
	public List<String> getPotentialCallPathEventsTAUdb(RMIPerfExplorerModel modelData) {
		//PerfExplorerOutput.println("getPotentialEvents()...");
		List<String> events = new ArrayList<String>();
		try {
			DB db = this.getDB();
//			StringBuilder buf = new StringBuilder();
//			if (db.getDBType().compareTo("db2") == 0) {
//				buf.append("select distinct cast (m.name as VARCHAR(256))");
//			} else {
//				buf.append("select distinct ie.name ");
//			}
//			if (db.getSchemaVersion() == 0) {
//				buf.append(" from interval_event ie inner join trial t on ie.trial = t.id ");
//			} else {
//				buf.append(" from timer ie inner join trial t on ie.trial = t.id ");
//				buf.append("inner join timer_callpath tcp on tcp.timer = ie.id and tcp.parent is not null ");
//			}
//			Object object = modelData.getCurrentSelection();
//			if (object instanceof View) {
//				buf.append(modelData.getViewSelectionPath(true, true, db.getDBType(), db.getSchemaVersion()));
//			} else {

			Set<String> keyset = null;
				List<Object> selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Trial) {
						keyset = TAUdbDatabaseAPI.getCallDataMap(modelData.getTrial().getID(), db).keySet();
					}
				} else {
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Trial) {
						ArrayList<Integer> trials = new ArrayList<Integer>();
						for (int i = 0 ; i < selections.size() ; i++) {
							trials.add( ((Trial)selections.get(i)).getID());
						}
						keyset = TAUdbDatabaseAPI.getCallDataMap(trials, db).keySet();

					}
				}
//			}
//

			for(String s: keyset){
				events.add(s);
			}
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the events from the database!";
			System.err.println(error);
			e.printStackTrace();
		}

		return events;
	}
	/**
	 * Get the events defined in these profiles.  The client passes in a model
	 * with one or more experiments selected, and the code will get all the
	 * events which are common among all trials for those experiemnts.
	 * 
	 * @param modelData
	 * @return List
	 */
	public List<String> getPotentialEvents(RMIPerfExplorerModel modelData) {
		//PerfExplorerOutput.println("getPotentialEvents()...");
		List<String> events = new ArrayList<String>();
		try {
			DB db = this.getDB();
			StringBuilder buf = new StringBuilder();
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("select distinct cast (m.name as VARCHAR(256))");
			} else {
				buf.append("select distinct ie.name ");
			}
			if (db.getSchemaVersion() == 0) {
				buf.append(" from interval_event ie inner join trial t on ie.trial = t.id ");
			} else {
				buf.append(" from timer ie inner join trial t on ie.trial = t.id ");
				buf.append("inner join timer_callpath tcp on tcp.timer = ie.id and tcp.parent is null ");
			}
			Object object = modelData.getCurrentSelection();
			if (object instanceof View) {
				buf.append(modelData.getViewSelectionPath(true, true, db.getDBType(), db.getSchemaVersion()));
			} else {
				if (db.getSchemaVersion() == 0) {
					buf.append(" inner join experiment e on t.experiment = e.id ");
				}
				List<Object> selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application = ");
						buf.append(modelData.getApplication().getID());
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment = ");
						buf.append(modelData.getExperiment().getID());
					} else if (selection instanceof Trial) {
						buf.append(" where t.id = ");
						buf.append(modelData.getTrial().getID());
					}
				} else {
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Application app = (Application)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(app.getID());
						}
						buf.append(")");
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Experiment exp = (Experiment)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(exp.getID());
						}
						buf.append(")");
					} else if (selection instanceof Trial) {
						buf.append(" where t.id in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Trial trial = (Trial)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(trial.getID());
						}
						buf.append(")");
					}
				}
			}
			if (db.getSchemaVersion() == 0) {
				buf.append(" and (group_name is null or group_name not like '%TAU_CALLPATH%') ");
			}
			PreparedStatement statement = db.prepareStatement(buf.toString());
			//PerfExplorerOutput.println(statement.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				events.add(results.getString(1));
			}
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the events from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		return events;
	}

	/**
	 * Get the events defined in these profiles.  The client passes in a model
	 * with one or more experiments selected, and the code will get all the
	 * events which are common among all trials for those experiemnts.
	 * 
	 * @param modelData
	 * @return List
	 */
	public List<String> getPotentialAtomicEvents(RMIPerfExplorerModel modelData) {
		//PerfExplorerOutput.println("getPotentialEvents()...");
		List<String> events = new ArrayList<String>();
		try {
			DB db = this.getDB();
			StringBuilder buf = new StringBuilder();
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("select distinct cast (m.name as VARCHAR(256))");
			} else {
				buf.append("select distinct ae.name ");
			}
			if (db.getSchemaVersion() == 0) {
				buf.append(" from atomic_event ae inner join trial t on ae.trial = t.id ");
			} else {
				buf.append(" from counter ae inner join trial t on ae.trial = t.id ");
			}
			Object object = modelData.getCurrentSelection();
			if (object instanceof View) {
				buf.append(modelData.getViewSelectionPath(true, true, db.getDBType(), db.getSchemaVersion()));
			} else {
				if (db.getSchemaVersion() == 0) {
					buf.append(" inner join experiment e on t.experiment = e.id ");
				}
				List<Object> selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application = ");
						buf.append(modelData.getApplication().getID());
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment = ");
						buf.append(modelData.getExperiment().getID());
					} else if (selection instanceof Trial) {
						buf.append(" where t.id = ");
						buf.append(modelData.getTrial().getID());
					}
				} else {
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Application app = (Application)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(app.getID());
						}
						buf.append(")");
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Experiment exp = (Experiment)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(exp.getID());
						}
						buf.append(")");
					} else if (selection instanceof Trial) {
						buf.append(" where t.id in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Trial trial = (Trial)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(trial.getID());
						}
						buf.append(")");
					}
				}
			}
			PreparedStatement statement = db.prepareStatement(buf.toString());
			//PerfExplorerOutput.println(statement.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				events.add(results.getString(1));
			}
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the events from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		return events;
	}

	/**
	 * Get the events defined in these profiles.  The client passes in a model
	 * with one or more experiments selected, and the code will get all the
	 * events which are common among all trials for those experiemnts.
	 * 
	 * @param modelData
	 * @return List
	 */
	public List<String> getPotentialPrimaryMetadata(RMIPerfExplorerModel modelData) {
		List<String> events = new ArrayList<String>();
		try {
			DB db = this.getDB();
			StringBuilder buf = new StringBuilder();
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("select distinct cast (name as VARCHAR(256))");
			} else {
				buf.append("select distinct pm.name ");
			}
			buf.append(" from primary_metadata pm left outer join trial t on pm.trial = t.id ");
			Object object = modelData.getCurrentSelection();
			if (object instanceof View) {
				buf.append(modelData.getViewSelectionPath(true, true, db.getDBType(), db.getSchemaVersion()));
			} else {
				List<Object> selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Trial) {
						buf.append(" where t.id = ");
						buf.append(modelData.getTrial().getID());
					}
				} else {
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Trial) {
						buf.append(" where t.id in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Trial trial = (Trial)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(trial.getID());
						}
						buf.append(")");
					}
				}
			}
			buf.append(" order by name asc ");
			PreparedStatement statement = db.prepareStatement(buf.toString());
			//PerfExplorerOutput.println(statement.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				events.add(results.getString(1));
			}
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the events from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		return events;
	}

//	private static String shortName(String longName) {
//		StringTokenizer st = new StringTokenizer(longName, "(");
//		String shorter = null;
//		try {
//			shorter = st.nextToken();
//			if (shorter.length() < longName.length()) {
//				shorter = shorter + "()";
//			}
//		} catch (NoSuchElementException e) {
//			shorter = longName;
//		}
//		return shorter;
//	}

	/**
	 * This method will request the column names for the application, 
	 * experiment or trial table in the database.
	 * 
	 * @param tableName
	 * @return String[]
	 */
	public String[] getMetaData (String tableName) {
		//PerfExplorerOutput.println("getMetaData()...");
		String[] columns = null;
		try {
			if (tableName.equalsIgnoreCase("application")) {
				columns = Application.getFieldNames(this.getDB());
			} else if (tableName.equalsIgnoreCase("experiment")) {
				columns = Experiment.getFieldNames(this.getDB());
			} else if (tableName.equalsIgnoreCase("trial")) {
				columns = Trial.getFieldNames(this.getDB());
			}
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the columns from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		return columns;
	}

	/**
	 * When building views in the PerfExplorer Client, it is necessary to
	 * define filters.  The filters are based on equalities/inequalities to
	 * columns in the database.  This method helps the user select the 
	 * potential values from the database.
	 * 
	 * @param tableName
	 * @param columnName
	 * @return List
	 */
	public List<String> getPossibleValues (String tableName, String columnName) {
		//PerfExplorerOutput.println("getPossibleValues()...");
		List<String> values = new ArrayList<String>();
		try {
			DB db = this.getDB();
			StringBuilder buf = new StringBuilder("select distinct ");
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("cast (");
				buf.append(columnName);
				buf.append(" as varchar(256))");
			} else {
				buf.append(columnName);
			}
			buf.append(" from ");
			buf.append(tableName.toLowerCase());
			PreparedStatement statement = db.prepareStatement(buf.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				values.add(results.getString(1));
			}
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the potential values from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		return values;
	}

	/**
	 * This method defines a new view for the PerfExplorerClient application.
	 * The name is the name of the new view.  The parent (if greater than 0)
	 * specifies the parent view.  The tableName, columnName, oper, and value
	 * fields define the filter which is to be used to define the view.
	 * 
	 * @param name
	 * @param parent
	 * @param tableName
	 * @param columnName
	 * @param oper
	 * @param value
	 * @return int
	 */
	public int createNewView(String name, int parent, String tableName, String columnName, String oper, String value) {
		//PerfExplorerOutput.println("createNewView()...");
		int viewID = 0;
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			if (parent > 0)
				statement = db.prepareStatement("insert into trial_view (name, table_name, column_name, operator, value, parent) values (?, ?, ?, ?, ?, ?)");
			else 
				statement = db.prepareStatement("insert into trial_view (name, table_name, column_name, operator, value) values (?, ?, ?, ?, ?)");
			statement.setString(1, name);
			statement.setString(2, tableName);
			statement.setString(3, columnName);
			statement.setString(4, oper);
			statement.setString(5, value);
			if (parent > 0)
				statement.setInt(6, parent);
			statement.execute();
			statement.close();
			String tmpStr = new String();
			if (db.getDBType().compareTo("mysql") == 0) {
				tmpStr = "select LAST_INSERT_ID();";
			} else if (db.getDBType().compareTo("db2") == 0) {
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial_view";
			} else if (db.getDBType().compareTo("derby") == 0) {
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial_view";
			} else if (db.getDBType().compareTo("h2") == 0) {
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial_view";
			} else if (db.getDBType().compareTo("oracle") == 0) {
				tmpStr = "SELECT " + db.getSchemaPrefix() + "tv_id_seq.currval FROM DUAL";
			} else { // postgresql 
				tmpStr = "select currval('trial_view_id_seq');";
			}
			viewID = Integer.parseInt(db.getDataItem(tmpStr));
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the columns from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		return viewID;
	}

	/**
	 * This method deletes a view for the PerfExplorerClient application.
	 * The id is the id of the view to be deleted.  All sub-views will
	 * be deleted as well.
	 * 
	 * @param id
	 */
	public void deleteView(String id) {
		//PerfExplorerOutput.println("createNewView()...");
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			statement = db.prepareStatement("delete from trial_view where id = ?");
			statement.setString(1, id);
			statement.execute();
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't delete the view from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
	}

	/**
	 * Get the subviews for this parent.  If the parent is 0, get all top-level
	 * views.
	 * 
	 * @param parent
	 * @return List of views
	 */
	public List<View> getViews (int parent) {
		DB db = this.getDB();
		return View.getViews(parent, db);
	}

	/**
	 * Depth first search to get the all the views in the DB.
	 * 
	 * @param parent
	 * @return List of views
	 */
	public List<View> getAllSubViews (int parent) {
		DB db = this.getDB();
    List<View> views = View.getViews(parent, db);
    ListIterator<View> litr = views.listIterator();
    while (litr.hasNext())
    {
			View pview = litr.next();
			List<View> subViews = getAllSubViews(pview.getID());
      for (View subView : subViews)
      {
				subView.setParent(pview);
        litr.add(subView);
      }
    }
		return views;
	}

	/**
	 * Get the trials which are filtered by the defined view.
	 * 
	 * @param views
	 * @return List
	 */
	public List<Trial> getTrialsForView (List<View> views, boolean getXMLMetadata) {
		DB db = this.getDB();
		
		return View.getTrialsForView(views, getXMLMetadata, db);
	}

	public List<Trial> getTrialsForTAUdbView(List<View> views) {
		return getTrialsForTAUdbView(views, false);
	}

	/**
	 * Get the trials which are filtered by the defined view(s).
	 * 
	 * @param views
	 * @return List
	 */
	public List<Trial> getTrialsForTAUdbView(List<View> views,
			boolean getXMLMetadata) {
		DB db = this.getDB();
		return View.getTrialsForTAUdbView(views, db, getXMLMetadata);
	}

	/* (non-Javadoc)
	 * @see common.RMIPerfExplorer#getVariationAnalysis(common.RMIPerfExplorerModel)
	 */
	public RMIVarianceData getVariationAnalysis(RMIPerfExplorerModel model) {
		RMIVarianceData data = new RMIVarianceData();
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			StringBuilder buf = new StringBuilder();

			if (db.getDBType().compareTo("oracle") == 0) {
				buf.append("select ie.name, ");
				buf.append("avg(ilp.excl), ");
				buf.append("avg(ilp.exclusive_percentage), ");
				buf.append("avg(ilp.call), ");
				buf.append("avg(ilp.excl / ilp.call), ");
				buf.append("max(ilp.excl), ");
				buf.append("min(ilp.excl), ");
				buf.append("stddev(ilp.excl) ");
			} else if (db.getDBType().compareTo("derby") == 0) {
				buf.append("select ie.name, ");
				buf.append("avg(ilp.exclusive), ");
				buf.append("avg(ilp.exclusive_percentage), ");
				buf.append("avg(ilp.num_calls), ");
				buf.append("avg(ilp.exclusive / ilp.num_calls), ");
				buf.append("max(ilp.exclusive), ");
				buf.append("min(ilp.exclusive), ");
				buf.append("0 "); // no support for stddev!
			} else if (db.getDBType().compareTo("db2") == 0) {
				buf.append("select cast (ie.name as varchar(256)), ");
				buf.append("avg(ilp.exclusive), ");
				buf.append("avg(ilp.exclusive_percentage), ");
				buf.append("avg(ilp.call), ");
				buf.append("avg(ilp.exclusive / ilp.call), ");
				buf.append("max(ilp.exclusive), ");
				buf.append("min(ilp.exclusive), ");
				buf.append("stddev(ilp.exclusive) ");
			} else if (db.getDBType().compareTo("mysql") == 0) {
				buf.append("select ie.name, ");
				buf.append("avg(ilp.exclusive), ");
				buf.append("avg(ilp.exclusive_percentage), ");
				buf.append("avg(ilp.call), ");
				buf.append("avg(ilp.exclusive / ilp.`call`), ");
				buf.append("max(ilp.exclusive), ");
				buf.append("min(ilp.exclusive), ");
				buf.append("stddev(ilp.exclusive) ");
			} else {
				buf.append("select ie.name, ");
				buf.append("avg(ilp.exclusive), ");
				buf.append("avg(ilp.exclusive_percentage), ");
				buf.append("avg(ilp.call), ");
				buf.append("avg(ilp.exclusive / ilp.call), ");
				buf.append("max(ilp.exclusive), ");
				buf.append("min(ilp.exclusive), ");
				buf.append("stddev(ilp.exclusive) ");
			}

			buf.append("from interval_location_profile ilp ");
			buf.append("inner join interval_event ie ");
			buf.append("on ilp.interval_event = ie.id ");
			buf.append("where ie.trial = ? and ilp.metric = ? ");

			buf.append("and (ie.group_name is null ");
			buf.append("or (ie.group_name not like '%TAU_CALLPATH%' ");
			buf.append("and group_name not like '%TAU_PHASE%')) ");
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("group by ie.id, cast (ie.name as VARCHAR(256)) order by cast (ie.name as VARCHAR(256)) ");
			} else {
				buf.append("group by ie.id, ie.name order by ie.name");
			}

			statement = db.prepareStatement(buf.toString());
			statement.setInt(1, model.getTrial().getID());
			Metric metric = (Metric)model.getCurrentSelection();
			statement.setInt(2, metric.getID());
			//PerfExplorerOutput.println(statement.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				data.addEventName(results.getString(1));
				double[] values = new double[8];
				values[0]= results.getDouble(2);
				values[1]= results.getDouble(3);
				values[2]= results.getDouble(4);
				values[3]= results.getDouble(5);
				values[4]= results.getDouble(6);
				values[5]= results.getDouble(7);
				String tmp= results.getString(8);
				if (tmp==null || tmp.trim().equalsIgnoreCase("nan") ||
					tmp.trim().equals("0")) {
					values[6] = 0.0;
					values[7] = 0.0;
				} else {
					values[6]= results.getDouble(8);
					// (stddev / range) * percentage
					values[7]= (values[6] / (values[4]-values[5])) * (values[1]);
				}
				data.addValues(values);
			}
			data.addValueName("name");
			data.addValueName("excl");
			data.addValueName("excl %");
			data.addValueName("calls");
			data.addValueName("excl/call");
			data.addValueName("max");
			data.addValueName("min");
			data.addValueName("stddev");
			data.addValueName("(stddev/range)*%");
			results.close();
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't get variation from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		
		return data;
	}

	/* (non-Javadoc)
	 * @see common.RMIPerfExplorer#getCubeData(common.RMIPerfExplorerModel)
	 */
	public RMICubeData getCubeData(RMIPerfExplorerModel model, int numFunctions) {
		RMICubeData data = null;
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			StringBuilder buf = new StringBuilder();
			String clusterType=model.getClusterValueType();
			String clusterPerType="inclusive_percentage";
			if(clusterType==null||clusterType.compareTo("inclusive")!=0){
				if (db.getDBType().compareTo("oracle") == 0) {
					clusterType="excl";
				}
				else{
					clusterType="exclusive";
				}
				clusterPerType="exclusive_percentage";
			}
			if (db.getDBType().compareTo("derby") == 0) {
				//sttdev is unsupported in derby!
				buf.append("select interval_event.id, avg("+clusterType+") ");
			} else {
				buf.append("select interval_event.id, stddev("+clusterType+") ");
			}
			buf.append("from interval_location_profile ");
			buf.append("inner join interval_event ");
			buf.append("on interval_event = interval_event.id ");
			buf.append("where trial = ? and metric = ? ");
			buf.append("and (group_name is null or (");
			buf.append("group_name not like '%TAU_CALLPATH%' ");
			buf.append("and group_name not like '%TAU_PHASE%')) ");
			buf.append("group by interval_event.id ");

			statement = db.prepareStatement(buf.toString());
			//PerfExplorerOutput.println(buf.toString());
			statement.setInt(1, model.getTrial().getID());
			Metric metric = (Metric)model.getCurrentSelection();
			statement.setInt(2, metric.getID());
			//PerfExplorerOutput.println(statement.toString());
			ResultSet results = statement.executeQuery();
			StringBuilder idString = new StringBuilder();
			boolean gotOne = false;
			List<String> goodMethods = new ArrayList<String>();
			while (results.next() != false) {
				if ((results.getString(2) != null) &&
					(!(results.getString(2).trim().equalsIgnoreCase("NaN"))) &&
					(!(results.getString(2).trim().equals(""))) &&
					(!(results.getString(2).trim().equals("0"))) &&
					(results.getDouble(2) != 0.0)) {
					if (gotOne)
						idString.append(",");
					idString.append(results.getString(1));
					gotOne = true;
					//PerfExplorerOutput.println(results.getInt(2));
					goodMethods.add(results.getString(1));
				}
			}
			results.close();
			statement.close();

			buf = new StringBuilder();
			if (db.getDBType().compareTo("db2") == 0) {
				// create a temporary table
				statement = db.prepareStatement("declare global temporary table working_table (id int) on commit preserve rows not logged ");
				statement.execute();
				statement.close();

				// populate it
				statement = db.prepareStatement("insert into SESSION.working_table (id) values (?)");
				for (Iterator<String> i = goodMethods.iterator() ; i.hasNext() ; ) {
					int next = Integer.parseInt(i.next());
					statement.setInt(1, next);
					statement.execute();
				}
				statement.close();

				buf.append("select interval_event.id, cast (name as varchar(256)), (stddev("+clusterType+")/ ");
				buf.append("(max("+clusterType+")-min("+clusterType+")))* ");
				buf.append("avg("+clusterPerType+") ");
				buf.append("from interval_location_profile ");
				buf.append("inner join interval_event ");
				buf.append("on interval_event = interval_event.id ");
				buf.append("inner join SESSION.working_table on ");
				buf.append("interval_event.id = SESSION.working_table.id ");
				buf.append("group by interval_event.id, cast (name as varchar(256))");

			} else {
				if (db.getDBType().compareTo("derby") == 0) {
					// stddev is unsupported!
					buf.append("select interval_event.id, name, avg("+clusterType+") ");
				} else {
					buf.append("select interval_event.id, name, (stddev("+clusterType+")/ ");
					buf.append("(max("+clusterType+")-min("+clusterType+")))* ");
					buf.append("avg("+clusterPerType+") ");
				}
				buf.append("from interval_location_profile ");
				buf.append("inner join interval_event ");
				buf.append("on interval_event = interval_event.id ");
				buf.append("where interval_event.id in (" + idString.toString() + ") ");
				buf.append("group by interval_event.id, name ");
			}
			buf.append("order by 3 desc");
			//PerfExplorerOutput.println(buf.toString());
			statement = db.prepareStatement(buf.toString());
			//PerfExplorerOutput.println(buf.toString());
			//PerfExplorerOutput.println(statement.toString());
			results = statement.executeQuery();
			int count = 0;
			int ids[] = new int[numFunctions];
			String names[] = new String[numFunctions];
			HashMap<String, Integer> nameIndex = new HashMap<String, Integer>();
			while (results.next() != false && count < numFunctions) {
				ids[count]= results.getInt(1);
				names[count] = results.getString(2);
				nameIndex.put(names[count], new Integer(count));
				if ((results.getString(3) != null) &&
					(!(results.getString(3).trim().equalsIgnoreCase("NaN"))) &&
					(!(results.getString(3).trim().equals(""))))
					count++;
			}
			results.close();
			statement.close();

			if (db.getDBType().compareTo("db2") == 0) {
				// drop the temporary table
				statement = db.prepareStatement("drop table SESSION.working_table");
				statement.execute();
				statement.close();
			}

			data = new RMICubeData(count);
			data.setNames(names);

			buf = new StringBuilder();
				
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("select node, context, thread, cast (name as varchar(256)), "+clusterType+" ");
			} else {
				buf.append("select node, context, thread, name, "+clusterType+" ");
			}
			buf.append("from interval_location_profile ");
			buf.append("inner join interval_event ");
			buf.append("on interval_event = interval_event.id ");
			buf.append("where interval_event in (");
			buf.append(ids[0]);
			for (int i = 1 ; i < count ; i++) {
				buf.append(", ");
				buf.append(ids[i]);
			}
			buf.append(") and metric = ? ");
			buf.append("order by 1, 2, 3, 4");
			//PerfExplorerOutput.println(buf.toString());
			statement = db.prepareStatement(buf.toString());
			statement.setInt(1, metric.getID());
			//PerfExplorerOutput.println(statement.toString());
			results = statement.executeQuery();
			int node = 0, context = 0, thread = 0;
			float[] values = new float[count];
			while (results.next() != false) {
				if (node != results.getInt(1) || context != results.getInt(2) || thread != results.getInt(3)) {
					data.addValues(values);
					values = new float[count];
					node = results.getInt(1);
					context = results.getInt(2);
					thread = results.getInt(3);
				}
				values[nameIndex.get(results.getString(4)).intValue()] = results.getFloat(5);
			}
			data.addValues(values);
			results.close();
			statement.close();

		} catch (Exception e) {
			String error = "ERROR: Couldn't get variation from the database!";
			System.err.println(error);
			e.printStackTrace();
		}
		
		return data;
	}

	public String getConnectionString() {
		return session.db().getConnectString();
	}
	
	public List<Integer> getSchemaVersions() {
		return this.schemaVersions;
	}
	
	public List<String> getConnectionStrings() {
		return this.sessionStrings;
	}
	
	public List<String> getConfigNames() {
		return this.configNames;
	}
	
	/**
	 * Requests an EventList object from the PerfDMF database,
	 * based on the trial id which is passed in.
	 * 
	 * @param trialID
	 * @return List of PerfDMF IntervalEvent objects.
	 */
	public List<RMISortableIntervalEvent> getEventList(int trialID, int metricIndex) {
		try {
			this.session.setTrial(trialID, false);
		} catch (DatabaseException e) {}
        DataSource dbDataSource = null;
		if(this.session.getDb().getSchemaVersion() >0 ){
			dbDataSource = new TAUdbDataSource(this.session);
		}else{
			dbDataSource = new DBDataSource(this.session);
		}
		Map<Integer, Function> events = this.session.getIntervalEvents(dbDataSource, metricIndex);
		List<RMISortableIntervalEvent> sortedEvents = new ArrayList<RMISortableIntervalEvent>();
		for (Iterator<Function> i = events.values().iterator() ; i.hasNext() ; ) {
			Function e = i.next();
			sortedEvents.add(new RMISortableIntervalEvent(e, trialID, this.session, metricIndex));
		}
		Collections.sort(sortedEvents);
		return sortedEvents;
	}
	
	/**
	 * Requests a TrialList object from the PerfDMF database,
	 * based on the application id which is passed in.
	 * 
	 * @param criteria a formatted string specifying the 
	 *		  criteria for selecting the trial
	 * @return List of PerfDMF Trial objects.
	 */
	public List<Trial> getTrialList(String criteria, boolean getXMLMetadata) {
		return QueryManager.getTrialList(criteria, getXMLMetadata);
	}

	/**
	 * Requests a TrialList object from the PerfDMF database,
	 * based on the application id which is passed in.
	 * 
	 * @param criteria a formatted string specifying the 
	 *		  criteria for selecting the trial
	 * @return List of PerfDMF Trial objects.
	 */
	public List<String> getChartFieldNames() {
		DB db = this.getDB();
		List<String> list = new ArrayList<String>();
		// This is called when PerfExplorer launches, so make sure we can do something
		// if the database isn't connected yet
		if (db != null && db.getSchemaVersion() == 0) {
			list.add ("application.id");
			list.add ("application.name");
			if (db != null) {
				String[] app = Application.getFieldNames(db);
				for (int i = 0 ; i < app.length ; i++) {
					list.add ("application." + app[i]);
				}
			}
			list.add ("experiment.id");
			list.add ("experiment.name");
			list.add ("experiment.applciation");
			if (db != null) {
				String[] exp = Experiment.getFieldNames(db);
				for (int i = 0 ; i < exp.length ; i++) {
					list.add ("experiment." + exp[i]);
				}
			}
			list.add ("trial.id");
			list.add ("trial.name");
			list.add ("trial.experiment");
			if (db != null) {
				String[] trial = Trial.getFieldNames(db);
				for (int i = 0 ; i < trial.length ; i++) {
					if (trial[i].equalsIgnoreCase(Trial.XML_METADATA_GZ)) {
						// don't add it
					} else if (trial[i].equalsIgnoreCase("node_count") ||
							   trial[i].equalsIgnoreCase("contexts_per_node")) {
						// don't add it
					} else if (trial[i].equalsIgnoreCase("threads_per_context")) {
						list.add ("trial.threads_of_execution");
					} else {
						list.add ("trial." + trial[i]);
					}
				}
			}
		} else {
			list.add ("trial.id");
			list.add ("trial.name");
			if (db != null) {
				String[] trial = Trial.getFieldNames(db);
				for (int i = 0 ; i < trial.length ; i++) {
					if (trial[i].equalsIgnoreCase("node_count") ||
							   trial[i].equalsIgnoreCase("contexts_per_node")) {
						// don't add it
					} else if (trial[i].equalsIgnoreCase("data_source")) {
						// don't add it
					} else if (trial[i].equalsIgnoreCase("threads_per_context")) {
						// don't add it
					} else if (trial[i].equalsIgnoreCase("total_threads")) {
						list.add ("trial.threads_of_execution");
					} else {
						list.add ("trial." + trial[i]);
					}
				}
			}
			list.add ("primary_metadata.name");
//			list.add ("secondary_metadata.name");
		}
		return list;
	}

	/**
	 * The main method to create the PerfExplorerServer.
	 * 
	 * @param args
	 */
	/*
	public static void main (String[] args) {
		CmdLineParser parser = new CmdLineParser();
		CmdLineParser.Option helpOpt = parser.addBooleanOption('h',"help");
		CmdLineParser.Option configfileOpt = parser.addStringOption('c',"configfile");
		CmdLineParser.Option portOpt = parser.addIntegerOption('p',"port");
		CmdLineParser.Option quietOpt = parser.addBooleanOption('q',"quiet");
		CmdLineParser.Option tauHomeOpt = parser.addBooleanOption('t',"tauHome");
		CmdLineParser.Option tauArchOpt = parser.addBooleanOption('a',"tauArch");
			
		try {   
			parser.parse(args);
		} catch (CmdLineParser.OptionException e) {
			System.err.println(e.getMessage());
			System.err.println(USAGE);
			System.exit(-1);
		}  

		Boolean help = (Boolean) parser.getOptionValue(helpOpt);
		String configFile = (String) parser.getOptionValue(configfileOpt);
		Integer port = (Integer) parser.getOptionValue(portOpt);
		Boolean quiet = (Boolean) parser.getOptionValue(quietOpt);
		String tauHome = (String) parser.getOptionValue(tauHomeOpt);
		String tauArch = (String) parser.getOptionValue(tauArchOpt);


		if (help != null && help.booleanValue()) {
			System.err.println(USAGE);
			System.exit(-1);
		}

		if (quiet == null) 
			quiet = new Boolean(false);

		if (configFile == null) {
			System.err.println("Please enter a valid config file.");
			System.err.println(USAGE);
			System.exit(-1);
		}

		if (System.getSecurityManager() == null) {
			System.setSecurityManager(new RMISecurityManager());
		}

		try {
			RMIPerfExplorer server = new PerfExplorerServer(configFile, port.intValue(), quiet.booleanValue(), tauHome, tauArch);
			Naming.rebind("PerfExplorerServer", server);
			PerfExplorerOutput.println("PerfExplorerServer bound.");
			Runtime.getRuntime().addShutdownHook(
				new java.lang.Thread() {
					public void run () {
						try {
							Naming.unbind("PerfExplorerServer");
							PerfExplorerOutput.println(
							"Server has shut down successfully.");
						} catch (Exception e) {
							System.err.println(
							"Server could not unbind from registry - giving up.");
						}
					}
				}
			);
		} catch(RuntimeException e) {
			System.err.println("Could not add a shutdown hook: " +
			e.getMessage());
		} catch (StubNotFoundException e) {
			System.err.println("You forgot to generate the stubs with RMIC.");
		} catch (ConnectException e) {
			System.err.println("Could not connect to registry. "+
							   "Is it running and on the right port?");
			System.err.println("Try running rmiregistry in the background.");
			System.exit(-1);
		} catch (ServerException e) {
			System.err.println("Registry reports a problem: ");
			System.err.println("Maybe the registry cannot find the stub.  " +
							   "Did you set the classpath?  ");
			System.err.println("You can avoid this if you start the " +
							   "registry in the same folder ");
			System.err.println("as the server's stub, or copy the stub " +
							   "to the folder the registry ");
			System.err.println("was started in.");
			System.exit(-1);
		} catch (ServerError e) {
			System.err.println("Registry reports an error: ");
			System.err.println("Maybe the registry cannot find the DayTime "+
							   "interface.  Did you set the classpath?");
			System.err.println("You can avoid this if you start the "+
							   "registry in the same folder");
			System.err.println("as the server's files, or copy the "+
							   "interface to the folder the registry");
			System.err.println("was started in.");
			System.exit(-1);
		} catch (Exception e) {
			System.err.println("Unhandled PerfExplorerServer exception: " +
							   e.getMessage());
			e.printStackTrace();
		}
	}
	*/

	public void setConnectionIndex(int connectionIndex) throws RemoteException {
		this.session = this.sessions.get(connectionIndex);		
		//PerfExplorerOutput.println("Switching to " + this.session.db().getConnectString() + ".");
	}

	public int getSessionCount() {
		return this.sessions.size();
	}

	public Map<String, Integer> checkScalabilityChartData(
			RMIPerfExplorerModel modelData) throws RemoteException {
		//PerfExplorerOutput.println("getPotentialMetrics()...");
		Map<String, Integer> counts = new HashMap<String, Integer>();
		StringBuilder buf = new StringBuilder();
		try {
			DB db = this.getDB();
			if (db.getSchemaVersion() == 0) {
				buf.append("select t.id, node_count * contexts_per_node * threads_per_context, t.experiment ");
			} else {
				buf.append("select t.id, t.total_threads, 0 ");
			}
			buf.append(" from trial t ");
			Object object = modelData.getCurrentSelection();
			if (object instanceof View) {
				buf.append(modelData.getViewSelectionPath(true, true, db.getDBType(), db.getSchemaVersion()));
			} else {
				buf.append(" inner join experiment e on t.experiment = e.id ");
				List<Object> selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application = ");
						buf.append(modelData.getApplication().getID());
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment = ");
						buf.append(modelData.getExperiment().getID());
					} else if (selection instanceof Trial) {
						buf.append(" where t.id = ");
						buf.append(modelData.getTrial().getID());
					}
				} else {
					Object selection = modelData.getCurrentSelection();
					if (selection instanceof Application) {
						buf.append(" where e.application in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Application app = (Application)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(app.getID());
						}
						buf.append(")");
					} else if (selection instanceof Experiment) {
						buf.append(" where t.experiment in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Experiment exp = (Experiment)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(exp.getID());
						}
						buf.append(")");
					} else if (selection instanceof Trial) {
						buf.append(" where t.id in (");
						for (int i = 0 ; i < selections.size() ; i++) {
							Trial trial = (Trial)selections.get(i);
							if (i > 0)
								buf.append(",");
							buf.append(trial.getID());
						}
						buf.append(")");
					}
				}
			}
			//PerfExplorerOutput.println(buf.toString());
			PreparedStatement statement = db.prepareStatement(buf.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				String threads = new String(results.getString(2) + ":" + results.getString(3));
				Integer count = counts.get(threads);
				if (count == null)
					count = new Integer(1);
				else
					count = count+1;
				counts.put(threads, count);
			}
			results.close();
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the thread counts from the database!";
			System.err.println(error);
			System.err.println(buf.toString());
			e.printStackTrace();
		}
		return counts;
	}

	public Map<String, double[][]> getUserEventData(RMIPerfExplorerModel model)
			throws RemoteException {
		Map<String, double[][]> userEvents = new HashMap<String, double[][]>();
		DB db = getDB();
		PreparedStatement statement = null;
		StringBuilder sql = null;
		// this has been validated at the client
		Trial trial = (Trial)model.getCurrentSelection();
		int threadsPerContext = Integer.parseInt(trial.getField("threads_per_context"));
		int threadsPerNode = Integer.parseInt(trial.getField("contexts_per_node")) * threadsPerContext;
		int total = threadsPerNode * Integer.parseInt(trial.getField("node_count"));
		try {
			// now, get the user events
			sql = new StringBuilder();
			sql.append("select a.name, ");
			sql.append("(p.node * " + threadsPerNode + ") + ");
			sql.append("(p.context * " + threadsPerContext + ") + ");
			sql.append("p.thread as thread, ");
			sql.append("p.sample_count, ");
			sql.append("p.maximum_value, ");
			sql.append("p.minimum_value, ");
			sql.append("p.mean_value, ");
			sql.append("p.standard_deviation ");
			sql.append("from atomic_event a ");
			sql.append("left outer join atomic_location_profile p ");
			sql.append("on a.id = p.atomic_event ");
			sql.append("where a.trial = ? ");
			sql.append(" order by 2,1 ");
			
			statement = db.prepareStatement(sql.toString());
			
			statement.setInt(1, trial.getID());
			//System.out.println(statement.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				String event = results.getString(1);
				double[][] data = userEvents.get(event);
				if (data == null) {
					data = new double[total][5];
				}
				Integer threadID = results.getInt(2);
				data[threadID][0] = results.getDouble(3);
				data[threadID][1] = results.getDouble(4);
				data[threadID][2] = results.getDouble(5);
				data[threadID][3] = results.getDouble(6);
				data[threadID][4] =  results.getDouble(7);
				userEvents.put(event, data);
			}
			results.close();
			statement.close();
			
		} catch (SQLException exception) {
			System.err.println(exception.getMessage());
			exception.printStackTrace();
			if (statement != null)
				System.err.println(statement);
			else
				System.err.println(sql);
		}
		return userEvents;
	}

	public int getSchemaVersion(int index) {
		if (!this.connected.get(index)) {
			this.connectToDatabase(this.actualConfigFiles.get(index), true, index);
		}
		return this.schemaVersions.get(index);
	}

}

