package server;

import java.rmi.*;
import java.rmi.server.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.HashMap;
import java.sql.SQLException;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.io.InputStream;

import common.*;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.perfdmf.database.*;
import jargs.gnu.CmdLineParser;

/**
 * The main PerfExplorer Server class.  This class is defined as a singleton,
 * with multiple threads.  The main thread processes interactive requests, 
 * and the background thread processes long-running analysis requests.
 * This server is accessed through RMI, and objects are passed back and forth
 * over the RMI link to the client.
 *
 * <P>CVS $Id: PerfExplorerServer.java,v 1.24 2005/11/03 22:10:24 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
/**
 * @author khuck
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
/**
 * @author khuck
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public class PerfExplorerServer extends UnicastRemoteObject implements RMIPerfExplorer {
	private static String USAGE = "Usage: PerfExplorerClient [{-h,--help}] {-c,--configfile}=<config_file> [{-e,--engine}=<analysis_engine>] [{-p,--port}=<port_number>]\n  where analysis_engine = R or Weka";

	private DatabaseAPI session = null;
	private Semaphore dbControl = null;
	private Queue requestQueue = null;
	private java.lang.Thread timerThread = null;
	private TimerThread timer = null;

	private static PerfExplorerServer theServer = null;
	/**
	 * Static method to get the server instance reference.
	 * Note: This method assumes the server has been instantiated.
	 * 
	 * @return
	 */
	public static PerfExplorerServer getServer () {
		return theServer;
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
	public static PerfExplorerServer getServer (String configFile, int analysisEngine) {
		try {
			if (theServer == null)
				theServer = new PerfExplorerServer (configFile, analysisEngine, 0, false);
		} catch (Exception e) {
			System.err.println("getServer exception: " + e.getMessage());
			e.printStackTrace();
		}
		return theServer;
	}

	/**
	 * Private constructor, which uses the PerfDMF configuration file to
	 * connect to the PerfDMF database.  A semaphore is also created to 
	 * control access to the database connection.
	 * 
	 * @param configFile
	 * @throws RemoteException
	 */
	private PerfExplorerServer(String configFile, int analysisEngine,
	int port, boolean quiet) throws RemoteException {
		super(port);
		PerfExplorerOutput.initialize(quiet);
		theServer = this;
		this.requestQueue = new Queue();
		try {
			this.session = new DatabaseAPI();
			this.session.initialize(configFile, false);
			this.dbControl = new Semaphore();
			System.out.println("Connected to database.");
		} catch (Exception e) {
			System.out.println("Error connecting to Database!");
		}
		timer = new TimerThread(this, analysisEngine);
		timerThread = new java.lang.Thread(timer);
		this.timerThread.start();
	}

	/**
	 * Returns the semaphore which controls access to the database connection.
	 * 
	 * @return
	 */
	public Semaphore getControl() { return dbControl; }

	/**
	 * Test method.
	 * 
	 */
	public String sayHello() {
		System.out.println("sayHello()...");
		String howdy = new String("Hello, client - this is server!");
		return howdy;
	}

	/**
	 * Requests an ApplicationList object from the PerfDMF database.
	 * 
	 * @return List of PerfDMF Application objects.
	 */
	public List getApplicationList() {
		System.out.println("getApplicationList()...");
		dbControl.WAIT("getApplicationList");
		List applications = this.session.getApplicationList();
		dbControl.SIGNAL("getApplicationList");
		return applications;
	}

	/**
	 * Requests an ExperimentList object from the PerfDMF database,
	 * based on the application id which is passed in.
	 * 
	 * @param applicationID
	 * @return List of PerfDMF Experiment objects.
	 */
	public List getExperimentList(int applicationID) {
		System.out.println("getExperimentList(" + applicationID + ")...");
		this.session.setApplication(applicationID);
		List experiments = null;
		dbControl.WAIT("getExperimentList");
		try {
			experiments = this.session.getExperimentList();
		} catch (DatabaseException e) {}
		dbControl.SIGNAL("getExperimentList");

		return experiments;
	}

	/**
	 * Requests an ExperimentList object from the PerfDMF database,
	 * based on the application id which is passed in.
	 * 
	 * @param experimentID
	 * @return List of PerfDMF Trial objects.
	 */
	public List getTrialList(int experimentID) {
		System.out.println("getTrialList(" + experimentID + ")...");
		try {
			this.session.setExperiment(experimentID);
		} catch (DatabaseException e) {}
		dbControl.WAIT("getTrialList");
		List trials = this.session.getTrialList();
		dbControl.SIGNAL("getTrialList");
		return trials;
	}

	public void stopServer() {
		System.out.println("stopServer()...");
		timer.cancel();
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
		StringBuffer status = new StringBuffer();
		System.out.println("requestAnalysis(" + model.toString() + ")...");
		try {
			if (!force && checkForRequest(model) != 0) {
				throw new PerfExplorerException("Request already exists");
			}
			int analysisID = insertRequest(model);
			status.append("Request " + analysisID + " queued.");
			model.setAnalysisID(analysisID);
			status.append("\nRequest accepted.");
			requestQueue.enqueue(model);
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
		System.out.print("getCorrelationResults(" + model.toString() + ")... ");
		RMIPerformanceResults analysisResults = new RMIPerformanceResults();
		dbControl.WAIT("getCorrelationResults");
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			if (model.getCurrentSelection() instanceof Metric) {
				statement = db.prepareStatement("select id from analysis_settings where application = ? and experiment = ? and trial = ? and metric = ? and method = ? order by id desc");
				statement.setInt(1, model.getApplication().getID());
				statement.setInt(2, model.getExperiment().getID());
				statement.setInt(3, model.getTrial().getID());
				statement.setInt(4, ((Metric)(model.getCurrentSelection())).getID());
				statement.setString(5, RMIPerfExplorerModel.CORRELATION_ANALYSIS);
			}else {
				statement = db.prepareStatement("select id from analysis_settings where application = ? and experiment = ? and trial = ? and metric is null and method = ? order by id desc");
				statement.setInt(1, model.getApplication().getID());
				statement.setInt(2, model.getExperiment().getID());
				statement.setInt(3, model.getTrial().getID());
				statement.setString(4, RMIPerfExplorerModel.CORRELATION_ANALYSIS);
			}
			//System.out.println(statement.toString());
			ResultSet results = statement.executeQuery();
			int analysisID = 0;
			if (results.next() != false) {
				analysisID = results.getInt(1);
			} else {
				// no results for this combination
				statement.close();
				dbControl.SIGNAL("getCorrelationResults");
				return analysisResults;
			}
			statement.close();

			// get the images and thumbnails
			statement = db.prepareStatement("select id, description, thumbnail_size, " + 
					"thumbnail, image_size, image, result_type from analysis_result where " + 
					"analysis_settings = ? and result_type = ? order by id asc");
			statement.setInt(1, analysisID);
			statement.setInt(2, AnalysisTaskWrapper.CORRELATION_SCATTERPLOT);
			//System.out.println(statement.toString());
			results = statement.executeQuery();
			while (results.next() != false) {
				String id = results.getString(1);
				String description = results.getString(2);
				int thumbSize = results.getInt(3);
				InputStream thumbStream = results.getBinaryStream(4);
				byte[] thumbData = new byte[thumbSize];
				int bytesRead = thumbStream.read(thumbData);
				int imageSize = results.getInt(5);
				InputStream imageStream = results.getBinaryStream(6);
				byte[] imageData = new byte[imageSize];
				bytesRead = imageStream.read(imageData);
				String engine = results.getString(7);
				//String k = results.getString(8);
				analysisResults.getDescriptions().add(description);
				analysisResults.getIDs().add(id);
				//analysisResults.getKs().add(k);
				analysisResults.getThumbnails().add(thumbData);
				analysisResults.getImages().add(imageData);
			}
			statement.close();

			/* The following code was commented out for performance reasons.
			 * If the raw performance results are desired, then uncomment
			 * out the following code.  Otherwise, this code will only return
			 * the graphs and charts.
			 */
			
/*			Enumeration e = analysisResults.getIDs().elements();
			while (e.hasMoreElements()) {
				String id = (String) e.nextElement();
				StringBuffer buf = new StringBuffer();
				buf.append("select ie.name, m.name, ard.value, ard.data_type, ard.cluster_index ");
				buf.append("from analysis_result_data ard ");
				buf.append("inner join interval_event ie on ard.interval_event = ie.id ");
				buf.append("inner join metric m on ard.metric = m.id ");
				buf.append("where ard.analysis_result = ? ");
				buf.append("order by ard.cluster_index, ie.name, ard.data_type ");
				statement = db.prepareStatement(buf.toString());
				statement.setString(1, id);
				//System.out.println(statement.toString());
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
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("getCorrelationResults");
		}
		dbControl.SIGNAL("getCorrelationResults");
		System.out.println(" Done!");
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
		System.out.print("getPerformanceResults(" + model.toString() + ")... ");
		RMIPerformanceResults analysisResults = new RMIPerformanceResults();
		dbControl.WAIT("getPerformanceResults");
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			if (model.getCurrentSelection() instanceof Metric) {
				statement = db.prepareStatement("select id from analysis_settings where application = ? and experiment = ? and trial = ? and metric = ? and method = ? order by id desc");
				statement.setInt(1, model.getApplication().getID());
				statement.setInt(2, model.getExperiment().getID());
				statement.setInt(3, model.getTrial().getID());
				statement.setInt(4, ((Metric)(model.getCurrentSelection())).getID());
				statement.setString(5, RMIPerfExplorerModel.K_MEANS);
			} else {
				statement = db.prepareStatement("select id from analysis_settings where application = ? and experiment = ? and trial = ? and metric is null and method = ? order by id desc");
				statement.setInt(1, model.getApplication().getID());
				statement.setInt(2, model.getExperiment().getID());
				statement.setInt(3, model.getTrial().getID());
				statement.setString(4, RMIPerfExplorerModel.K_MEANS);
			}
			ResultSet results = statement.executeQuery();
			int analysisID = 0;
			if (results.next() != false) {
				analysisID = results.getInt(1);
			} else {
				// no results for this combination
				statement.close();
				dbControl.SIGNAL("getPerformanceResults");
				return analysisResults;
			}
			statement.close();
			statement = db.prepareStatement("select id, description, thumbnail_size, " + 
					"thumbnail, image_size, image, result_type from analysis_result where " + 
					"analysis_settings = ? order by id asc");
			statement.setInt(1, analysisID);
			//System.out.println(statement.toString());
			results = statement.executeQuery();
			while (results.next() != false) {
				String id = results.getString(1);
				String description = results.getString(2);
				int thumbSize = results.getInt(3);
				InputStream thumbStream = results.getBinaryStream(4);
				byte[] thumbData = new byte[thumbSize];
				int bytesRead = thumbStream.read(thumbData);
				int imageSize = results.getInt(5);
				InputStream imageStream = results.getBinaryStream(6);
				byte[] imageData = new byte[imageSize];
				bytesRead = imageStream.read(imageData);
				String engine = results.getString(7);
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
				StringBuffer buf = new StringBuffer();
				buf.append("select ie.name, m.name, ard.value, ard.data_type, ard.cluster_index ");
				buf.append("from analysis_result_data ard ");
				buf.append("inner join interval_event ie on ard.interval_event = ie.id ");
				buf.append("inner join metric m on ard.metric = m.id ");
				buf.append("where ard.analysis_result = ? ");
				buf.append("order by ard.cluster_index, ie.name, ard.data_type ");
				statement = db.prepareStatement(buf.toString());
				statement.setString(1, id);
				//System.out.println(statement.toString());
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
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("getPerformanceResults");
		}
		dbControl.SIGNAL("getPerformanceResults");
		System.out.println(" Done!");
		return analysisResults;
	}

	/**
	 * When the background analysis task is completed, the request is popped
	 * of the front of the queue.
	 * 
	 */
	public void taskFinished () {
		RMIPerfExplorerModel model = requestQueue.dequeue();
		System.out.println(model.toString() + " finished!");
		// endRSession();
	}

	/**
	 * Get the next request from the front of the request queue.  Do not remove
	 * it from the queue, in case something happens, so we can re-process the
	 * request (NOT YET IMPLEMENTED - TODO).
	 * @return
	 */
	public RMIPerfExplorerModel getNextRequest () {
		RMIPerfExplorerModel model = requestQueue.peekNext();
		return model;
	}

	/**
	 * Convenience method to return a pointer to the datbase connection.
	 * @return
	 */
	public DB getDB (){
		return session.db();
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
		dbControl.WAIT("checkForRequest");
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			statement = db.prepareStatement("select id from analysis_settings where application = ? and experiment = ? and trial = ? and metric = ? and method = ? and dimension_reduction = ? and normalization = ?");
			statement.setInt(1, modelData.getApplication().getID());
			statement.setInt(2, modelData.getExperiment().getID());
			statement.setInt(3, modelData.getTrial().getID());
			statement.setInt(4, ((Metric)(modelData.getCurrentSelection())).getID());
			statement.setString(5, modelData.getClusterMethod());
			statement.setString(6, modelData.getDimensionReduction());
			statement.setString(7, modelData.getNormalization());
			ResultSet results = statement.executeQuery();
			if (results.next() != false) {
				analysisID = results.getInt(1);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			String error = "ERROR: Couldn't select the analysis settings from the database!";
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("checkForRequest");
			throw new PerfExplorerException(error, e);
		}
		dbControl.SIGNAL("checkForRequest");
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
		dbControl.WAIT("insertRequest");
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			statement = db.prepareStatement("insert into analysis_settings (application, experiment, trial, metric, method, dimension_reduction, normalization) values (?, ?, ?, ?, ?, ?, ?)");
			statement.setInt(1, modelData.getApplication().getID());
			statement.setInt(2, modelData.getExperiment().getID());
			statement.setInt(3, modelData.getTrial().getID());
			if (modelData.getCurrentSelection() instanceof Metric) 
				statement.setInt(4, ((Metric)(modelData.getCurrentSelection())).getID());
			else
				statement.setNull(4, java.sql.Types.INTEGER);
			statement.setString(5, modelData.getClusterMethod());
			statement.setString(6, modelData.getDimensionReduction());
			statement.setString(7, modelData.getNormalization());
			statement.execute();
			statement.close();
			String tmpStr = new String();
			if (db.getDBType().compareTo("mysql") == 0) {
		   	tmpStr = "select LAST_INSERT_ID();";
			} else if (db.getDBType().compareTo("db2") == 0) {
		   	tmpStr = "select IDENTITY_VAL_LOCAL() FROM analysis_settings";
			} else if (db.getDBType().compareTo("oracle") == 0) {
		   	tmpStr = "SELECT as_id_seq.currval FROM DUAL";
			} else { // postgresql 
		   	tmpStr = "select currval('analysis_settings_id_seq');";
			}
			analysisID = Integer.parseInt(db.getDataItem(tmpStr));
		} catch (SQLException e) {
			String error = "ERROR: Couldn't insert the analysis settings into the database!\nPlease make sure that the analysis_settings and analysis_results tables\nhave been created in the database, and that\nR and the Omegahat interface have been installed on this machine.";
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("insertRequest");
			throw new PerfExplorerException(error, e);
		}
		dbControl.SIGNAL("insertRequest");
		return analysisID;
	}

	/**
	 * Make an interactive request for Chart data.
	 * 
	 * @param modelData
	 * @param dataType
	 * @return RMIChartData
	 */
	public RMIChartData requestChartData(RMIPerfExplorerModel modelData, int dataType) {
		System.out.println("requestChartData(" + modelData.toString() + ")...");
		ChartData chartData = ChartData.getChartData(modelData, dataType);
		return chartData;
	}

	/**
	 * Get the groups defined in these profiles.  The client passes in a model
	 * with one or more experiments selected, and the code will get all the
	 * groups which are common among all trials for those experiemnts.
	 * 
	 * @param modelData
	 * @return List
	 */
	public List getPotentialGroups(RMIPerfExplorerModel modelData) {
		System.out.println("getPotentialGroups()...");
		List groups = new ArrayList();
		dbControl.WAIT("getPotentialGroups");
		try {
			DB db = this.getDB();
			StringBuffer buf;
			buf = new StringBuffer("select distinct ie.group_name ");
			buf.append(" from interval_event ie inner join trial t on ie.trial = t.id ");
			Object object = modelData.getCurrentSelection();
			if (object instanceof RMIView) {
				buf.append(modelData.getViewSelectionPath(true, true));
			} else {
				buf.append(" where t.experiment in (");
				List selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					buf.append (modelData.getExperiment().getID());
				} else {
					for (int i = 0 ; i < selections.size() ; i++) {
						Experiment exp = (Experiment)selections.get(i);
						if (i > 0)
							buf.append(",");
						buf.append(exp.getID());
					}
				}
				buf.append(")");
			}
			PreparedStatement statement = db.prepareStatement(buf.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				groups.add(results.getString(1));
			}
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the groups from the database!";
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("getPotentialGroups");
		}
		dbControl.SIGNAL("getPotentialGroups");
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
	public List getPotentialMetrics(RMIPerfExplorerModel modelData) {
		System.out.println("getPotentialMetrics()...");
		List metrics = new ArrayList();
		dbControl.WAIT("getPotentialMetrics");
		try {
			DB db = this.getDB();
			StringBuffer buf;
			buf= new StringBuffer("select distinct count(m.name), m.name ");
			buf.append(" from metric m inner join trial t on m.trial = t.id ");
			Object object = modelData.getCurrentSelection();
			if (object instanceof RMIView) {
				buf.append(modelData.getViewSelectionPath(true, true));
			} else {
				buf.append(" where t.experiment in (");
				List selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					buf.append (modelData.getExperiment().getID());
				} else {
					for (int i = 0 ; i < selections.size() ; i++) {
						Experiment exp = (Experiment)selections.get(i);
						if (i > 0)
							buf.append(",");
						buf.append(exp.getID());
					}
				}
				buf.append(")");
			}
			buf.append(" group by m.name order by count(m.name) desc");
			//System.out.println(buf.toString());
			PreparedStatement statement = db.prepareStatement(buf.toString());
			ResultSet results = statement.executeQuery();
			// only get the metrics that are in all trials.
			int trialCount = 0;
			while (results.next() != false) {
				if (trialCount == 0)
					trialCount = results.getInt(1);
				if (results.getInt(1) == trialCount)
					metrics.add(results.getString(2));
			}
			results.close();
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the metrics from the database!";
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("getPotentialMetrics");
		}
		dbControl.SIGNAL("getPotentialMetrics");
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
	public List getPotentialEvents(RMIPerfExplorerModel modelData) {
		System.out.println("getPotentialEvents()...");
		List events = new ArrayList();
		dbControl.WAIT("getPotentialEvents");
		try {
			DB db = this.getDB();
			StringBuffer buf;
			buf = new StringBuffer("select distinct ie.name ");
			buf.append(" from interval_event ie inner join trial t on ie.trial = t.id ");
			Object object = modelData.getCurrentSelection();
			if (object instanceof RMIView) {
				buf.append(modelData.getViewSelectionPath(true, true));
			} else {
				buf.append(" where t.experiment in (");
				List selections = modelData.getMultiSelection();
				if (selections == null) {
					// just one selection
					buf.append (modelData.getExperiment().getID());
				} else {
					for (int i = 0 ; i < selections.size() ; i++) {
						Experiment exp = (Experiment)selections.get(i);
						if (i > 0)
							buf.append(",");
						buf.append(exp.getID());
					}
				}
				buf.append(")");
			}
			PreparedStatement statement = db.prepareStatement(buf.toString());
			//System.out.println(statement.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				events.add(results.getString(1));
			}
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the events from the database!";
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("getPotentialEvents");
		}
		dbControl.SIGNAL("getPotentialEvents");
		return events;
	}

	/**
	 * This method will request the column names for the application, 
	 * experiment or trial table in the database.
	 * 
	 * @param tableName
	 * @return String[]
	 */
	public String[] getMetaData (String tableName) {
		System.out.println("getMetaData()...");
		String[] columns = null;
		dbControl.WAIT("getMetaData");
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
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("getMetaData");
		}
		dbControl.SIGNAL("getMetaData");
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
	public List getPossibleValues (String tableName, String columnName) {
		System.out.println("getPossibleValues()...");
		List values = new ArrayList();
		dbControl.WAIT("getPossibleValues");
		try {
			DB db = this.getDB();
			StringBuffer buf = new StringBuffer("select distinct ");
			buf.append(columnName);
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
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("getPossibleValues");
		}
		dbControl.SIGNAL("getPossibleValues");
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
		System.out.println("createNewView()...");
		int viewID = 0;
		dbControl.WAIT("createNewView");
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
			} else if (db.getDBType().compareTo("oracle") == 0) {
				tmpStr = "SELECT " + db.getSchemaPrefix() + "tv_id_seq.currval FROM DUAL";
			} else { // postgresql 
				tmpStr = "select currval('trial_view_id_seq');";
			}
			viewID = Integer.parseInt(db.getDataItem(tmpStr));
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the columns from the database!";
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("createNewView");
		}
		dbControl.SIGNAL("createNewView");
		return viewID;
	}

	/**
	 * Get the subviews for this parent.  If the parent is 0, get all top-level
	 * views.
	 * 
	 * @param parent
	 * @return List of views
	 */
	public List getViews (int parent) {
		System.out.println("getViews()...");
		List views = new ArrayList();
		dbControl.WAIT("getViews");
		try {
			DB db = this.getDB();
			Iterator names = RMIView.getFieldNames(db);
			if (!names.hasNext()) {
				// the database is not modified to support views
				throw new Exception ("The Database is not modified to support views.");
			}
			StringBuffer buf = new StringBuffer("select ");
			// assumes at least one column...
			buf.append((String) names.next());
			while (names.hasNext()) {
				buf.append(", ");
				buf.append((String) names.next());
			}
			buf.append(" from trial_view");
			if (parent == -1) { // get all views!
				// no while clause
			} else if (parent == 0) {
				buf.append(" where parent is null");
			} else {
				buf.append(" where parent = ");
				buf.append(parent);
			}
			PreparedStatement statement = db.prepareStatement(buf.toString());
			//System.out.println(statement.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				RMIView view = new RMIView();
				for (int i = 1 ; i <= RMIView.getFieldCount() ; i++) {
					view.addField(results.getString(i));
				}
				views.add(view);
			}
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select views from the database!";
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("getViews");
		}
		dbControl.SIGNAL("getViews");
		return views;
	}

	/**
	 * Get the trials which are filtered by the defined view.
	 * 
	 * @param views
	 * @return List
	 */
	/* (non-Javadoc)
	 * @see common.RMIPerfExplorer#getTrialsForView(java.util.List)
	 */
	public List getTrialsForView (List views) {
		System.out.println("getTrialsForView()...");
		List trials = new ArrayList();
		dbControl.WAIT("getTrialsForView");
		try {
			DB db = this.getDB();
			StringBuffer whereClause = new StringBuffer();
			whereClause.append(" inner join application a on e.application = a.id WHERE ");
			for (int i = 0 ; i < views.size() ; i++) {
				if (i > 0) {
					whereClause.append (" AND ");
				}
				RMIView view = (RMIView) views.get(i);

				if (view.getField("TABLE_NAME").equalsIgnoreCase("Application")) {
					whereClause.append (" a.");
				} else if (view.getField("TABLE_NAME").equalsIgnoreCase("Experiment")) {
					whereClause.append (" e.");
				} else /*if (view.getField("table_name").equalsIgnoreCase("Trial")) */ {
					whereClause.append (" t.");
				}
				whereClause.append (view.getField("COLUMN_NAME"));
				whereClause.append (" " + view.getField("OPERATOR") + " '");
				whereClause.append (view.getField("VALUE"));
				whereClause.append ("' ");

			}
			// System.out.println(whereClause.toString());
			trials = Trial.getTrialList(db, whereClause.toString());
		} catch (Exception e) {
			String error = "ERROR: Couldn't select views from the database!";
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("getTrialsForView");
		}
		dbControl.SIGNAL("getTrialsForView");
		return trials;
	}

	/* (non-Javadoc)
	 * @see common.RMIPerfExplorer#getVariationAnalysis(common.RMIPerfExplorerModel)
	 */
	public RMIVarianceData getVariationAnalysis(RMIPerfExplorerModel model) {
		RMIVarianceData data = new RMIVarianceData();
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			StringBuffer buf = new StringBuffer();

			if (db.getDBType().compareTo("oracle") == 0) {
				buf.append("select ie.name, ");
				buf.append("avg(ilp.excl), ");
				buf.append("avg(ilp.exclusive_percentage), ");
				buf.append("avg(ilp.call), ");
				buf.append("avg(ilp.excl / ilp.call), ");
				buf.append("max(ilp.excl), ");
				buf.append("min(ilp.excl), ");
				buf.append("stddev(ilp.excl) ");
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

			buf.append("and ie.group_name not like '%TAU_CALLPATH%' ");
			buf.append("and group_name not like '%TAU_PHASE%' ");
			buf.append("group by ie.id, ie.name order by ie.name");

			statement = db.prepareStatement(buf.toString());
			statement.setInt(1, model.getTrial().getID());
			Metric metric = (Metric)model.getCurrentSelection();
			statement.setInt(2, metric.getID());
			//System.out.println(statement.toString());
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
			data.addValueName("avg exclusive");
			data.addValueName("avg exclusive %");
			data.addValueName("avg calls");
			data.addValueName("avg exclusive per call");
			data.addValueName("max");
			data.addValueName("min");
			data.addValueName("stddev");
			data.addValueName("(stddev/range)*%");
			results.close();
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't get variation from the database!";
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("getVariationAnalysis");
		}
		
		return data;
	}

	/* (non-Javadoc)
	 * @see common.RMIPerfExplorer#getCubeData(common.RMIPerfExplorerModel)
	 */
	public RMICubeData getCubeData(RMIPerfExplorerModel model) {
		RMICubeData data = null;
		try {
			DB db = this.getDB();
			PreparedStatement statement = null;
			StringBuffer buf = new StringBuffer();
			if (db.getDBType().compareTo("oracle") == 0) {
				buf.append("select id, stddev(excl) ");
			} else {
				buf.append("select id, stddev(exclusive) ");
			}
			buf.append("from interval_location_profile ");
			buf.append("inner join interval_event ");
			buf.append("on interval_event = id ");
			buf.append("where trial = ? and metric = ? ");
			buf.append("and group_name not like '%TAU_CALLPATH%' ");
			buf.append("and group_name not like '%TAU_PHASE%' ");
			buf.append("group by interval_event.id ");

			statement = db.prepareStatement(buf.toString());
			statement.setInt(1, model.getTrial().getID());
			Metric metric = (Metric)model.getCurrentSelection();
			statement.setInt(2, metric.getID());
			//System.out.println(statement.toString());
			ResultSet results = statement.executeQuery();
			StringBuffer idString = new StringBuffer();
			boolean gotOne = false;
			while (results.next() != false) {
				if ((results.getString(2) != null) &&
					(!(results.getString(2).trim().equalsIgnoreCase("NaN"))) &&
					(!(results.getString(2).trim().equals("0"))) &&
					(!(results.getString(2).trim().equals("")))) {
					if (gotOne)
						idString.append(",");
					idString.append(results.getString(1));
					gotOne = true;
				}
			}
			results.close();
			statement.close();

			buf = new StringBuffer();
			if (db.getDBType().compareTo("oracle") == 0) {
				buf.append("select id, name, (stddev(excl)/ ");
				buf.append("(max(excl)-min(excl))) * ");
			} else {
				buf.append("select id, name, (stddev(exclusive)/ ");
				buf.append("(max(exclusive)-min(exclusive)))* ");
			}
			buf.append("avg(exclusive_percentage) ");
			buf.append("from interval_location_profile ");
			buf.append("inner join interval_event ");
			buf.append("on interval_event = id ");
			buf.append("where id in (" + idString.toString() + ") ");
			buf.append("group by interval_event.id, name ");
			buf.append("order by 3 desc");
			statement = db.prepareStatement(buf.toString());
			//System.out.println(buf.toString());
			//System.out.println(statement.toString());
			results = statement.executeQuery();
			int count = 0;
			int ids[] = new int[4];
			String names[] = new String[4];
			HashMap nameIndex = new HashMap();
			while (results.next() != false && count < 4) {
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

			data = new RMICubeData(count);
			data.setNames(names);

			buf = new StringBuffer();
			if (db.getDBType().compareTo("oracle") == 0) {
				buf.append("select node, context, thread, name, excl ");
			} else {
				buf.append("select node, context, thread, name, exclusive ");
			}
			buf.append("from interval_location_profile ");
			buf.append("inner join interval_event ");
			buf.append("on interval_event = id ");
			buf.append("where interval_event in (");
			buf.append(ids[0]);
			for (int i = 1 ; i < count ; i++) {
				buf.append(", ");
				buf.append(ids[i]);
			}
			buf.append(") and metric = ? ");
			buf.append("order by 1, 2, 3, 4");
			statement = db.prepareStatement(buf.toString());
			statement.setInt(1, metric.getID());
			//System.out.println(statement.toString());
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
				values[((Integer)nameIndex.get(results.getString(4))).intValue()] = results.getFloat(5);
			}
			data.addValues(values);
			results.close();
			statement.close();

		} catch (Exception e) {
			String error = "ERROR: Couldn't get variation from the database!";
			System.out.println(error);
			e.printStackTrace();
			dbControl.SIGNAL("getVariationAnalysis");
		}
		
		return data;
	}
	
	/**
	 * The main method to create the PerfExplorerServer.
	 * 
	 * @param args
	 */
	public static void main (String[] args) {
		CmdLineParser parser = new CmdLineParser();
		CmdLineParser.Option helpOpt = parser.addBooleanOption('h',"help");
		CmdLineParser.Option configfileOpt = parser.addStringOption('c',"configfile");
		CmdLineParser.Option engineOpt = parser.addStringOption('e',"engine");
		CmdLineParser.Option portOpt = parser.addIntegerOption('p',"port");
		CmdLineParser.Option quietOpt = parser.addBooleanOption('q',"quiet");
			
		try {   
			parser.parse(args);
		} catch (CmdLineParser.OptionException e) {
			System.err.println(e.getMessage());
			System.err.println(USAGE);
			System.exit(-1);
		}  

		Boolean help = (Boolean) parser.getOptionValue(helpOpt);
		String configFile = (String) parser.getOptionValue(configfileOpt);
		String engine = (String) parser.getOptionValue(engineOpt);
		Integer port = (Integer) parser.getOptionValue(portOpt);
		Boolean quiet = (Boolean) parser.getOptionValue(quietOpt);

		int analysisEngine = AnalysisTaskWrapper.WEKA_ENGINE;

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

		if (System.getSecurityManager() == null) {
			System.setSecurityManager(new RMISecurityManager());
		}

		try {
			RMIPerfExplorer server = new PerfExplorerServer(configFile,
			analysisEngine, port.intValue(), quiet.booleanValue());
			Naming.rebind("PerfExplorerServer", server);
			System.out.println("PerfExplorerServer bound.");
			Runtime.getRuntime().addShutdownHook(
				new java.lang.Thread() {
					public void run () {
						try {
							Naming.unbind("PerfExplorerServer");
							System.out.println(
							"Server has shut down successfully.");
						} catch (Exception e) {
							System.out.println(
							"Server could not unbind from registry - giving up.");
						}
					}
				}
			);
		} catch(RuntimeException e) {
			System.out.println("Could not add a shutdown hook: " +
			e.getMessage());
		} catch (StubNotFoundException e) {
			System.out.println("You forgot to generate the stubs with RMIC.");
		} catch (ConnectException e) {
			System.out.println("Could not connect to registry. "+
							   "Is it running and on the right port?");
			System.out.println("Try running rmiregistry in the background.");
			System.exit(-1);
		} catch (ServerException e) {
			System.out.println("Registry reports a problem: ");
			System.out.println("Maybe the registry cannot find the stub.  " +
							   "Did you set the classpath?  ");
			System.out.println("You can avoid this if you start the " +
							   "registry in the same folder ");
			System.out.println("as the server's stub, or copy the stub " +
							   "to the folder the registry ");
			System.out.println("was started in.");
			System.exit(-1);
		} catch (ServerError e) {
			System.out.println("Registry reports an error: ");
			System.out.println("Maybe the registry cannot find the DayTime "+
							   "interface.  Did you set the classpath?");
			System.out.println("You can avoid this if you start the "+
							   "registry in the same folder");
			System.out.println("as the server's files, or copy the "+
							   "interface to the folder the registry");
			System.out.println("was started in.");
			System.exit(-1);
		} catch (Exception e) {
			System.err.println("Unhandled PerfExplorerServer exception: " +
							   e.getMessage());
			e.printStackTrace();
		}

	}
}

