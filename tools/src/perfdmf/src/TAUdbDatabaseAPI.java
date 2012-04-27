package edu.uoregon.tau.perfdmf;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.LinkedHashMap;
import java.util.ArrayList;
import java.util.Map.Entry;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import org.postgresql.PGConnection;
import org.postgresql.copy.CopyManager;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

import edu.uoregon.tau.perfdmf.database.ConnectionManager;
import edu.uoregon.tau.perfdmf.database.DB;

public class TAUdbDatabaseAPI extends DatabaseAPI {
	final static int MEAN_WITHOUT_NULL = -1;
	final static int TOTAL = -2;
	final static int STDDEV_WITHOUT_NULL = -3;
	final static int MIN = -4;
	final static int MAX = -5;
	final static int MEAN_WITH_NULL = -6;
	final static int STDDEV_WITH_NULL = -7;
	
	public TAUdbDatabaseAPI(DatabaseAPI api) {
		super();
        this.database = api.database;
        this.connector = api.connector;
        this.db = this.connector.getDB();
	}

	public TAUdbDatabaseAPI() {
		super();
	}

	public static int uploadTrial(DB db, Trial trial) {

        DataSource dataSource = trial.getDataSource();

        try {
            db.setAutoCommit(false);
        } catch (SQLException e) {
            throw new DatabaseException("Saving Trial Failed: couldn't set AutoCommit to false", e);
        }

        int newTrialID = -1;

        try {
            // save the trial metadata (which returns the new id)
            newTrialID = trial.saveTrial(db);
            trial.setID(newTrialID);
            
            //TODO: Deal with upload size, I think this is for the progress bar.
          //  computeUploadSize(dataSource);
            
            // upload the metrics and get a map that maps the metrics 0 -> n-1 to their unique DB IDs (e.g. 83, 84)
            uploadMetrics(newTrialID, dataSource.getMetrics(), db);
            Map<Metric, Integer> metricMap = getMetricIDMap(newTrialID, dataSource, db);
            
            uploadFunctions(newTrialID, dataSource, db);
            Map<Function, Integer> functionMap = getFunctionsIDMap(newTrialID, dataSource, db);
            
            uploadThreads(newTrialID, dataSource, db);
            Map<Thread, Integer> threadMap = getThreadsMap(newTrialID, dataSource, db);
            
            uploadTimerGroups(functionMap, db);
            uploadTimerParameter(functionMap, db);
            uploadCallpathInfo(dataSource, functionMap, metricMap, threadMap, db, false);
            Map<Thread, Integer> derivedThreadMap = uploadDerivedThreads(newTrialID, dataSource, db);            
            uploadCallpathInfo(dataSource, functionMap, metricMap, derivedThreadMap, db, true);

            if(db.getDBType().equals("postgresql")){
                uploadFunctionProfilesPSQL(dataSource, functionMap, metricMap, threadMap, derivedThreadMap, db);
            }else{
                uploadFunctionProfiles(dataSource, functionMap, metricMap, threadMap, db);
                uploadStatistics(dataSource, functionMap, metricMap, derivedThreadMap, db);
            }

            

           uploadUserEvents(newTrialID, functionMap, dataSource, db);
           Map<UserEvent, Integer> userEventMap = getUserEventsMap(newTrialID, dataSource, db);

            uploadUserEventProfiles( dataSource, userEventMap, db, threadMap);
                        
            uploadMetadata(trial, functionMap, threadMap, db);

          //TODO: Deal with cancel upload
//          if (this.cancelUpload) {
//              db.rollback();
//              deleteTrial(newTrialID);
//              return -1;
//          }
            /*
             *We might in the future need to do something like this to detect if the batch statement uses too much memory 				
				double freePercent =  ((double)Runtime.getRuntime().freeMemory())/Runtime.getRuntime().totalMemory();
				if(freePercent< 0.05){
//					System.out.print("Clearing batch: "+freePercent*100+"%");
					timerValueInsert.executeBatch();
					timerValueInsert.close();
					timerValueInsert = db
							.prepareStatement("INSERT INTO "
									+ db.getSchemaPrefix()
									+ "timer_value (timer, thread, metric, inclusive_percent, inclusive_value, exclusive_percent, "
									+ " exclusive_value) VALUES (?, ?, ?, ?, ?, ?, ?)");

//					 freePercent =  ((double)Runtime.getRuntime().freeMemory())/Runtime.getRuntime().totalMemory();
//					System.out.print("After batch: +freePercent*100+"%");

				}
             */
        } catch (SQLException e) {
            try {
                db.rollback();
                e.printStackTrace();
                throw new DatabaseException("Saving Trial Failed, rollbacks successful", e);
            } catch (SQLException e2) {
                throw new DatabaseException("Saving Trial Failed, rollbacks failed!", e2);
            }

        }

        try {
            db.commit();
            db.setAutoCommit(true);
        } catch (SQLException e) {
            throw new DatabaseException("Saving Trial Failed: commit failed!", e);
        }

        //long stop = System.currentTimeMillis();
        //long elapsedMillis = stop - start;
        //double elapsedSeconds = (double) (elapsedMillis) / 1000.0;
        //        System.out.println("Elapsed time: " + elapsedSeconds + " seconds.");
        return newTrialID;
	}

	private static Map<Thread, Integer> uploadDerivedThreads(int trialID,
			DataSource dataSource, DB db) throws SQLException {
		PreparedStatement stmt = db.prepareStatement("INSERT INTO "
				+ db.getSchemaPrefix()
				+ "thread (trial, node_rank, context_rank, thread_rank,  " +
				" thread_index) VALUES (?, ?, ?, ?, ?)");
		List<Thread> threads = dataSource.getAggThreads();

		for (Thread t : threads) {
			int node_rank = t.getNodeID();
			int context_rank = t.getContextID();
			int thread_rank = t.getThreadID();
			int thread_index = node_rank; // same negative value 
			stmt.setInt(1, trialID);
			stmt.setInt(2, node_rank);
			stmt.setInt(3, context_rank);
			stmt.setInt(4, thread_rank);
			stmt.setInt(5, thread_index);
			stmt.addBatch();
		}
		stmt.executeBatch();
		stmt.close();
		return getDerivedThreadsMap(trialID, dataSource, db);
	}
	

	private static void uploadStatistics(DataSource dataSource,
			Map<Function, Integer> functionMap, Map<Metric, Integer> metricMap,
			Map<Thread, Integer> derivedThreadMap, DB db) throws SQLException {

		Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
		PreparedStatement timerValueInsert = db
				.prepareStatement("INSERT INTO "
						+ db.getSchemaPrefix()
						+ "timer_value (timer, thread, metric, inclusive_percent, inclusive_value, exclusive_percent, "
						+ " exclusive_value) VALUES (?, ?, ?, ?, ?, ?, ?)");
		for (Metric metric : dataSource.getMetrics()) {
			Integer metricID = metricMap.get(metric);

			for (Iterator<Function> func = dataSource.getFunctions(); func
					.hasNext();) {
				Function function = func.next();
				if (function.isGroupMember(derived)) {
					continue;
				}
				Integer timerID = functionMap.get(function);

				
				for (Thread thread : dataSource.getAggThreads()) {
					insertDerivedThreadValue(timerValueInsert, metric,
							metricID, timerID, thread, derivedThreadMap,
							function);
				}
			}


		}
		timerValueInsert.executeBatch();
		timerValueInsert.close();

	}

	private static void insertDerivedThreadValue(
			PreparedStatement timerValueInsert, Metric metric,
			Integer metricID, Integer timerID, Thread thread, 
			Map<Thread, Integer> derivedThreadMap, Function function
			) throws SQLException {
		Integer threadID = derivedThreadMap.get(thread);
		FunctionProfile fp = thread.getFunctionProfile(function);
		if (fp != null) { // only if this thread calls this function
			// TODO: Deal with cancelUpload
			// if (this.cancelUpload)
			// return;
			timerValueInsert.setInt(1,timerID);
			timerValueInsert.setInt(2, threadID);
			timerValueInsert.setInt(3, metricID);
			
			timerValueInsert.setDouble(4, fp.getInclusivePercent(metric.getID()));
			timerValueInsert.setDouble(5, fp.getInclusive(metric.getID()));
			timerValueInsert.setDouble(6, fp.getExclusivePercent(metric.getID()));
			timerValueInsert.setDouble(7, fp.getExclusive(metric.getID()));
			//TODO: Find the sum_exclusive_square values
//						timerValueInsert.setDouble(8, fp.get)
			
			timerValueInsert.addBatch();	
		}
	}

	private static void uploadCallpathInfo(DataSource dataSource,
			Map<Function, Integer> functionMap, Map<Metric, Integer> metricMap,
			Map<Thread, Integer> threadMap, DB db, boolean aggregates) throws SQLException {

		PreparedStatement timerCallpathInsert = db
				.prepareStatement("INSERT INTO "
						+ db.getSchemaPrefix()
						+ "timer_callpath (timer, thread, calls, subroutines, parent) "
						+ "VALUES (?, ?, ?, ?, ?)");

		Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
		Iterator<Function> funcs = functionMap.keySet().iterator();
		while (funcs.hasNext()) {
			Function function = funcs.next();
			if (function.isGroupMember(derived)) {
				continue;
			}
			Integer timerID = functionMap.get(function);

			List<Thread> threads = null;
			if (aggregates) 
				threads = dataSource.getAggThreads();
			else 
				threads = dataSource.getAllThreads();
			for (Thread thread : threads) {
				Integer threadID = threadMap.get(thread);

				FunctionProfile fp = thread.getFunctionProfile(function);

				if (fp != null) { // only if this thread calls this function
					// TODO: Deal with cancelUpload
					// if (this.cancelUpload)
					// return;
					timerCallpathInsert.setInt(1, timerID);
					timerCallpathInsert.setInt(2, threadID);
					timerCallpathInsert.setInt(3, (int) fp.getNumCalls());
					timerCallpathInsert.setInt(4, (int) fp.getNumSubr());
					String parentName = CallPathUtilFuncs.getParentName(fp.getName());
					if (!parentName.equals("")) {
						Function parent = dataSource.getFunction(parentName);
						int parentID = functionMap.get(parent);

						timerCallpathInsert.setInt(5, parentID);
					} else {
						timerCallpathInsert.setNull(5, java.sql.Types.INTEGER);
					}
					timerCallpathInsert.addBatch();
				}
			}
		}
		timerCallpathInsert.executeBatch();
		timerCallpathInsert.close();
	}

	private static void uploadThreads(int trialID,
			DataSource dataSource, DB db) throws SQLException {
		int maxContextPerNode = dataSource.getMaxNCTNumbers()[1] +1;
		int maxThreadsPerContext = dataSource.getMaxNCTNumbers()[2] +1;


		PreparedStatement stmt = db.prepareStatement("INSERT INTO "
						+ db.getSchemaPrefix()
						+ "thread (trial, node_rank, context_rank, thread_rank,  " +
						" thread_index) VALUES (?, ?, ?, ?, ?)");
		List<Thread> threads = dataSource.getAllThreads();
		for (Thread t : threads) {

			int node_rank = t.getNodeID();
			int context_rank = t.getContextID();
			int thread_rank = t.getThreadID();
			int thread_index = node_rank * maxContextPerNode * maxThreadsPerContext + context_rank * maxThreadsPerContext + thread_rank; 
			
	

			stmt.setInt(1, trialID);
			stmt.setInt(2, node_rank);
			stmt.setInt(3, context_rank);
			stmt.setInt(4, thread_rank);
			stmt.setInt(5, thread_index);

			stmt.addBatch();
			
		}
		stmt.executeBatch();
		stmt.close();
	}
	private static Map<Thread, Integer> getThreadsMap(int trialID,DataSource dataSource, DB db) throws SQLException {
		Map<Thread, Integer> map = new HashMap<Thread, Integer>();

		PreparedStatement statement = db.prepareStatement("SELECT node_rank, context_rank, thread_rank, id FROM "
						+ db.getSchemaPrefix()
						+ "thread WHERE trial=?");
		statement.setInt(1, trialID);
		   statement.execute();
           ResultSet results = statement.getResultSet();
      

		while (results.next()) {
			int node = results.getInt(1);
			int context =results.getInt(2);
			int thread = results.getInt(3);
			int id = results.getInt(4);
			Thread t = dataSource.getThread(node, context, thread);
			map.put(t, id);
		}
		statement.close();
        return map;

	}

	private static Map<Thread, Integer> getDerivedThreadsMap(int trialID,DataSource dataSource, DB db) throws SQLException {
		Map<Thread, Integer> map = new HashMap<Thread, Integer>();

		PreparedStatement statement = db.prepareStatement("SELECT thread_index, id FROM "
						+ db.getSchemaPrefix()
						+ "thread WHERE trial=? and thread_index < 0 order by thread_index desc");
		statement.setInt(1, trialID);
		statement.execute();
        ResultSet results = statement.getResultSet();

		while (results.next()) {
			int thread_index = results.getInt(1);
			int id = results.getInt(2);
			switch (thread_index) {
			case MEAN_WITHOUT_NULL:
				map.put(dataSource.getMeanData(), id);
				break; 
			case TOTAL:
				map.put(dataSource.getTotalData(), id);
				break;
			case STDDEV_WITHOUT_NULL:
				map.put(dataSource.getStdDevData(), id);
				break;
			case MIN:
				map.put(dataSource.getMinData(), id);
				break;
			case MAX:
				map.put(dataSource.getMaxData(), id);
				break;
			case MEAN_WITH_NULL:
				map.put(dataSource.getMeanDataAll(), id);
				break;
			case STDDEV_WITH_NULL:
				map.put(dataSource.getStdDevDataAll(), id);
				break;
			default:
				break;
			}
		}
		statement.close();
        return map;

	}


	private static void uploadTimerParameter(Map<Function, Integer> map, DB db) throws SQLException {
		Set<Function> funcs = map.keySet();
		PreparedStatement statement = db.prepareStatement("INSERT INTO "
				+ db.getSchemaPrefix()
				+ "timer_parameter (timer, parameter_name, parameter_value) VALUES (?, ?, ?)");
		for (Function f : funcs) {
			int timer = map.get(f);
			List<Parameter> params = f.getSourceLink().getParameters();
			if (params != null) {
				for (Parameter p : params) {
					statement.setInt(1, timer);
					statement.setString(2, p.getName());
					statement.setString(3, p.getValue());
					statement.addBatch();
				}
			}
		}
		statement.executeBatch();
		statement.close();		
	}

	private static void uploadTimerGroups(Map<Function, Integer> map, DB db) throws SQLException {
		Set<Function> funcs = map.keySet();
		PreparedStatement statement = db.prepareStatement("INSERT INTO "
				+ db.getSchemaPrefix()
				+ "timer_group (timer, group_name) VALUES (?, ?)");
		for (Function f : funcs) {
			int timer = map.get(f);
			List<Group> groups = f.getGroups();
			if (groups != null) {
				for (Group g : groups) {
					statement.setInt(1, timer);
					statement.setString(2, g.getName());
					statement.addBatch();
				}
			}
		}
		statement.executeBatch();
		statement.close();

	}

	private static void uploadMetrics(int trialID, List<Metric> metrics, DB db)
			throws SQLException {

		PreparedStatement insert = db.prepareStatement("INSERT INTO "
				+ db.getSchemaPrefix() + "metric (name, trial, derived) VALUES (?, ?, ?)");
	
		for (Metric metric : metrics) {
			insert.setString(1, metric.getName());
			insert.setInt(2, trialID);
			insert.setBoolean(3,metric.getDerivedMetric());
			insert.addBatch();

		}
		
		insert.executeBatch();
		insert.close();		
	}

	private static Map<Metric, Integer> getMetricIDMap(int trialID,
			DataSource dataSource, DB db) throws SQLException {
		Map<Metric, Integer> map = new HashMap<Metric, Integer>();

		PreparedStatement select = db.prepareStatement("SELECT id, name FROM "
				+ db.getSchemaPrefix() + "metric WHERE trial=?");
		select.setInt(1, trialID);
		select.execute();
		ResultSet results = select.getResultSet();
		while (results.next()) {
			int metricID = results.getInt(1);
			String name = results.getString(2);
			Metric metric = dataSource.getMetric(name);

			metric.setDbMetricID(metricID);
			map.put(metric, metricID);

		}

		select.close();
		return map;
	}
	  // fills the timer table
	    private static void  uploadFunctions(int trialID, DataSource dataSource, DB db) throws SQLException {
	       
	        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
	           PreparedStatement statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
	                    + "timer (trial, name, source_file, line_number, line_number_end, column_number, column_number_end, short_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)");

	        for (Iterator<Function> it = dataSource.getFunctions(); it.hasNext();) {
	            Function f = it.next();
	            if (f.isGroupMember(derived)) {
	                continue; //Should we save the derived callpath functions??
	            }
	            SourceRegion source = f.getSourceLink();
	 
	            statement.setInt(1, trialID);
	            statement.setString(2, f.getName());
	            statement.setString(3, source.getFilename());
	            statement.setInt(4, source.getStartLine());
	            statement.setInt(5, source.getEndLine());
	            statement.setInt(6, source.getStartColumn());
	            statement.setInt(7, source.getEndColumn());
	            statement.setString(8, source.getShortName());
	            statement.addBatch();
//TODO: increment itemsDone for progress bar
//	            this.itemsDone++;
	        }
            statement.executeBatch();
            statement.close();

	    }
	    private static Map<Function, Integer>  getFunctionsIDMap(int trialID, DataSource dataSource, DB db) throws SQLException {
	    		           
		Map<Function, Integer> map = new HashMap<Function, Integer>();

		PreparedStatement statement = db
				.prepareStatement("SELECT id, name FROM " + db.getSchemaPrefix()
						+ "timer WHERE trial=?");
		statement.setInt(1, trialID);
		statement.execute();
		ResultSet results = statement.getResultSet();

		while (results.next()) {
			int funcID = results.getInt(1);
			String name = results.getString(2);

			Function func = dataSource.getFunction(name);
			map.put(func, funcID);
		}
		statement.close();
		return map;
	}

	private static void uploadFunctionProfilesPSQL(DataSource dataSource,
			Map<Function, Integer> functionMap, Map<Metric, Integer> metricMap,
			Map<Thread, Integer> threadMap, Map<Thread, Integer> derivedThreadMap, DB db) throws SQLException {
		org.postgresql.PGConnection conn;
		if (db.getConnection() instanceof org.postgresql.PGConnection) {
			conn = (PGConnection) db.getConnection();
		} else {
			System.err
					.println("Upload using copy was called on a non postgresql database.");
			return;
		}
		CopyManager copy = conn.getCopyAPI();
		StringBuffer buf = new StringBuffer();

		Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");

		for (Metric metric : dataSource.getMetrics()) {
			Integer metricID = metricMap.get(metric);

			for (Iterator<Function> func = dataSource.getFunctions(); func
					.hasNext();) {
				Function function = func.next();
				if (function.isGroupMember(derived)) {
					continue;
				}
				Integer timerID = functionMap.get(function);
				for (Thread thread : dataSource.getAllThreads()) {
					Integer threadID = threadMap.get(thread);
					FunctionProfile fp = thread.getFunctionProfile(function);
					if (fp != null) { // only if this thread calls this function
						buf.append(timerID + "\t");
						buf.append(threadID + "\t");
						buf.append(metricID + "\t");

						buf.append(fp.getInclusivePercent(metric.getID())
								+ "\t");
						buf.append(fp.getInclusive(metric.getID()) + "\t");
						buf.append(fp.getExclusivePercent(metric.getID())
								+ "\t");
						buf.append(fp.getExclusive(metric.getID()) + "\n");
					}
				}
				// now the aggregate threads
				for (Thread thread : dataSource.getAggThreads()) {
					Integer threadID = derivedThreadMap.get(thread);
					FunctionProfile fp = thread.getFunctionProfile(function);
					if (fp != null) { // only if this thread calls this function
						buf.append(timerID + "\t");
						buf.append(threadID + "\t");
						buf.append(metricID + "\t");

						buf.append(fp.getInclusivePercent(metric.getID())
								+ "\t");
						buf.append(fp.getInclusive(metric.getID()) + "\t");
						buf.append(fp.getExclusivePercent(metric.getID())
								+ "\t");
						buf.append(fp.getExclusive(metric.getID()) + "\n");
					}
				}
			}
		}
		// copy data from the given input stream to the table
		InputStream input = new ByteArrayInputStream(buf.toString().getBytes());
		try {
			copy.copyIn(
					"COPY timer_value (timer, thread, metric, inclusive_percent, inclusive_value, exclusive_percent, "
							+ " exclusive_value) FROM STDIN", input);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private static void uploadFunctionProfiles(DataSource dataSource,
			Map<Function, Integer> functionMap, Map<Metric, Integer> metricMap, Map<Thread, Integer> threadMap,
			DB db) throws SQLException {

		PreparedStatement timerValueInsert = db
				.prepareStatement("INSERT INTO "
						+ db.getSchemaPrefix()
						+ "timer_value (timer, thread, metric, inclusive_percent, inclusive_value, exclusive_percent, "
						+ " exclusive_value) VALUES (?, ?, ?, ?, ?, ?, ?)");
		


		Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");

		for (Metric metric:  dataSource.getMetrics()) {
			Integer metricID = metricMap.get(metric);

			for (Iterator<Function> func = dataSource.getFunctions(); func.hasNext();) {
				Function function = func.next();
				if (function.isGroupMember(derived)) {
					continue;
				}
				Integer timerID = functionMap.get(function);
				for (Thread thread : dataSource.getAllThreads()) {
					insertDerivedThreadValue(timerValueInsert, metric,
							metricID, timerID, thread, threadMap, function);
				}
			}
		}
		timerValueInsert.executeBatch();
		timerValueInsert.close();
	}

	private static void uploadUserEvents(int trialID, Map<Function, Integer> functionMap, DataSource dataSource,
			DB db) throws SQLException {
		
		// first, build a map of timer names to integers
		Map<String, Integer> contextMap = new HashMap<String, Integer>();
		for (Function key : functionMap.keySet()) {
			contextMap.put(key.getName(), functionMap.get(key)); 
		}

		// TODO: Need to load information for parent timer
		PreparedStatement statement = db.prepareStatement("INSERT INTO "
				+ db.getSchemaPrefix() + "counter (trial, name, parent) VALUES (?, ?, ?)");

		for (Iterator<UserEvent> it = dataSource.getUserEvents(); it.hasNext();) {
			UserEvent ue = it.next();

			statement.setInt(1, trialID);
			statement.setString(2, ue.getName());
			if (ue.isContextEvent()) {
				// this is a context event, so get the context
				int contextStart = ue.getName().indexOf(" : ");
				String context = ue.getName().substring(contextStart + 3).trim();
				Integer func = contextMap.get(context);
				// if this is not a callpath profile, we may not have this timer
				if (func == null) {
					context = context.substring(context.lastIndexOf(" => ") + 4).trim();
					func = contextMap.get(context);
				}
				if (func != null) {
					statement.setInt(3, func);
				} else {
					// we are out of ideas, there is no parent.
					statement.setNull(3, java.sql.Types.INTEGER);
				}
			} else {
				statement.setNull(3, java.sql.Types.INTEGER);
			}
			statement.addBatch();

			// TODO: Add this to progress bar
			// this.itemsDone++;
		}
		statement.executeBatch();
		statement.close();
	}

	private static Map<UserEvent, Integer> getUserEventsMap(int trialID,
			DataSource dataSource, DB db) throws SQLException {
		Map<UserEvent, Integer> map = new HashMap<UserEvent, Integer>();

		PreparedStatement statement = db
				.prepareStatement("SELECT id, name FROM "
						+ db.getSchemaPrefix() + "counter WHERE trial=?");
		statement.setInt(1, trialID);
		statement.execute();
		ResultSet results = statement.getResultSet();

		while (results.next()) {
			int funcID = results.getInt(1);
			String name = results.getString(2);

			UserEvent func = dataSource.getUserEvent(name);
			map.put(func, funcID);
		}
		statement.close();
		return map;
	}

	private static void uploadUserEventProfiles(DataSource dataSource,
			Map<UserEvent, Integer> userEventMap, DB db, Map<Thread, Integer> threadMap) throws SQLException {

		List<Thread> threads = dataSource.getThreads();
		PreparedStatement statement = db
				.prepareStatement("INSERT INTO "
						+ db.getSchemaPrefix()
						+ "counter_value (counter, thread, sample_count, maximum_value, minimum_value, mean_value, standard_deviation)"
						+ "VALUES (?, ?, ?, ?, ?, ?, ?)");

		for (Thread thread : threads) {

			for (Iterator<UserEventProfile> it4 = thread.getUserEventProfiles(); it4
					.hasNext();) {
				UserEventProfile uep = it4.next();
				// TODO: handle canceling upload
				// if (this.cancelUpload)
				// return;

				if (uep != null) {
					int atomicEventID = userEventMap.get(uep.getUserEvent())
							.intValue();
				
					statement.setInt(1, atomicEventID);
					statement.setInt(2, threadMap.get(thread));
					statement.setInt(3, (int) uep.getNumSamples());
					statement.setDouble(4, uep.getMaxValue());
					statement.setDouble(5, uep.getMinValue());
					statement.setDouble(6, uep.getMeanValue());
					statement.setDouble(7, uep.getSumSquared());
					statement.addBatch();
				}

			}
		}
		statement.executeBatch();
		statement.close();

	}

	private static void uploadMetadata(Trial trial,
			Map<Function, Integer> functionMap, Map<Thread, Integer> threadMap,
			DB db) throws SQLException {
		int trialID = trial.getID();
		
		// save the primary metadata
		
        PreparedStatement stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                + "primary_metadata (trial, name, value) VALUES (?, ?, ?)");
		for (Map.Entry<String, String> entry : trial.getMetaData().entrySet()) {
		    String key = entry.getKey();
		    String value = entry.getValue();
            stmt.setInt(1, trialID);
            stmt.setString(2, key);
            stmt.setString(3, value);
            stmt.addBatch();
		}
        stmt.executeBatch();
        stmt.close();
        
        stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                + "secondary_metadata (trial, thread, name, value) VALUES (?, ?, ?, ?)");
        for (Thread thread : trial.getDataSource().getThreads()) {
			for (String key : trial.getUncommonMetaData().keySet()) {
			    String value = thread.getMetaData().get(key);
	            stmt.setInt(1, trialID);
	            stmt.setInt(2, threadMap.get(thread));
	            stmt.setString(3, key);
	            stmt.setString(4, value);
	            stmt.addBatch();
			}
    	}
        stmt.executeBatch();
        stmt.close();

        if (trial.getDataSource().getMetadataFile() != null) {
        	try {
	        	String meta = DataSource.readFileAsString(trial.getDataSource().getMetadataFile());
	        	if (MetaDataParserJSON.isJSON(meta)) {
	        		Gson gson = new Gson();
	        		Object obj = gson.fromJson(meta, Object.class);
	        		if (obj.getClass() == LinkedHashMap.class) {
	        			Map<String, Object> map = (LinkedHashMap<String,Object>)obj;
	        			for (Map.Entry<String, Object> entry : map.entrySet()) {
	        				processElement(entry, null, trialID, db);
	        			}
	        		}
	        	}
        	} catch (IOException ioe) {
        		System.err.println("Error parsing metadata file.");
        		System.err.println(ioe.getMessage());
        	} catch (JsonSyntaxException jse) {
        		System.err.println("Error parsing JSON metadata file.");
        		System.err.println(jse.getMessage());
        	}
        }
	}
	
	private static void processElement(Entry<String, Object> entry,
			Integer parent, int trialID, DB db) throws SQLException {
		String key = entry.getKey();
		Object value = entry.getValue();
        PreparedStatement stmt = null;
        // handle special case of top-level metadata with no value
		if (value == null && parent == null) {
	        stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
	                + "primary_metadata (trial, name, value) VALUES (?, ?, ?)");
	        stmt.setInt(1, trialID);
	        stmt.setString(2, key);
	        stmt.setNull(3, java.sql.Types.VARCHAR);
	        stmt.execute();
	        stmt.close();
	    // second case, nested metadata with null value
		} else if (value == null) {
	        stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
	                + "secondary_metadata (trial, name, value, parent) VALUES (?, ?, ?, ?)");
	        stmt.setInt(1, trialID);
	        stmt.setString(2, key);
	        stmt.setNull(3, java.sql.Types.VARCHAR);
	        stmt.setInt(4, parent);
	        stmt.execute();
	        stmt.close();
	    // ok, we have a value.
		} else {
			// is there an inner object?
			if (value.getClass() == LinkedHashMap.class) {
				// insert this object as the parent
		        stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
		                + "secondary_metadata (trial, name, value, parent) VALUES (?, ?, ?, ?)");
		        stmt.setInt(1, trialID);
		        stmt.setString(2, key);
		        stmt.setNull(3, java.sql.Types.VARCHAR);
		        if (parent == null) {
		        	stmt.setNull(4, java.sql.Types.INTEGER);
		        } else {
		        	stmt.setInt(4, parent);
		        }
		        stmt.execute();
		        stmt.close();
	            String tmpStr = new String();
	            if (db.getDBType().compareTo("mysql") == 0)
	                tmpStr = "select LAST_INSERT_ID();";
	            else if (db.getDBType().compareTo("db2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM secondary_metadata";
	            else if (db.getDBType().compareTo("derby") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM secondary_metadata";
	            else if (db.getDBType().compareTo("h2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM secondary_metadata";
	            else if (db.getDBType().compareTo("oracle") == 0)
	                tmpStr = "select " + db.getSchemaPrefix() + "secondary_metadata_id_seq.currval FROM dual";
	            else
	                tmpStr = "select currval('secondary_metadata_id_seq');";
	            int newParent = Integer.parseInt(db.getDataItem(tmpStr));

	            // process the children
	            Map<String, Object> map = (LinkedHashMap<String,Object>)value;
				for (Map.Entry<String, Object> innerEntry : map.entrySet()) {
					processElement(innerEntry, newParent, trialID, db);
				}
			// this is a regular value, could be nested.
			} else {
		        if (parent == null && !(value.getClass() == ArrayList.class)) {
			        stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
			                + "primary_metadata (trial, name, value) VALUES (?, ?, ?)");
			        stmt.setInt(1, trialID);
			        stmt.setString(2, key);
			        stmt.setString(3, value.toString());
		        } else {
			        stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
			                + "secondary_metadata (trial, name, value, parent, is_array) VALUES (?, ?, ?, ?, ?)");
			        stmt.setInt(1, trialID);
			        stmt.setString(2, key);
			        stmt.setString(3, value.toString());
			        if (parent == null) {
			        	stmt.setNull(4, java.sql.Types.INTEGER);
			        } else {
			        	stmt.setInt(4, parent);
			        }
			        // if this value is an array, say so.
		        	stmt.setBoolean(5, (value.getClass() == ArrayList.class));
		        }
		        stmt.execute();
		        stmt.close();
			}
		}
	}


}
