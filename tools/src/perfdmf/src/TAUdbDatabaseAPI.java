package edu.uoregon.tau.perfdmf;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.h2.store.Data;

import edu.uoregon.tau.perfdmf.database.DB;

public class TAUdbDatabaseAPI {

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
            Map<Metric, Integer> metricMap = uploadMetrics(newTrialID, dataSource.getMetrics(), db);
            Map<Function, Integer> functionMap = uploadFunctions(newTrialID, dataSource, db);
            Map<Thread, Integer> threadMap = uploadThreads(newTrialID, dataSource, db);
            
            uploadTimerGroups(functionMap, db);
            uploadTimerParameter(functionMap, db);
            uploadTimerCallpath(functionMap,  db);
            

//TODO: Upload values later.....
            uploadFunctionProfiles(newTrialID, dataSource, functionMap, metricMap, summaryOnly);

            Map<UserEvent, Integer> userEventMap = uploadUserEvents(newTrialID, dataSource, db);

            uploadUserEventProfiles( dataSource, userEventMap, db, threadMap);
//TODO: Deal with cancel upload
//            if (this.cancelUpload) {
//                db.rollback();
//                deleteTrial(newTrialID);
//                return -1;
//            }

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

	private static Map<Thread, Integer> uploadThreads(int trialID,
			DataSource dataSource, DB db) throws SQLException {
		Map<Thread, Integer> map = new HashMap<Thread, Integer>();
		int maxContextPerNode = dataSource.getMaxNCTNumbers()[1] +1;
		int maxThreadsPerContext = dataSource.getMaxNCTNumbers()[2] +1;

		List<Thread> threads = dataSource.getAllThreads();
		for (Thread t : threads) {

			int node_rank = t.getNodeID();
			int context_rank = t.getContextID();
			int thread_rank = t.getThreadID();
			int process_id = 0;//TODO:  Is this saved somewhere I can access if it was collected?
			int thread_id = 0;//TODO:  Is this saved somewhere I can access if it was collected?
			int thread_index = node_rank * maxContextPerNode * maxThreadsPerContext + context_rank * maxThreadsPerContext + thread_rank; 
			
	

			PreparedStatement stmt = db
					.prepareStatement("INSERT INTO "
							+ db.getSchemaPrefix()
							+ "thread (trial, node_rank, context_rank, thread_rank, process_id, thread_id, thread_index) VALUES (?, ?, ?, ?, ?, ?,?)");
			stmt.setInt(1, trialID);
			stmt.setInt(2, node_rank);
			stmt.setInt(3, context_rank);
			stmt.setInt(4, thread_rank);
			stmt.setInt(5, process_id);
			stmt.setInt(6, thread_id);
			stmt.setInt(7, thread_index);

			stmt.executeUpdate();
			stmt.close();

			String tmpStr = new String();
			if (db.getDBType().compareTo("mysql") == 0)
				tmpStr = "select LAST_INSERT_ID();";
			else if (db.getDBType().compareTo("db2") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM thread";
			else if (db.getDBType().compareTo("derby") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM thread";
			else if (db.getDBType().compareTo("h2") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM thread";
			else if (db.getDBType().compareTo("oracle") == 0)
				tmpStr = "select " + db.getSchemaPrefix()
						+ "thread_id_seq.currval FROM dual"; //Not sure if this is right...
			else
				tmpStr = "select currval('thread_id_seq');";
			int dbThreadID = Integer.parseInt(db.getDataItem(tmpStr));
			
			map.put(t, new Integer(dbThreadID));
		}
		
		return map;
	}

	private static void uploadTimerCallpath(Map<Function, Integer> map, DB db) throws SQLException {
		System.err.println("Warning: Not saving Callpath information yet");
		
		//TODO: Upload the callpath information like this maybe?
		Set<Function> funcs = map.keySet();
		PreparedStatement statement = db.prepareStatement("INSERT INTO "
				+ db.getSchemaPrefix()
				+ "timer_callpath (timer, parameter_name, parameter_value) VALUES (?, ?, ?)");
		for (Function f : funcs) {
			if(f.isCallPathFunction()){
				
				
			}
//			int timer = map.get(f);
//			List<Callpath> callpaths = f.getCallpaths();
//			if (callpaths != null) {
//				for (Callpath p : callpaths) {
//					statement.setInt(1, timer);
//					statement.setString(2, p.getName());
//					statement.setString(3, p.getValue());
//					statement.addBatch();
//				}
//			}
		}
//		statement.executeBatch();
//		statement.close();		
		
	}

	private static void uploadTimerParameter(Map<Function, Integer> map, DB db) throws SQLException {
		System.err.println("Warning: Not saving parameter information yet");
		
		//TODO: Upload the parameter information like this maybe?
//		Set<Function> funcs = map.keySet();
//		PreparedStatement statement = db.prepareStatement("INSERT INTO "
//				+ db.getSchemaPrefix()
//				+ "timer_parameter (timer, parameter_name, parameter_value) VALUES (?, ?, ?)");
//		for (Function f : funcs) {
//			int timer = map.get(f);
//			List<Parameter> params = f.getParameters();
//			if (params != null) {
//				for (Parameter p : params) {
//					statement.setInt(1, timer);
//					statement.setString(2, p.getName());
//					statement.setString(3, p.getValue());
//					statement.addBatch();
//				}
//			}
//		}
//		statement.executeBatch();
//		statement.close();		
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
					statement.setString(2, f.getName());
					statement.addBatch();
				}
			}
		}
		statement.executeBatch();
		statement.close();

	}
	private static Map<Metric, Integer> uploadMetrics(int trialID,
			List<Metric> metrics, DB db) throws SQLException {
		  Map<Metric, Integer> map = new HashMap<Metric, Integer>();
		for (Metric metric: metrics){
	            PreparedStatement stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
	                    + "metric (name, trial) VALUES (?, ?)");
	            stmt.setString(1, metric.getName());
	            stmt.setInt(2, trialID);
	            stmt.executeUpdate();
	            stmt.close();

	            String tmpStr = new String();
	            if (db.getDBType().compareTo("mysql") == 0)
	                tmpStr = "select LAST_INSERT_ID();";
	            else if (db.getDBType().compareTo("db2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
	            else if (db.getDBType().compareTo("derby") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
	            else if (db.getDBType().compareTo("h2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
	            else if (db.getDBType().compareTo("oracle") == 0)
	                tmpStr = "select " + db.getSchemaPrefix() + "metric_id_seq.currval FROM dual";
	            else
	                tmpStr = "select currval('metric_id_seq');";
	            int dbMetricID = Integer.parseInt(db.getDataItem(tmpStr));
	            map .put(metric, new Integer(dbMetricID));
	            metric.setDbMetricID(dbMetricID);
	        }
	        return map;
	    }
	  // fills the timer table
	    private static Map<Function, Integer>  uploadFunctions(int trialID, DataSource dataSource, DB db) throws SQLException {
	        Map<Function, Integer> map = new HashMap<Function, Integer>();
	       
	        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
	        for (Iterator<Function> it = dataSource.getFunctions(); it.hasNext();) {
	            Function f = it.next();
	            if (f.isGroupMember(derived)) {
	                continue; //Should we save the derived callpath functions??
	            }
	            SourceRegion source = f.getSourceLink();

	            PreparedStatement statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
	                    + "timer (trial, name, source_file, line_number, line_number_end, column_number, column_number_end) VALUES (?, ?, ?, ?, ?, ?, ?)");

	            statement.setInt(1, trialID);
	            statement.setString(2, f.getName());
	            statement.setString(3, source.getFilename());
	            statement.setInt(4, source.getStartLine());
	            statement.setInt(5, source.getEndLine());
	            statement.setInt(6, source.getStartColumn());
	            statement.setInt(7, source.getEndColumn());
	            statement.executeUpdate();
	            statement.close();

	            String tmpStr = new String();
	            if (db.getDBType().compareTo("mysql") == 0)
	                tmpStr = "select LAST_INSERT_ID();";
	            else if (db.getDBType().compareTo("db2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM timer";
	            else if (db.getDBType().compareTo("derby") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM timer";
	            else if (db.getDBType().compareTo("h2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM timer";
	            else if (db.getDBType().compareTo("oracle") == 0)
	                tmpStr = "select " + db.getSchemaPrefix() + "timer_seq.currval FROM dual";
	            else
	                tmpStr = "select currval('timer_id_seq');";
	            int newIntervalEventID = Integer.parseInt(db.getDataItem(tmpStr));

	            map.put(f, new Integer(newIntervalEventID));
//TODO: increment itemsDone for progress bar
//	            this.itemsDone++;
	        }
	        return map;
	    }

	private void uploadFunctionProfiles(DataSource dataSource,
			Map<Function, Integer> functionMap, Map<Metric, Integer> metricMap,
			DB db) throws SQLException {

		PreparedStatement threadInsertStatement = null;

		threadInsertStatement = db
				.prepareStatement("INSERT INTO "
						+ db.getSchemaPrefix()
						+ "timer_value (timer, thread, metric, inclusive_percentage, inclusive_value, exclusive_percentage, " +
						" exclusive_value, call, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");

		Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");

		for (Iterator<Metric> it5 = dataSource.getMetrics().iterator(); it5
				.hasNext();) {
			Metric metric = it5.next();
			Integer dbMetricID = metricMap.get(metric);

			for (Iterator<Function> it4 = dataSource.getFunctions(); it4
					.hasNext();) {
				Function function = it4.next();
				if (function.isGroupMember(derived)) {
					continue;
				}
				Integer intervalEventID = functionMap.get(function);

				edu.uoregon.tau.perfdmf.Thread totalData = dataSource
						.getTotalData();
				// TODO: Save total thread as -2 thread

				for (Thread thread : dataSource.getAllThreads()) {

					FunctionProfile fp = thread.getFunctionProfile(function);

					if (fp != null) { // only if this thread calls this function
					// TODO: Deal with cancelUpload
					// if (this.cancelUpload)
					// return;

						addBatchFunctionProfile(threadInsertStatement, thread,
								metric.getID(), dbMetricID.intValue(), fp,
								intervalEventID.intValue(), false, dataSource
										.getAllThreads().size());
					}
				}
			}
		}

		// totalInsertStatement.executeBatch();
		// meanInsertStatement.executeBatch();
		// threadInsertStatement.executeBatch();


		threadInsertStatement.close();

	}

	    private static Map<UserEvent, Integer> uploadUserEvents(int trialID, DataSource dataSource, DB db) throws SQLException {
	        Map<UserEvent, Integer> map = new HashMap<UserEvent, Integer>();

	        String group = null; // no groups right now?

	        for (Iterator<UserEvent> it = dataSource.getUserEvents(); it.hasNext();) {
	            UserEvent ue = it.next();
	        
//TODO: Need to load information for parent timer
	            PreparedStatement statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
	                    + "counter (trial, name) VALUES (?, ?)");
	            statement.setInt(1, trialID);
	            statement.setString(2, ue.getName());
	            statement.executeUpdate();
	            statement.close();

	            String tmpStr = new String();
	            if (db.getDBType().compareTo("mysql") == 0)
	                tmpStr = "select LAST_INSERT_ID();";
	            else if (db.getDBType().compareTo("db2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM counter";
	            else if (db.getDBType().compareTo("derby") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM counter";
	            else if (db.getDBType().compareTo("h2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM counter";
	            else if (db.getDBType().compareTo("oracle") == 0)
	                tmpStr = "select " + db.getSchemaPrefix() + "counter_id_seq.currval FROM dual";
	            else
	                tmpStr = "select currval('counter_id_seq');";
	            int newAtomicEventID = Integer.parseInt(db.getDataItem(tmpStr));
	            map.put(ue, new Integer(newAtomicEventID));
//TODO: Add this to progress bar
//	            this.itemsDone++;
	        }
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

}
