package edu.uoregon.tau.perfdmf.taudb;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.UUID;
import java.util.Vector;

import org.postgresql.PGConnection;
import org.postgresql.copy.CopyManager;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

import edu.uoregon.tau.common.MetaDataMap.MetaDataKey;
import edu.uoregon.tau.common.MetaDataMap.MetaDataValue;
import edu.uoregon.tau.perfdmf.CallPathUtilFuncs;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Group;
import edu.uoregon.tau.perfdmf.IntervalLocationProfile;
import edu.uoregon.tau.perfdmf.MetaDataParserJSON;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Parameter;
import edu.uoregon.tau.perfdmf.SourceRegion;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.UserEvent;
import edu.uoregon.tau.perfdmf.UserEventProfile;
import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfdmf.database.DB;
//import org.h2.util.Profiler;

public class TAUdbDatabaseAPI extends DatabaseAPI {
	final static int MEAN_WITHOUT_NULL = -1;
	final static int TOTAL = -2;
	final static int STDDEV_WITHOUT_NULL = -3;
	final static int MIN = -4;
	final static int MAX = -5;
	final static int MEAN_WITH_NULL = -6;
	final static int STDDEV_WITH_NULL = -7;
    private Map<Integer, Experiment> experiments = null;
    private Map<Integer, View> views = null;
	private View view = null;

	public static class TimerCallData {
		public TimerCallData(Function f, Thread t, double d) {
			this.function = f;
			this.thread = t;
			this.timestamp = d;
		}
		public Function function = null;
		public Thread thread = null;
		public double timestamp = 0.0; // for future support of snapshots
		/* (non-Javadoc)
		 * @see java.lang.Object#hashCode()
		 */
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result
					+ ((function == null) ? 0 : function.hashCode());
			result = prime * result
					+ ((thread == null) ? 0 : thread.hashCode());
			long temp;
			temp = Double.doubleToLongBits(timestamp);
			result = prime * result + (int) (temp ^ (temp >>> 32));
			return result;
		}
		/* (non-Javadoc)
		 * @see java.lang.Object#equals(java.lang.Object)
		 */
		@Override
		public boolean equals(Object obj) {
			if (this == obj) {
				return true;
			}
			if (obj == null) {
				return false;
			}
			if (!(obj instanceof TimerCallData)) {
				return false;
			}
			TimerCallData other = (TimerCallData) obj;
			if (function == null) {
				if (other.function != null) {
					return false;
				}
			} else if (!function.equals(other.function)) {
				return false;
			}
			if (thread == null) {
				if (other.thread != null) {
					return false;
				}
			} else if (!thread.equals(other.thread)) {
				return false;
			}
			if (Double.doubleToLongBits(timestamp) != Double
					.doubleToLongBits(other.timestamp)) {
				return false;
			}
			return true;
		}
	}
	
	public TAUdbDatabaseAPI(DatabaseAPI api) {
		super();
		api.copy(this);
	}

	public TAUdbDatabaseAPI() {
		super();
	}

	public static int uploadTrial(DB db, Trial trial, boolean summaryOnly) {

        DataSource dataSource = trial.getDataSource();

        try {
            db.setAutoCommit(false);
        } catch (SQLException e) {
            throw new DatabaseException("Saving Trial Failed: couldn't set AutoCommit to false", e);
        }

        int newTrialID = -1;
        Map<Function, Integer> callpathMap = null;
        Map<Thread, Integer> threadMap = null;
        
        try {
            // save the trial metadata (which returns the new id)
            newTrialID = trial.saveTrial(db);
            trial.setID(newTrialID);
            
            //TODO: Deal with upload size, I think this is for the progress bar.
          //  computeUploadSize(dataSource);
            
            // upload the metrics and get a map that maps the metrics 0 -> n-1 to their unique DB IDs (e.g. 83, 84)
			long before = System.currentTimeMillis();
			long veryStart = before;
			System.out.print("Inserting metrics...");
            uploadMetrics(newTrialID, dataSource.getMetrics(), db);
            Map<Metric, Integer> metricMap = getMetricIDMap(newTrialID, dataSource, db);
			long after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			before = after;
            
			System.out.print("Inserting timers...");
            uploadFunctions(newTrialID, dataSource, db);
            Map<Function, Integer> functionMap = getFunctionsIDMap(newTrialID, dataSource, db);
			after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			before = after;
            
			if (!summaryOnly) { 
			System.out.print("Inserting threads...");
            uploadThreads(newTrialID, dataSource, db);
            threadMap = getThreadsMap(newTrialID, dataSource, db, false);
			after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			before = after;
			}
            
			System.out.print("Inserting timer groups and parameters...");
            uploadTimerGroups(functionMap, db);
            uploadTimerParameter(functionMap, db);
			after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			before = after;
            // this seems confusing... but let me explain. We have uploaded the timers,
            // which are just the flat profile. This call will upload the full call graph,
            // including all the nodes and edges. However, they are mapped from Function
            // objects to timer_callpath rows in the database.
			System.out.print("Inserting call graph...");
            callpathMap = uploadCallpathInfo(dataSource, functionMap, db);
			after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			before = after;
            
			if (summaryOnly) { System.out.println("Only doing summary!"); }

			if (!summaryOnly) { 
            // now that the graph is created, insert the call and subroutine data.
			System.out.print("Inserting per-thread call data...");
            uploadCallDataInfo(dataSource, callpathMap, metricMap, threadMap, db, false);
			after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			before = after;
			}

            // also do it for the derived threads
			System.out.print("Inserting derived threads...");
            Map<Thread, Integer> derivedThreadMap = uploadDerivedThreads(newTrialID, dataSource, db);
			after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			before = after;
			System.out.print("Inserting derived threads call data...");
            uploadCallDataInfo(dataSource, callpathMap, metricMap, derivedThreadMap, db, true);
			after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			before = after;
			System.out.print("Querying new call data for IDs...");
            Map<TimerCallData, Integer> timerCallDataMap = getCallDataMap(newTrialID, dataSource, callpathMap, db);
			after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			before = after;

			System.out.print("Inserting timer measurements...");
			if (!summaryOnly) { 
            // now upload the measurements
            if(db.getDBType().equals("postgresql")){
                uploadFunctionProfilesPSQL(dataSource, timerCallDataMap, metricMap, db);
            }else{
                uploadFunctionProfiles(dataSource, timerCallDataMap, metricMap, db);
                uploadStatistics(dataSource, timerCallDataMap, metricMap, db);
            }
			} else {
                uploadStatistics(dataSource, timerCallDataMap, metricMap, db);
			}
			after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			before = after;

			if (!summaryOnly) { 
			System.out.print("Inserting counters...");
           uploadUserEvents(newTrialID, functionMap, dataSource, db);
           Map<UserEvent, Integer> userEventMap = getUserEventsMap(newTrialID, dataSource, db);

            uploadUserEventProfiles(dataSource, userEventMap, db, threadMap);
			after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			before = after;
		    }
                        
			System.out.print("Inserting metadata...");
            uploadMetadata(dataSource, trial, callpathMap, threadMap, db, summaryOnly);
			after = System.currentTimeMillis();
			System.out.println(" done. (" + (after - before) / 1000.0 + " seconds)");
			System.out.println("Total time to load : " + (after - veryStart) / 1000.0 + " seconds");

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
			Map<TimerCallData, Integer> timerCallDataMap, Map<Metric, Integer> metricMap,
			DB db) throws SQLException {

		Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
		PreparedStatement timerValueInsert = db
				.prepareStatement("INSERT INTO "
						+ db.getSchemaPrefix()
						+ "timer_value (timer_call_data, metric, inclusive_percent, inclusive_value, exclusive_percent, "
						+ " exclusive_value) VALUES (?, ?, ?, ?, ?, ?)");
		for (Metric metric : dataSource.getMetrics()) {
			Integer metricID = metricMap.get(metric);
			if (metricID == null) {
				// not saving this metric
				continue;
			}

			for (Iterator<Function> func = dataSource.getFunctionIterator(); func
					.hasNext();) {
				Function function = func.next();
				if (function.isGroupMember(derived)) {
					continue;
				}
				
				for (Thread thread : dataSource.getAggThreads()) {
					TimerCallData tcd = new TimerCallData(function, thread, 0.0);
					Integer tcdID = timerCallDataMap.get(tcd);
					//this function may not have been called on this thread!
					if (tcdID != null) {
						int timerCallDataID = tcdID.intValue();
						insertDerivedThreadValue(timerValueInsert, metric,
								metricID, timerCallDataID, thread, function);
					}
				}
			}


		}
		timerValueInsert.executeBatch();
		timerValueInsert.close();

	}

	private static void insertDerivedThreadValue(
			PreparedStatement timerValueInsert, Metric metric,
			Integer metricID, Integer timerCallPathID, Thread thread, Function function
			) throws SQLException {
		FunctionProfile fp = thread.getFunctionProfile(function);
		if (fp != null) { // only if this thread calls this function
			// TODO: Deal with cancelUpload
			// if (this.cancelUpload)
			// return;
			timerValueInsert.setInt(1, timerCallPathID);
			timerValueInsert.setInt(2, metricID);
			
			timerValueInsert.setDouble(3, fp.getInclusivePercent(metric.getID()));
			timerValueInsert.setDouble(4, fp.getInclusive(metric.getID()));
			timerValueInsert.setDouble(5, fp.getExclusivePercent(metric.getID()));
			timerValueInsert.setDouble(6, fp.getExclusive(metric.getID()));
			//TODO: Find the sum_exclusive_square values
//						timerValueInsert.setDouble(7, fp.get)
			if (fp.getExclusive(metric.getID()) == 3.142998606E-313){
				System.out.println("Found it");

			}
			
			timerValueInsert.addBatch();	
		}
	}

	private static Integer getIDForCallpath(DataSource dataSource, DB db, Function current, 
			Map<Function, Integer> callpathMap, Map<Function, Integer> functionMap) throws SQLException {
		Integer id = callpathMap.get(current);
		if (id == null) {
			Integer parent = null;
			// get the parent's id
			String parentName = CallPathUtilFuncs.getParentName(current.getName());
			if (!parentName.equals("")) {
				Function parentFunction = dataSource.getFunction(parentName);
				if (parentFunction == null) {
					//System.out.println("Creating non-existent timer! : " + parentName);				
					parentFunction = dataSource.addFunction(parentName, dataSource.getNumberOfMetrics());
				}
                parent = getIDForCallpath(dataSource, db, parentFunction, callpathMap, functionMap);
			}
			// get the timer name
			String timerName = current.getCallpathNodeName();
			Function trueCurrent = dataSource.getFunction(timerName);
			// get the timer ID
			Integer timer = functionMap.get(trueCurrent);
			if (timer == null) {
				return null;
			}
			PreparedStatement statement = db.prepareStatement("INSERT INTO "
					+ db.getSchemaPrefix()
					+ "timer_callpath (timer, parent) "
					+ "VALUES (?, ?)");
			statement.setInt(1, timer.intValue());
			if (parent == null) {
				statement.setNull(2, java.sql.Types.INTEGER);
			} else {
				statement.setInt(2, parent.intValue());
			}
	        statement.execute();
	        statement.close();
            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0)
                tmpStr = "select LAST_INSERT_ID();";
            else if (db.getDBType().compareTo("db2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM timer_callpath";
            else if (db.getDBType().compareTo("sqlite") == 0)
                tmpStr = "select seq from sqlite_sequence where name = 'timer_callpath'";
            else if (db.getDBType().compareTo("derby") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM timer_callpath";
            else if (db.getDBType().compareTo("h2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM timer_callpath";
            else if (db.getDBType().compareTo("oracle") == 0)
                tmpStr = "select " + db.getSchemaPrefix() + "timer_callpath_id_seq.currval FROM dual";
            else
                tmpStr = "select currval('timer_callpath_id_seq');";
            id = Integer.parseInt(db.getDataItem(tmpStr));
            callpathMap.put(current, id);
        }
		return id.intValue();
	}
	
	private static Map<Function, Integer> uploadCallpathInfo(DataSource dataSource,
			Map<Function, Integer> functionMap, DB db) throws SQLException {

		Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
		Map<Function, Integer> callpathMap = new HashMap<Function, Integer>();
		Iterator<Function> funcs = dataSource.getFunctionIteratorCopy();
		while (funcs.hasNext()) {
			Function function = funcs.next();
			if (function.isGroupMember(derived)) {
				continue;
			}
			getIDForCallpath(dataSource, db, function, callpathMap, functionMap);
		}
		return callpathMap;
	}
	
    private static Map<Function, Integer>  getTimerCallPathMap(int trialID, DataSource dataSource, DB db) throws SQLException {
        
	Map<Function, Integer> map = new HashMap<Function, Integer>();

/* This ugly statement is called a CTE, or Common Table Expression.
 * Essentially, this is a recursive statement which builds the callgraph
 * structure for us, including reconstructing the timer names with 
 * the arrows. The "with" statement creates a virtual table "cp".
 * The virtual table is populated with the roots of the trees (our
 * flat profile). Those results are unioned with a recursive 
 * examination of the tree, where leaves are joined on to the 
 * existing tree. As long as the table grows in size, the recursion
 * continues. The final virtual table is queried and ordered
 * by the last select statement.
 * For more details, see: http://wiki.postgresql.org/wiki/CTEReadme
 */

	PreparedStatement statement = 
	  db.prepareStatement("with recursive cp (id, parent, timer, name) as ( " +
			"SELECT tc.id, tc.parent, tc.timer, t.name FROM " +
			db.getSchemaPrefix() +
			"timer_callpath tc inner join " +
			db.getSchemaPrefix() +
			"timer t on tc.timer = t.id where " +
			"t.trial = " + trialID + " and tc.parent is null" +
			"UNION ALL" +
			"SELECT d.id, d.parent, d.timer, " +
			"concat (cp.name, ' => ', dt.name) FROM " +
			db.getSchemaPrefix() +
			"timer_callpath AS d JOIN cp ON (d.parent = cp.id) join " +
			db.getSchemaPrefix() +
			"timer dt on d.timer = dt.id where dt.trial = " + trialID + " ) " +
			"SELECT distinct * FROM cp order by parent;");
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

	
	private static void uploadCallDataInfo(DataSource dataSource,
			Map<Function, Integer> functionMap, Map<Metric, Integer> metricMap,
			Map<Thread, Integer> threadMap, DB db, boolean aggregates) throws SQLException {

		PreparedStatement timerCallpathInsert = db
				.prepareStatement("INSERT INTO "
						+ db.getSchemaPrefix()
						+ "timer_call_data (timer_callpath, thread, calls, subroutines) "
						+ "VALUES (?, ?, ?, ?)");

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

				if (threadID == null) { 
					System.out.println("Function: "+function+" Can't find Thread ID for thread: "+thread);
					continue;
				}
				
				FunctionProfile fp = thread.getFunctionProfile(function);

				if (fp != null && fp.getNumCalls() > 0) { // only if this thread calls this function
					// TODO: Deal with cancelUpload
					// if (this.cancelUpload)
					// return;
					timerCallpathInsert.setInt(1, timerID);
					timerCallpathInsert.setInt(2, threadID);
					timerCallpathInsert.setInt(3, (int) fp.getNumCalls());
					timerCallpathInsert.setInt(4, (int) fp.getNumSubr());
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
	public static Map<String, Integer> getCallDataMap(int trialID, DB db) throws SQLException {
		Map<String, Integer> map = new HashMap<String, Integer>();
		StringBuilder sb = new StringBuilder();
		/* Apparently, the H2 database doesn't handle parameters in the 
		   recursive part of the query. So, we don't add them as parameters. */
		sb.append("with recursive cp (id, parent, timer, name) as ( " +
				"SELECT tc.id, tc.parent, tc.timer, t.name FROM " +
//				"SELECT tc.id,t.name FROM " +
				db.getSchemaPrefix() +
				"timer_callpath tc inner join " +
				db.getSchemaPrefix() +
				"timer t on tc.timer = t.id where ");
				sb.append("t.trial = " + trialID + " and tc.parent is null ");
				sb.append("UNION ALL SELECT d.id, d.parent, d.timer, ");
        if (db.getDBType().compareTo("h2") == 0) {
			sb.append("concat (cp.name, ' => ', dt.name) FROM ");
        } else {
			sb.append("cp.name || ' => ' || dt.name FROM ");
        }
			sb.append(db.getSchemaPrefix() +
				"timer_callpath AS d JOIN cp ON (d.parent = cp.id) join " +
				db.getSchemaPrefix());
				sb.append("timer dt on d.timer = dt.id where dt.trial = " + trialID + ") ");
	        	sb.append("SELECT distinct tcd.id, tcd.time_range, cp.name, h.node_rank, h.context_rank, h.thread_rank, cp.id FROM cp join ");
		sb.append(db.getSchemaPrefix());
		sb.append("timer_call_data tcd on tcd.timer_callpath = cp.id join ");
		sb.append(db.getSchemaPrefix());
		sb.append("thread h on tcd.thread = h.id where h.trial = " + trialID);
		PreparedStatement statement = db.prepareStatement(sb.toString());
		//System.out.println(statement);
		statement.execute();
		ResultSet results = statement.getResultSet();

		while (results.next()) {
			int id = results.getInt(1);
			Double timestamp = results.getDouble(2);
			String functionName = results.getString(3);
			int node = results.getInt(4);
			int context =results.getInt(5);
			int thread = results.getInt(6);
			int timerID = results.getInt(7);
			//System.out.println(timerID);
			map.put(functionName, timerID);
		}
		statement.close();
        return map;
	}
	public static Map<String, ArrayList<Integer>> getCallDataMap(List<Integer> trials, DB db) throws SQLException {
		 Map<String, ArrayList<Integer>>map = new  HashMap<String, ArrayList<Integer>>();
		StringBuilder sb = new StringBuilder();
		/* Apparently, the H2 database doesn't handle parameters in the 
		   recursive part of the query. So, we don't add them as parameters. */
		sb.append("with recursive cp (id, parent, timer, name) as ( " +
				"SELECT tc.id, tc.parent, tc.timer, t.name FROM " +
//				"SELECT tc.id,t.name FROM " +
				db.getSchemaPrefix() +
				"timer_callpath tc inner join " +
				db.getSchemaPrefix() +
				"timer t on tc.timer = t.id ");
		for(int i=0; i<trials.size();i++){
			if(i!=0) { sb.append(" or "); } else { sb.append(" where ( "); }
			sb.append("t.trial = " + trials.get(i) + " ");
		}
		if (trials.size() > 0) { sb.append(")"); }
		sb.append(" and tc.parent is null ");//Removed paren, may not be acceptable

				sb.append("UNION ALL SELECT d.id, d.parent, d.timer, ");
        if (db.getDBType().compareTo("h2") == 0) {
			sb.append("concat (cp.name, ' => ', dt.name) FROM ");
        } else {
			sb.append("cp.name || ' => ' || dt.name FROM ");
        }
			sb.append(db.getSchemaPrefix() + "timer_callpath AS d JOIN cp ON (d.parent = cp.id) join " +
			db.getSchemaPrefix());
			sb.append("timer dt on d.timer = dt.id ");
		    for(int i=0; i<trials.size();i++) {
			  if(i!=0) { sb.append(" or "); } else { sb.append(" where ( "); }
			  sb.append("dt.trial = " + trials.get(i) + " ");
			}
			if (trials.size() > 0) { sb.append(")"); }
			sb.append(")");
	        sb.append("SELECT distinct tcd.id, tcd.time_range, cp.name, h.node_rank, h.context_rank, h.thread_rank, cp.id FROM cp join ");
		sb.append(db.getSchemaPrefix());
		sb.append("timer_call_data tcd on tcd.timer_callpath = cp.id join ");
		sb.append(db.getSchemaPrefix());
		sb.append("thread h on tcd.thread = h.id");
	    for(int i=0; i<trials.size();i++) {
		  if(i!=0) { sb.append(" or "); } else { sb.append(" where ( "); }
		  sb.append("h.trial = " + trials.get(i) + " ");
		}
		if (trials.size() > 0) { sb.append(")"); }
		PreparedStatement statement = db.prepareStatement(sb.toString());
		//System.out.println(statement);
		try{
		statement.execute();
		}catch(Exception e){
			e.printStackTrace();
		}
		ResultSet results = statement.getResultSet();

		while (results.next()) {
			int id = results.getInt(1);
			Double timestamp = results.getDouble(2);
			String functionName = results.getString(3);
			int node = results.getInt(4);
			int context =results.getInt(5);
			int thread = results.getInt(6);
			int timerID = results.getInt(7);
			//System.out.println(timerID);
			ArrayList<Integer>array = map.get(functionName);
			if(array == null){
				array = new ArrayList<Integer>();
				map.put(functionName, array);
			}
			array.add(timerID);
		}
		statement.close();
        return map;
	}
	
	private static Map<TimerCallData, Integer> getCallDataMap(int trialID, DataSource dataSource, Map<Function, Integer> callpathMap, DB db) throws SQLException {
		Map<TimerCallData, Integer> map = new HashMap<TimerCallData, Integer>();
		StringBuilder sb = new StringBuilder();
		Map<Integer,Function> reverseMap = new HashMap<Integer,Function>();
		// The callpath map is from function to ID. we need the reverse, so build it.
		if (callpathMap != null) {
			for (Function key : callpathMap.keySet()) {
				reverseMap.put(callpathMap.get(key), key);
			}
		} else {
			// build a new one from the functions! (for saving a pre-loaded trial)
			Iterator<Function> iter = dataSource.getFunctionIterator();
			while (iter.hasNext()) {
				Function f = iter.next();
				reverseMap.put(f.getID(), f);
			}
		}
		sb.append("select distinct tcd.id, tcd.time_range, cp.id, h.node_rank, h.context_rank, h.thread_rank ");
		sb.append("from timer_callpath cp ");
		sb.append("join timer_call_data tcd on tcd.timer_callpath = cp.id ");
		sb.append("join thread h on tcd.thread = h.id where h.trial = " + trialID);
		PreparedStatement statement = db.prepareStatement(sb.toString());
		//System.out.println(statement);
		statement.execute();
		ResultSet results = statement.getResultSet();

		while (results.next()) {
			int id = results.getInt(1);
			Double timestamp = results.getDouble(2);
			Function f = reverseMap.get(results.getInt(3));
			int node = results.getInt(4);
			int context =results.getInt(5);
			int thread = results.getInt(6);
			Thread t = null;
			if (node < 0) {
				switch (node) {
				case MEAN_WITHOUT_NULL:
					t = dataSource.getMeanData();
					break; 
				case TOTAL:
					t = dataSource.getTotalData();
					break;
				case STDDEV_WITHOUT_NULL:
					t = dataSource.getStdDevData();
					break;
				case MIN:
					t = dataSource.getMinData();
					break;
				case MAX:
					t = dataSource.getMaxData();
					break;
				case MEAN_WITH_NULL:
					t = dataSource.getMeanDataAll();
					break;
				case STDDEV_WITH_NULL:
					t = dataSource.getStdDevDataAll();
					break;
				default:
					break;
				}
			} else {
				t = dataSource.getThread(node, context, thread);
			}
			TimerCallData cpd = new TimerCallData(f, t, timestamp);
			map.put(cpd, id);
		}
		statement.close();
        return map;
	}

	public static Map<Thread, Integer> getThreadsMap(int trialID,DataSource dataSource, DB db, boolean createIfNotFound) throws SQLException {
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
			if (t == null && createIfNotFound && node >= 0) {
				dataSource.addThread(node, context, thread);
			}
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
				map.put(dataSource.getMeanDataNoNull(), id);
				break; 
			case TOTAL:
				map.put(dataSource.getTotalData(), id);
				break;
			case STDDEV_WITHOUT_NULL:
				map.put(dataSource.getStdDevDataNoNull(), id);
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
		Group callpath = dataSource.getGroup("TAU_CALLPATH");
		PreparedStatement statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
				+ "timer (trial, name, source_file, line_number, line_number_end, column_number, column_number_end, short_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)");

		for (Iterator<Function> it = dataSource.getFunctionIterator(); it.hasNext();) {
			Function f = it.next();
			if (f.isGroupMember(derived)) {
				continue; //Should we save the derived callpath functions??
			}
			if (f.isCallPathFunction()) {
				continue; //don't save the a => b => c stuff here - in the timer_callpath table instead.
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
			Map<TimerCallData, Integer> timerCallDataMap, Map<Metric, Integer> metricMap,
			DB db) throws SQLException {
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

			for (Iterator<Function> func = dataSource.getFunctionIterator(); func
					.hasNext();) {
				Function function = func.next();
				if (function.isGroupMember(derived)) {
					continue;
				}
				for (Thread thread : dataSource.getAllThreads()) {
					TimerCallData tcd = new TimerCallData(function, thread, 0.0);
					Integer timerCallDataID = timerCallDataMap.get(tcd);
					FunctionProfile fp = thread.getFunctionProfile(function);
					if (fp != null && timerCallDataID != null) { // only if this thread calls this function
						buf.append(timerCallDataID + "\t");
						buf.append(metricID + "\t");

						buf.append(checkDoubleValue(fp.getInclusivePercent(metric.getID()))
								+ "\t");
						buf.append(checkDoubleValue(fp.getInclusive(metric.getID())) + "\t");
						buf.append(checkDoubleValue(fp.getExclusivePercent(metric.getID()))
								+ "\t");
						buf.append(checkDoubleValue(fp.getExclusive(metric.getID())) + "\n");
					}
				}
				// now the aggregate threads
				for (Thread thread : dataSource.getAggThreads()) {
					TimerCallData tcd = new TimerCallData(function, thread, 0.0);
					Integer timerCallDataID = timerCallDataMap.get(tcd);
					FunctionProfile fp = thread.getFunctionProfile(function);
					if (fp != null && timerCallDataID != null) { // only if this thread calls this function
						buf.append(timerCallDataID + "\t");
						buf.append(metricID + "\t");

						buf.append(checkDoubleValue(fp.getInclusivePercent(metric.getID()))
								+ "\t");
						buf.append(checkDoubleValue(fp.getInclusive(metric.getID())) + "\t");
						buf.append(checkDoubleValue(fp.getExclusivePercent(metric.getID()))
								+ "\t");
						buf.append(checkDoubleValue(fp.getExclusive(metric.getID())) + "\n");
					}
				}
			}
		}
		// copy data from the given input stream to the table
		InputStream input = new ByteArrayInputStream(buf.toString().getBytes());
		try {
			copy.copyIn(
					"COPY timer_value (timer_call_data, metric, inclusive_percent, inclusive_value, exclusive_percent, "
							+ " exclusive_value) FROM STDIN", input);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	private static double checkDoubleValue(double d){
		/*
		 * From the PostgreSQL Docs:
		 * The double precision type typically has a range of around 1E-307 to 1E+308 with a precision of at least 15 digits. 
		 * Values that are too large or too small will cause an error.
		 * Rounding may take place if the precision of an input number is too high. 
		 */
		if(d<1E-307){
			return 0;
		}else if (d>1E+308){
			return 9E+307;
		}
		return d;
			
		
	}

	private static void uploadFunctionProfiles(DataSource dataSource,
			Map<TimerCallData, Integer> timerCallDataMap, Map<Metric, Integer> metricMap, DB db) throws SQLException {

		PreparedStatement timerValueInsert = db
				.prepareStatement("INSERT INTO "
						+ db.getSchemaPrefix()
						+ "timer_value (timer_call_data, metric, inclusive_percent, inclusive_value, exclusive_percent, "
						+ " exclusive_value) VALUES (?, ?, ?, ?, ?, ?)");

		Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");

		for (Metric metric:  dataSource.getMetrics()) {
			Integer metricID = metricMap.get(metric);
			if (metricID == null) {
				// skip this metric
				continue;
			}

			for (Iterator<Function> func = dataSource.getFunctionIterator(); func.hasNext();) {
				Function function = func.next();
				if (function.isGroupMember(derived)) {
					continue;
				}
				for (Thread thread : dataSource.getAllThreads()) {
					TimerCallData tcd = new TimerCallData(function, thread, 0.0); //TODO - this should be a timestamp, not null
					Integer ID = timerCallDataMap.get(tcd);
					// not all threads call all functions! If null, skip this permutation.
					if (ID != null) {
						int timerCallDataID = ID.intValue();
						insertDerivedThreadValue(timerValueInsert, metric,
							metricID, timerCallDataID, thread, function);
					}
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
				+ db.getSchemaPrefix() + "counter (trial, name) VALUES (?, ?)");

		for (Iterator<UserEvent> it = dataSource.getUserEventIterator(); it.hasNext();) {
			UserEvent ue = it.next();

			statement.setInt(1, trialID);
			statement.setString(2, ue.getName());
/*			if (ue.isContextEvent()) {
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
*/			statement.addBatch();

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
			if(func==null)
			{
				System.out.println("Warning: Event "+name+" not found in data-source.");
			}
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
					UserEvent ue = uep.getUserEvent();
					Integer uei=userEventMap.get(ue);
						if(uei==null){
							System.out.println("Error: No id mapped for "+ue.getName());
						}
					int atomicEventID = uei
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

	private static void uploadMetadata(DataSource dataSource, Trial trial,
			Map<Function, Integer> functionMap, Map<Thread, Integer> threadMap,
			DB db, boolean summaryOnly) throws SQLException {
		int trialID = trial.getID();
		
		// save the primary metadata
		
        PreparedStatement stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                + "primary_metadata (trial, name, value) VALUES (?, ?, ?)");
		for (Map.Entry<MetaDataKey, MetaDataValue> entry : trial.getMetaData().entrySet()) {
		    String key = entry.getKey().name;
		    String value = entry.getValue().value.toString();
            stmt.setInt(1, trialID);
            stmt.setString(2, key);
            stmt.setString(3, value);
            stmt.addBatch();
		}
        stmt.executeBatch();
        stmt.close();
		if (!summaryOnly) { 
        Map<List<Integer>, Integer> timestampMap = new HashMap<List<Integer>, Integer>();
        stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                + "secondary_metadata (id, trial, thread, timer_callpath, time_range, parent, name, value, is_array) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
        for (Thread thread : trial.getDataSource().getThreads()) {
			for (MetaDataKey key : trial.getUncommonMetaData().keySet()) {
			    MetaDataValue value = thread.getMetaData().get(key);
			    if (value == null) { continue; }

			    int timer_callpath = 0;
			    int time_range = 0;
	            if (key.timer_context != null) {
		            timer_callpath = functionMap.get(dataSource.getFunction(key.timer_context));
					time_range = getTimestampID(db, key, timestampMap, stmt);
	            }
	      
		       
		        stmt.setString(1, UUID.randomUUID().toString());
		        stmt.setInt(2, trialID);
	            stmt.setInt(3, threadMap.get(thread));
	            if (key.timer_context != null) {
		            stmt.setInt(4, timer_callpath);
		            stmt.setInt(5, time_range);
		            stmt.setNull(6, java.sql.Types.INTEGER);
	            } else {
		            stmt.setNull(4, java.sql.Types.INTEGER);
		            stmt.setNull(5, java.sql.Types.INTEGER);
		            stmt.setNull(6, java.sql.Types.INTEGER);
	            }
	            stmt.setString(7, key.name);
	            stmt.setString(8, value.value.toString());
	            stmt.setBoolean(9, false);
				//System.out.println(stmt.toString());
	            stmt.addBatch();
			}
    	}
      	stmt.executeBatch();
        stmt.close();
		}

        if (trial.getDataSource().getMetadataFile() != null) {
        	try {
        		System.out.print("Parsing metadata file: " + trial.getDataSource().getMetadataFile() + "...");
	        	String meta = DataSource.readFileAsString(trial.getDataSource().getMetadataFile());
	        	if (MetaDataParserJSON.isJSON(meta)) {
	        		//Profiler prof = new Profiler();
	        		//prof.startCollecting();
	        		Gson gson = new Gson();
	        		Object obj = gson.fromJson(meta, Object.class);
	        		System.out.print("done.\nUploading to database...");
	        		if (obj.getClass() == LinkedHashMap.class) {
	        			Map<String, Object> map = (LinkedHashMap<String,Object>)obj;
	    		        stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
	    		                + "primary_metadata (trial, name, value) VALUES (?, ?, ?)");
	    		        int count = 0;
	        			for (Map.Entry<String, Object> entry : map.entrySet()) {
	        				count = processPrimaryElement(stmt, entry, null, trialID);
	        			}
        				System.out.print("Primary Metadata Fields Processed: " + count);
		        		System.out.print(", executing batch... ");
	        			stmt.executeBatch();
		        		System.out.println("done.");
	        			stmt.close();
	    		        stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
	    		                + "secondary_metadata (id, trial, name, value, parent, is_array) VALUES (?, ?, ?, ?, ?, ?)");
	        			for (Map.Entry<String, Object> entry : map.entrySet()) {
	        				count = processElement(stmt, entry, null, trialID);
	        			}
        				System.out.print("Secondary Metadata Fields Processed: " + count);
						if (count > 1) {
		        		  System.out.print(", executing batch... ");
	        			  stmt.executeBatch();
		        		  System.out.println("done.");
						}
	        			stmt.close();
	        		}
	        		//prof.stopCollecting();
	        		//System.out.println(prof.getTop(10));
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

	private static int getTimestampID(DB db, MetaDataKey key, Map<List<Integer>, Integer> timestampMap, PreparedStatement stmt)
			throws SQLException {
		List<Long> timestamp = new ArrayList<Long>(2);
		timestamp.set(0, new Long(key.call_number));
		timestamp.set(1, new Long(key.timestamp));
		Integer time_range = timestampMap.get(timestamp);
		if (time_range != null) {
			return time_range.intValue();
		}
		
		//ok, execute the current batch, if necessary
        try {
        	stmt.executeBatch();
        } catch (SQLException esql) {
        	// this batch might have been empty. Ignore?
        }

		// first, create the time_range objects
		PreparedStatement stmt2 = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
		        + "time_range (iteration_start, iteration_end, time_start, time_end) VALUES (?, ?, ?, ?)");
		stmt2.setInt(1, key.call_number);
		stmt2.setInt(2, key.call_number);
		stmt2.setLong(3, key.timestamp);
		stmt2.setLong(4, key.timestamp);
		stmt2.execute();
		stmt2.close();
		
		// get the id of the new time_range object
		String tmpStr = new String();
		if (db.getDBType().compareTo("mysql") == 0)
		    tmpStr = "select LAST_INSERT_ID();";
		else if (db.getDBType().compareTo("db2") == 0)
		    tmpStr = "select IDENTITY_VAL_LOCAL() FROM time_range";
		else if (db.getDBType().compareTo("derby") == 0)
		    tmpStr = "select IDENTITY_VAL_LOCAL() FROM time_range";
		else if (db.getDBType().compareTo("sqlite") == 0)
		    tmpStr = "select seq from sqlite_sequence where name = 'time_range'";
		else if (db.getDBType().compareTo("h2") == 0)
		    tmpStr = "select IDENTITY_VAL_LOCAL() FROM time_range";
		else if (db.getDBType().compareTo("oracle") == 0)
		    tmpStr = "select " + db.getSchemaPrefix() + "time_range_id_seq.currval FROM dual";
		else
		    tmpStr = "select currval('time_range_id_seq');";
		time_range = Integer.parseInt(db.getDataItem(tmpStr));
		return time_range.intValue();
	}
	
	private static int processElement(PreparedStatement stmt, Entry<String, Object> entry,
			String parent, int trialID) throws SQLException {
		int count = 0;
		String key = entry.getKey();
		Object value = entry.getValue();
		String myID = UUID.randomUUID().toString();
        // handle special case of top-level metadata with no value
		if (value == null && parent == null) {
	        stmt.setString(1, myID);
	        stmt.setInt(2, trialID);
	        stmt.setString(3, key);
	        stmt.setNull(4, java.sql.Types.VARCHAR);
	        stmt.setNull(5, java.sql.Types.VARCHAR);
	        stmt.setBoolean(6, false);
	        stmt.addBatch();
	        count++;
	    // second case, nested metadata with null value
		} else if (value == null) {
	        stmt.setString(1, myID);
			stmt.setInt(2, trialID);
	        stmt.setString(3, key);
	        stmt.setString(4, parent);
	        stmt.setNull(5, java.sql.Types.VARCHAR);
	        stmt.setBoolean(6, false);
	        stmt.addBatch();
	        count++;
	    // ok, we have a value.
		} else {
			// is there an inner object?
			if (value.getClass() == LinkedHashMap.class) {
				// insert this object as the parent
		        stmt.setString(1, myID);
		        stmt.setInt(2, trialID);
		        stmt.setString(3, key);
		        stmt.setNull(4, java.sql.Types.VARCHAR);
		        if (parent == null) {
		        	stmt.setNull(5, java.sql.Types.VARCHAR);
		        } else {
		        	stmt.setString(5, parent);
		        }
		        stmt.setBoolean(6, false);
		        stmt.addBatch();
		        count++;

	            // process the children
	            Map<String, Object> map = (LinkedHashMap<String,Object>)value;
				for (Map.Entry<String, Object> innerEntry : map.entrySet()) {
					count += processElement(stmt, innerEntry, myID, trialID);
				}
			// this is a regular value, could be nested.
			} else {
		        stmt.setString(1, myID);
		        stmt.setInt(2, trialID);
		        stmt.setString(3, key);
		        stmt.setString(4, value.toString());
		        if (parent == null) {
		        	stmt.setNull(5, java.sql.Types.VARCHAR);
		        } else {
		        	stmt.setString(5, parent);
		        }
		        stmt.setBoolean(6, (value.getClass() == ArrayList.class));
		        stmt.addBatch();
		        count++;
			}
		}
		return count;
	}

	private static int processPrimaryElement(PreparedStatement stmt, Entry<String, Object> entry,
			String parent, int trialID) throws SQLException {
		int count = 0;
		String key = entry.getKey();
		Object value = entry.getValue();
		String myID = UUID.randomUUID().toString();
        // handle special case of top-level metadata with no value
		if (value == null && parent == null) {
	        stmt.setInt(1, trialID);
	        stmt.setString(2, key);
	        stmt.setString(3, "false");
	        stmt.addBatch();
	        count++;
	    // second case, nested metadata with null value
		} else if (value == null) {
			// skip it.
		} else {
			// is there an inner object?
			if (value.getClass() == LinkedHashMap.class) {
			    // skip it.
			// this is a regular value, could be nested.
			} else {
		        stmt.setInt(1, trialID);
		        stmt.setString(2, key);
		        stmt.setString(3, value.toString());
		        stmt.addBatch();
		        count++;
			}
		}
		return count;
	}

	public void setView(View view) {
		this.view = view;
	}
	
	public View getView() {
		return this.view;
	}
	public View getView(int ID){
		List<View> vs = null;
		if (views == null) {
			int parent = 0;
			if (view != null)
				parent = view.getID();
			vs = View.getViews(parent, db);
			views = new HashMap<Integer, View>();
			for (View v : vs) {
				v.setParent(view);
				views.put(v.getID(), v);
			}
		}
		return views.get(ID);

	}
	
	// returns Vector of View objects
	public List<View> getViewList() throws DatabaseException {
		List<View> vs = null;
		if (views == null) {
			int parent = 0;
			if (view != null)
				parent = view.getID();
			vs = View.getViews(parent, db);
			views = new HashMap<Integer, View>();
			for (View v : vs) {
				v.setParent(view);
				views.put(v.getID(), v);
			}
		}
		return new ArrayList<View>(views.values());
	}

    // returns Vector of Trial objects
	public List<Trial> getTrialList(boolean getMetadata) {
		if (trials == null) {
			List<View> parents = new ArrayList<View>();
			View tmpView = view;
			while (tmpView != null) {
				parents.add(tmpView);
				tmpView = tmpView.getParent();
			}
			trials = new HashMap<Integer, Trial>();
			List<Trial> ts = View.getTrialsForTAUdbView(parents, db);
			if(ts!=null){
			for (Trial t : ts) {
				trials.put(t.getID(), t);
			}}
			return ts;
		}
		return new ArrayList<Trial>(trials.values());
	}
	  /**
     * This returns a list of all unique primary metadata names
     * @return
     */
	public List<String> getPrimaryMetadataNames() {
		List<String> metadataNames = new ArrayList<String>();
		try {
			String sql = "SELECT DISTINCT name FROM " + db.getSchemaPrefix()
					+ "primary_metadata ORDER BY name";
			PreparedStatement statement;

			statement = db.prepareStatement(sql);

			ResultSet results = statement.executeQuery();
			while (results.next()) {
				metadataNames.add(results.getString(1));
			}
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//System.out.println(metadataNames);

		return metadataNames;
	}


    /**
     * Saves the Trial object to the database
     * 
     * @param trial
     * @param saveMetricIndex
     * @return the database index ID of the saved trial record
     */

    public synchronized int saveTrial(Trial trial, Metric saveMetric) throws DatabaseException {
        //long start = System.currentTimeMillis();

        DataSource dataSource = trial.getDataSource();

        //Build an array of group names. This speeds lookup of group names.
        List<String> groupList = new ArrayList<String>();
        for (Iterator<Group> groupIterator = dataSource.getGroups(); groupIterator.hasNext();) {
            Group g = groupIterator.next();
            groupList.add(g.getName());
        }

        // get the metric count
        metrics = trial.getDataSource().getMetrics();
        int metricCount = metrics.size();

        // create the Vectors to store the data
        intervalEvents = new ArrayList<Function>();
        intervalEventData = new Vector<IntervalLocationProfile>();
        atomicEvents = new ArrayList<UserEvent>();
        atomicEventData = new Vector<UserEventProfile>();

        //int fcount = 0;
        //int ucount = 0;

        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");

        this.intervalEvents = dataSource.getFunctions();
        this.atomicEvents = dataSource.getUserEvents();

        totalItems = intervalEvents.size() + intervalEventData.size() + atomicEvents.size() + atomicEventData.size();

        // Now upload to the database
        try {
            db.setAutoCommit(false);
        } catch (SQLException e) {
            throw new DatabaseException("Saving Trial Failed: couldn't set AutoCommit to false", e);
        }

        int newTrialID = 0;

        int saveMetricIndex = -1;
        if (saveMetric != null) {
            saveMetricIndex = saveMetric.getID();
        }

        try {
            // output the trial data, which also saves the intervalEvents,
            // intervalEvent data, user events and user event data

            Hashtable<Integer, Integer> metricHash = null;
            if (saveMetric == null) { // this means save the whole thing???
            	uploadTrial(db, trial, false);
            } else {
                newTrialID = trial.getID();
                metricHash = saveMetrics(newTrialID, trial, saveMetricIndex);
                Map<Metric, Integer> metricMap = new HashMap<Metric, Integer>();
                // create a hash map with one metric, so that we can reuse code but only save one metric.
                metricMap.put(trial.getDataSource().getMetrics().get(saveMetricIndex), metricHash.get(saveMetricIndex));
                Map<TimerCallData, Integer> timerCallDataMap = getCallDataMap(newTrialID, dataSource, null, db);

                // now upload the measurements
                if(db.getDBType().equals("postgresql")){
                    uploadFunctionProfilesPSQL(dataSource, timerCallDataMap, metricMap, db);
                }else{
                    uploadFunctionProfiles(dataSource, timerCallDataMap, metricMap, db);
                    uploadStatistics(dataSource, timerCallDataMap, metricMap, db);
                }
            }

            for (Iterator<Integer> it = metricHash.keySet().iterator(); it.hasNext();) {
                Integer key = it.next();
                int value = metricHash.get(key).intValue();

                for (Iterator<Metric> it2 = trial.getDataSource().getMetrics().iterator(); it2.hasNext();) {
                    Metric metric = it2.next();
                    if (metric.getID() == key.intValue()) {
                        if (value != -1) {
                            metric.setDbMetricID(value);
                        }
                    }
                }
            }

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
        //System.out.println("Elapsed time: " + elapsedSeconds + " seconds.");
        return newTrialID;
    }
    public FunctionProfile getIntervalEventDetail(Function intervalEvent) throws SQLException {
        StringBuffer buf = new StringBuffer();
		buf.append(" WHERE tcd.timer_callpath = " + intervalEvent.getDatabaseID());
        if (metrics != null && metrics.size() > 0) {
            buf.append(" AND tv.metric in (");
            Metric metric;
            for (Iterator<Metric> en = metrics.iterator(); en.hasNext();) {
                metric = en.next();
                buf.append(metric.getID());
                if (en.hasNext()) {
                    buf.append(", ");
                } else {
                    buf.append(") ");
                }
            }
        }
        return IntervalLocationProfile.getIntervalEventDetail(db, intervalEvent, buf.toString());
    }
	

}
