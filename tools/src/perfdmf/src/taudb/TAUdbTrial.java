package edu.uoregon.tau.perfdmf.taudb;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Vector;

import edu.uoregon.tau.common.MetaDataMap;
import edu.uoregon.tau.common.MetaDataMap.MetaDataKey;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.database.DB;

/**
 * @author khuck
 * 
 */
public class TAUdbTrial extends edu.uoregon.tau.perfdmf.Trial {
	// public class TAUdbTrial {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8756722915938384030L;
	public static final String[] TRIAL_COLUMNS = { "name", "data_source",
			"node_count", "contexts_per_node", "threads_per_context",
			"total_threads" };
	private TAUdbSession session = null;
	//All of following should be inherited from Trial.
	//private int trialID = 0; // database ID, zero indicates not set
	//private String name = null;
	private TAUdbDataSourceType dataSourceType = null;
	private int nodeCount = 0;
	private int contextsPerNode = 0;
	private int threadsPerContext = 0;
	private int totalThreads = 0;
	private Map<String, String> primaryMetadata;
	private Map<Integer, TAUdbSecondaryMetadata> secondaryMetadata;
	private Map<Integer, TAUdbTimer> timers;
	private Set<TAUdbTimerGroup> timerGroups;
	private Map<Integer, TAUdbTimerCallpath> timerCallpaths;
	private Map<Integer, TAUdbThread> threads;
	private Map<Integer, TAUdbTimerCall> timerCalls;
	private List<TAUdbTimerValue> timerValues;
	private Map<Integer, TAUdbThread> derivedThreads;
	private Map<Integer, TAUdbMetric> metrics;
	private Map<Integer, TAUdbCounter> counters;
	private List<TAUdbCounterValue> counterValues;

	// private TAUdbTrial () {
	// super();
	// }

	public TAUdbTrial(TAUdbSession session, int id, String name,
			TAUdbDataSourceType dataSourceType, int nodeCount, int contextsPerNode,
			int threadsPerContext, int totalThreads) {
		this.session = session;
		this.trialID = id;
		this.name = name;
		this.dataSourceType = dataSourceType;
		this.nodeCount = nodeCount;
		this.contextsPerNode = contextsPerNode;
		this.threadsPerContext = threadsPerContext;
		this.totalThreads = totalThreads;
		// always do this now?
		this.loadPrimaryMetadata();
		this.timerGroups = new HashSet<TAUdbTimerGroup>();
	}

	public static TAUdbTrial getTrial(TAUdbSession session, int id,
			boolean complete) {
		TAUdbTrial trial = null;
		String query = "select id, name, data_source, node_count, contexts_per_node, threads_per_context, total_threads from trial ";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(
					query);
			ResultSet results = statement.executeQuery();
			while (results.next()) {
				String name = results.getString(2);
				Integer sourceID = results.getInt(3);
				TAUdbDataSourceType source = session.getDataSources().get(sourceID);
				int nodeCount = results.getInt(4);
				int contextsPerNode = results.getInt(5);
				int threadsPerContext = results.getInt(6);
				int totalThreads = results.getInt(7);
				trial = new TAUdbTrial(session, id, name, source, nodeCount,
						contextsPerNode, threadsPerContext, totalThreads);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		if (trial != null & complete) {
			trial.loadEverything();
		}
		return trial;
	}

	private void loadEverything() {
		this.timers = TAUdbTimer.getTimers(session, this);
		this.timerCallpaths = TAUdbTimerCallpath.getTimerCallpaths(session,
				this);
		this.threads = TAUdbThread.getThreads(session, this, false);
		this.derivedThreads = TAUdbThread.getThreads(session, this, true);
		this.timerCalls = TAUdbTimerCall.getTimerCalls(session, this);
		this.metrics = TAUdbMetric.getMetrics(session, this);
		this.timerValues = TAUdbTimerValue.getTimerValues(session, this);
		this.counters = TAUdbCounter.getCounters(session, this);
		this.counterValues = TAUdbCounterValue.getCounterValues(session, this);
	}

	public void addTimerGroup(TAUdbTimerGroup group) {
		this.timerGroups.add(group);
	}

	private void loadPrimaryMetadata() {
		
		
		this.primaryMetadata = new HashMap<String, String>();
		String query = "select name, value from primary_metadata where trial = ? order by name ";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(
					query);
			statement.setInt(1, this.trialID);
			ResultSet results = statement.executeQuery();
			while (results.next()) {
				String key = results.getString(1);
				String value = results.getString(2);
				this.primaryMetadata.put(key, value);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
	}


	/**
	 * @return the dataSource
	 */
	public TAUdbDataSourceType getDataSourceType() {
		return dataSourceType;
	}

	/**
	 * @param dataSource
	 *            the dataSource to set
	 */
	public void setDataSource(TAUdbDataSourceType dataSource) {
		this.dataSourceType = dataSource;
	}

	/**
	 * @return the nodeCount
	 */
	public int getNodeCount() {
		return nodeCount;
	}

	/**
	 * @param nodeCount
	 *            the nodeCount to set
	 */
	public void setNodeCount(int nodeCount) {
		this.nodeCount = nodeCount;
	}

	/**
	 * @return the contextsPerNode
	 */
	public int getContextsPerNode() {
		return contextsPerNode;
	}

	/**
	 * @param contextsPerNode
	 *            the contextsPerNode to set
	 */
	public void setContextsPerNode(int contextsPerNode) {
		this.contextsPerNode = contextsPerNode;
	}

	/**
	 * @return the threadsPerContext
	 */
	public int getThreadsPerContext() {
		return threadsPerContext;
	}

	/**
	 * @param threadsPerContext
	 *            the threadsPerContext to set
	 */
	public void setThreadsPerContext(int threadsPerContext) {
		this.threadsPerContext = threadsPerContext;
	}

	/**
	 * @return the totalThreads
	 */
	public int getTotalThreads() {
		return totalThreads;
	}

	/**
	 * @param totalThreads
	 *            the totalThreads to set
	 */
	public void setTotalThreads(int totalThreads) {
		this.totalThreads = totalThreads;
	}

	/**
	 * @return the timers
	 */
	public Map<Integer, TAUdbTimer> getTimers() {
		return timers;
	}

	/**
	 * @param timers
	 *            the timers to set
	 */
	public void setTimers(Map<Integer, TAUdbTimer> timers) {
		this.timers = timers;
	}

	/**
	 * @return the timerGroups
	 */
	public Set<TAUdbTimerGroup> getTimerGroups() {
		return timerGroups;
	}

	/**
	 * @param timerGroups
	 *            the timerGroups to set
	 */
	public void setTimerGroups(Set<TAUdbTimerGroup> timerGroups) {
		this.timerGroups = timerGroups;
	}

	/**
	 * @return the timerCallpaths
	 */
	public Map<Integer, TAUdbTimerCallpath> getTimerCallpaths() {
		return timerCallpaths;
	}

	/**
	 * @param timerCallpaths
	 *            the timerCallpaths to set
	 */
	public void setTimerCallpaths(
			Map<Integer, TAUdbTimerCallpath> timerCallpaths) {
		this.timerCallpaths = timerCallpaths;
	}

	/**
	 * @return the threads
	 */
	public Map<Integer, TAUdbThread> getThreads() {
		return threads;
	}

	/**
	 * @param threads
	 *            the threads to set
	 */
	public void setThreads(Map<Integer, TAUdbThread> threads) {
		this.threads = threads;
	}

	/**
	 * @return the derivedThreads
	 */
	public Map<Integer, TAUdbThread> getDerivedThreads() {
		return derivedThreads;
	}

	/**
	 * @param derivedThreads
	 *            the derivedThreads to set
	 */
	public void setDerivedThreads(Map<Integer, TAUdbThread> derivedThreads) {
		this.derivedThreads = derivedThreads;
	}

	/**
	 * @return the timerCalls
	 */
	public Map<Integer, TAUdbTimerCall> getTimerCalls() {
		return timerCalls;
	}

	/**
	 * @param timerCalls
	 *            the timerCalls to set
	 */
	public void setTimerCalls(Map<Integer, TAUdbTimerCall> timerCalls) {
		this.timerCalls = timerCalls;
	}

	/**
	 * @return the session
	 */
	public TAUdbSession getSession() {
		return session;
	}

	/**
	 * @param session
	 *            the session to set
	 */
	public void setSession(TAUdbSession session) {
		this.session = session;
	}

	/**
	 * @return the primaryMetadata
	 */
	public Map<String, String> getPrimaryMetadata() {
		return primaryMetadata;
	}

	/**
	 * @param primaryMetadata
	 *            the primaryMetadata to set
	 */
	public void setPrimaryMetadata(Map<String, String> primaryMetadata) {
		this.primaryMetadata = primaryMetadata;
	}

	/**
	 * @return the metrics
	 */
	public Map<Integer, TAUdbMetric> getTAUdbMetrics() {
		return metrics;
	}

	/**
	 * @param metrics
	 *            the metrics to set
	 */
	public void setMetrics(Map<Integer, TAUdbMetric> metrics) {
		this.metrics = metrics;
	}

	/**
	 * @return the timerValues
	 */
	public List<TAUdbTimerValue> getTimerValues() {
		return timerValues;
	}

	/**
	 * @param timerValues
	 *            the timerValues to set
	 */
	public void setTimerValues(List<TAUdbTimerValue> timerValues) {
		this.timerValues = timerValues;
	}

	/**
	 * @return the counters
	 */
	public Map<Integer, TAUdbCounter> getCounters() {
		return counters;
	}

	/**
	 * @param counters
	 *            the counters to set
	 */
	public void setCounters(Map<Integer, TAUdbCounter> counters) {
		this.counters = counters;
	}

	/**
	 * @return the counterValue
	 */
	public List<TAUdbCounterValue> getCounterValues() {
		return counterValues;
	}

	/**
	 * @param counterValue
	 *            the counterValue to set
	 */
	public void setCounterValues(List<TAUdbCounterValue> counterValues) {
		this.counterValues = counterValues;
	}

	/**
	 * @return the secondaryMetadata
	 */
	public Map<Integer, TAUdbSecondaryMetadata> getSecondaryMetadata() {
		return secondaryMetadata;
	}

	/**
	 * @param secondaryMetadata
	 *            the secondaryMetadata to set
	 */
	public void setSecondaryMetadata(
			Map<Integer, TAUdbSecondaryMetadata> secondaryMetadata) {
		this.secondaryMetadata = secondaryMetadata;
	}

	public MetaDataMap getMetaData() {
		MetaDataMap core = super.getMetaData();
		if (core == null || core.size() <= 0) {
			MetaDataMap newm = new MetaDataMap();

			if (primaryMetadata != null) {
				Set<Entry<String, String>> entries = primaryMetadata.entrySet();
				Iterator<Entry<String, String>> it = entries.iterator();
				while (it.hasNext()) {
					Entry<String, String> en = it.next();
					newm.put(en.getKey(), en.getValue());
				}
			}

			return newm;
		} else {
			return super.getMetaData();
		}
	}



	public static Map<Integer, TAUdbTrial> getTrials(TAUdbSession session) {
		Map<Integer, TAUdbTrial> trials = new HashMap<Integer, TAUdbTrial>();
		String query = "select id, name, data_source, node_count, contexts_per_node, threads_per_context, total_threads from trial ";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(
					query);
			ResultSet results = statement.executeQuery();
			while (results.next()) {
				Integer id = results.getInt(1);
				String name = results.getString(2);
				Integer sourceID = results.getInt(3);
				TAUdbDataSourceType source = session.getDataSources().get(sourceID);
				int nodeCount = results.getInt(4);
				int contextsPerNode = results.getInt(5);
				int threadsPerContext = results.getInt(6);
				int totalThreads = results.getInt(7);
				TAUdbTrial trial = new TAUdbTrial(session, id, name, source,
						nodeCount, contextsPerNode, threadsPerContext,
						totalThreads);
				trials.put(id, trial);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return trials;
	}

	/**
	 * @param args
	 */
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		TAUdbSession session = new TAUdbSession("callpath", false);
		Map<Integer, TAUdbTrial> trials = TAUdbTrial.getTrials(session);
		for (Integer id : trials.keySet()) {
			TAUdbTrial trial = trials.get(id);
			trial.loadEverything();
			System.out.println(trial.toString());
			System.out.println(trial.getDataSource().toString());
		}
		session.close();
	}

	public int saveTrial(DB db) {
		return TAUdbTrial.saveTrialTAUdb(db, trialID, dataSource, name);
	}
	
	public void rename(DB db, String newName) {
		String sql = "UPDATE "
				+ db.getSchemaPrefix()
				+ "trial SET name=? where id=?";
		PreparedStatement statement;
		try {
			statement = db.prepareStatement(sql);
			statement.setString(1, newName);
			statement.setInt(2, trialID);
			statement.executeUpdate();
			statement.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}

		
		this.setName(newName);
		
	}
	public static int saveTrialTAUdb(DB db, int trialID, DataSource dataSource,
			String name) {
		if (db.getSchemaVersion() < 1) {
			System.err
					.println("You can't save a TAUdbTrial to a PerfDMF database.");
			return -1;
		}
		boolean itExists = exists(db, trialID);
		int newTrialID = 0;
		if (dataSource == null) {
			System.out
					.println("Given a null Datasource, must be loaded first, TAUdbTrial");
			return trialID;
		}
		if (itExists) {
			// TODO: Write "update" code
			System.out
					.println("Updates to TAUdb trials have not been implemented yet");
			return trialID;
		}

		try {
			int node_count = dataSource.getMaxNode();
			int contexts_per_node = dataSource.getMaxContextPerNode();
			int threads_per_context = dataSource.getMaxThreadsPerContext();
			int datasource_id = dataSource.getFileType();
			int total_threads = dataSource.getNumThreads();

			String sql = "INSERT INTO "
					+ db.getSchemaPrefix()
					+ "trial (name, data_source,  node_count, contexts_per_node, threads_per_context, total_threads)"
					+ "VALUES (?,?,?,?,?,?" + ") ";
			PreparedStatement statement = db.prepareStatement(sql);
			statement.setString(1, name);

			statement.setInt(2, datasource_id);

			statement.setInt(3, node_count);
			statement.setInt(4, contexts_per_node);
			statement.setInt(5, threads_per_context);
			statement.setInt(6, total_threads);

			statement.executeUpdate();
			statement.close();

			String tmpStr = new String();
			if (db.getDBType().compareTo("mysql") == 0)
				tmpStr = "select LAST_INSERT_ID();";
			else if (db.getDBType().compareTo("db2") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
			else if (db.getDBType().compareTo("sqlite") == 0)
				tmpStr = "select seq from sqlite_sequence where name = 'trial'";
			else if (db.getDBType().compareTo("derby") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
			else if (db.getDBType().compareTo("h2") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
			else if (db.getDBType().compareTo("oracle") == 0)
				tmpStr = "select " + db.getSchemaPrefix()
						+ "trial_id_seq.currval FROM dual";
			else
				tmpStr = "select currval('trial_id_seq');";
			newTrialID = Integer.parseInt(db.getDataItem(tmpStr));

		} catch (SQLException e) {
			System.out.println("An error occurred while saving the trial.");
			e.printStackTrace();
		}
		return newTrialID;
	}

	private static boolean exists(DB db, int trialID) {
		boolean retval = false;
		try {
			PreparedStatement statement = db
					.prepareStatement("SELECT name FROM "
							+ db.getSchemaPrefix() + "trial WHERE id = ?");
			statement.setInt(1, trialID);
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				retval = true;
				break;
			}
			results.close();
		} catch (SQLException e) {
			System.out
					.println("An error occurred while checking to see if the trial exists.");
			e.printStackTrace();
		}
		return retval;
	}

	public static Vector<Trial> getTrialList(DB db, boolean getXMLMetadata,
			String whereClause) {
		try {

			Trial.getMetaData(db);

			// create a string to hit the database
			String buf = "SELECT DISTINCT t.id, t.name, t.data_source, t.node_count, t.contexts_per_node, "
					+ "t.threads_per_context, t.total_threads FROM "
					+ db.getSchemaPrefix() + "trial t " + whereClause
					+" order by t.id";
			Vector<Trial> trials = new Vector<Trial>();

			// System.out.println(buf.toString());
			ResultSet resultSet = db.executeQuery(buf.toString());
			TAUdbSession session = new TAUdbSession(db);
			//int dex=1;
			while (resultSet.next() != false) {
				int pos = 1;

				pos = 1;
				int newID = resultSet.getInt(pos++);
				String newName = (resultSet.getString(pos++));
				int data_source = resultSet.getInt(pos++);
				int node_count = resultSet.getInt(pos++);
				int contextsPerNode = resultSet.getInt(pos++);
				int threadsPerContext = resultSet.getInt(pos++);
				int totalThreads = resultSet.getInt(pos++);

				Trial trial = new TAUdbTrial(session, newID, newName, null,
						node_count, contextsPerNode, threadsPerContext,
						totalThreads);

				trial.setDatabase(db.getDatabase());
                trial.setField("data_source",String.valueOf(data_source));
                trial.setField("node_count",String.valueOf(node_count));
                trial.setField("contexts_per_node",String.valueOf(contextsPerNode));
                trial.setField("threads_per_context",String.valueOf(threadsPerContext));
                trial.setField("total_threads",String.valueOf(totalThreads));
				if (getXMLMetadata) {
					trial.loadXMLMetadata(db);
				}

				trials.addElement(trial);
				//System.out.println("Added trial "+dex);
				//dex++;
			}
			resultSet.close();
			// TODO: Deal with adding the metrics to the trial
			// get the function details
			/*//Don't get metrics until requested.
			Enumeration<Trial> en = trials.elements();
			Trial trial;
			//dex =0;
			while (en.hasMoreElements()) {
				trial = en.nextElement();
				trial.getTrialMetrics(db);
				//System.out.println("got metrics for trial "+dex++);
				
			}*/

			Collections.sort(trials);

			return trials;

		} catch (Exception ex) {
			ex.printStackTrace();
			return null;
		}
	}

	// gets the metric data for the trial
	public void getTrialMetrics(DB db) {
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select id, name, derived ");
		buf.append("from " + db.getSchemaPrefix() + "metric ");
		buf.append("where trial = ");
		buf.append(getID());
		buf.append(" order by id ");
		// System.out.println(buf.toString());

		// get the results
		try {
			ResultSet resultSet = db.executeQuery(buf.toString());
			while (resultSet.next() != false) {
				Metric tmp = new TAUdbMetric(this.session, resultSet.getInt(1),
						this, resultSet.getString(2), resultSet.getBoolean(3));
				addMetric(tmp);
			}
			resultSet.close();
		} catch (Exception ex) {
			ex.printStackTrace();
			return;
		}
		return;
	}

	// private static int getDBMetric(int trialID, int metric) {
	// return 0;
	// }

	public static void deleteMetric(DB db, int trialID, int metricID)
			throws SQLException {
		PreparedStatement statement = null;

		db.setAutoCommit(false);
		try {
			// delete from the interval_location_profile table
			statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
					+ "timer_value WHERE metric = ?");
			statement.setInt(1, metricID);
			statement.execute();
	
			statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
					+ "metric WHERE id = ?");
			statement.setInt(1, metricID);
			statement.execute();
			db.commit();
		} catch (SQLException e) {
			db.rollback();
		}
		db.setAutoCommit(true);
	}

	private static PreparedStatement setTrialIDs(PreparedStatement statement,int[] trialIDs) throws SQLException{
		for(int i=0;i<trialIDs.length;i++){
			statement.setInt(i+1, trialIDs[i]);
		}
		return statement;
	}
	
	public static void deleteTrial(DB db, int[] trialIDs) throws SQLException {
		
		for(int i=0;i<trialIDs.length;i++){
		
		String trialList=" = " + trialIDs[i];
		String idClause =  " WHERE id "+trialList;
		String whereClause=" WHERE trial "+trialList;
		
		System.out.print("Deleting trial ID " + trialIDs[i]);
		long before = System.currentTimeMillis();
		db.setAutoCommit(false);
		try{

		PreparedStatement statement = null;

		// There's a chances that these might not work with MySQL, but after
		// reading the manual

		statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
				+ "primary_metadata"+ whereClause);
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		// "USING" is faster than a subquery - but not all databases support it.
		if (db.getDBType().compareTo("postgresql") == 0) {
			statement = db .prepareStatement(" DELETE FROM " +
				db.getSchemaPrefix() + "time_range x using " + db.getSchemaPrefix() +
				"secondary_metadata y where y.trial " + trialList + " and x.id = y.time_range ");
		} else {
			statement = db .prepareStatement(" DELETE FROM " +
				db.getSchemaPrefix() + "time_range where id in (select time_range from " +
				db.getSchemaPrefix() + "secondary_metadata " + whereClause + ")");
		}
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
				+ "secondary_metadata "+ whereClause);
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		// "USING" is faster than a subquery - but not all databases support it.
		if (db.getDBType().compareTo("postgresql") == 0) {
			statement = db.prepareStatement(" DELETE FROM " 
				+ db.getSchemaPrefix() + "counter_value x using " 
				+ db.getSchemaPrefix() + "counter y where y.trial "
				+ trialList + " and x.counter = y.id ");
		} else {
			statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
				+ "counter_value WHERE counter in (SELECT id FROM "
				+ db.getSchemaPrefix() + "counter "+ whereClause+")");
		}
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		// "USING" is faster than a subquery - but not all databases support it.
		if (db.getDBType().compareTo("postgresql") == 0) {
			statement = db .prepareStatement(" DELETE FROM " 
				+ db.getSchemaPrefix() + "timer_value tv using "
				+ db.getSchemaPrefix() + "timer_call_data tcd, "
				+ db.getSchemaPrefix() + "timer_callpath tcp, " 
				+ db.getSchemaPrefix() + "timer t where t.trial " + trialList 
				+ " and tcp.timer = t.id and tcd.timer_callpath = tcp.id and tv.timer_call_data = tcd.id" );
		} else {
			statement = db .prepareStatement(" DELETE FROM "
				+ db.getSchemaPrefix()
				+ "timer_value tv WHERE tv.timer_call_data IN (SELECT tcd.id FROM "
				+ db.getSchemaPrefix()
				+ "timer_call_data tcd WHERE tcd.timer_callpath IN (SELECT tcp.id FROM "
				+ db.getSchemaPrefix()
				+ "timer_callpath tcp WHERE tcp.timer IN (SELECT t.id FROM "
				+ db.getSchemaPrefix() + "timer t "+ whereClause+")))");
		}
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		// "USING" is faster than a subquery - but not all databases support it.
		if (db.getDBType().compareTo("postgresql") == 0) {
			statement = db .prepareStatement(" DELETE FROM "
				+ db.getSchemaPrefix() + "timer_call_data tcd using "
				+ db.getSchemaPrefix() + "timer_callpath tcp, "
				+ db.getSchemaPrefix() + "timer t where t.trial " + trialList
				+ " and tcp.timer = t.id and tcd.timer_callpath = tcp.id " );
		} else {
			statement = db .prepareStatement(" DELETE FROM "
				+ db.getSchemaPrefix()
				+ "timer_call_data tcd WHERE tcd.timer_callpath IN (SELECT tcp.id FROM "
				+ db.getSchemaPrefix()
				+ "timer_callpath tcp WHERE tcp.timer IN (SELECT t.id FROM "
				+ db.getSchemaPrefix() + "timer t "+ whereClause+"))");
		}
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		// "USING" is faster than a subquery - but not all databases support it.
		if (db.getDBType().compareTo("postgresql") == 0) {
			statement = db .prepareStatement(" DELETE FROM "
				+ db.getSchemaPrefix() + "timer_callpath tcp using "
				+ db.getSchemaPrefix() + "timer t where t.trial " 
				+ trialList + " and tcp.timer = t.id " );
		} else {
			statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
				+ "timer_callpath WHERE timer IN (SELECT id FROM "
				+ db.getSchemaPrefix() + "timer "+ whereClause+")");
		}
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		// "USING" is faster than a subquery - but not all databases support it.
		if (db.getDBType().compareTo("postgresql") == 0) {
			statement = db .prepareStatement(" DELETE FROM "
				+ db.getSchemaPrefix() + "timer_parameter tp using "
				+ db.getSchemaPrefix() + "timer t where t.trial " 
				+ trialList + " and tp.timer = t.id " );
		} else {
			statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
				+ "timer_parameter WHERE timer IN (SELECT id FROM "
				+ db.getSchemaPrefix() + "timer "+ whereClause+")");
		}
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		// "USING" is faster than a subquery - but not all databases support it.
		if (db.getDBType().compareTo("postgresql") == 0) {
			statement = db .prepareStatement(" DELETE FROM "
				+ db.getSchemaPrefix() + "timer_group tg using "
				+ db.getSchemaPrefix() + "timer t where t.trial " 
				+ trialList + " and tg.timer = t.id " );
		} else {
			statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
				+ "timer_group WHERE timer IN (SELECT id FROM "
				+ db.getSchemaPrefix() + "timer "+ whereClause+")");
		}
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
				+ "counter "+ whereClause);
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
				+ "thread "+ whereClause);
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
				+ "timer "+ whereClause);
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
				+ "metric "+ whereClause);
		//statement=setTrialIDs(statement,trialIDs);
		System.out.print(".");
		statement.execute();

		statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
				+ "trial "+ idClause);
		//statement=setTrialIDs(statement,trialIDs);
		statement.execute();
		long after = System.currentTimeMillis();
		System.out.println("done. (" + (after - before) / 1000.0 + " seconds)");
		db.commit();
		} catch (SQLException e) {
			db.rollback();
		}
		db.setAutoCommit(true);
		}
	}
	

	public void updatePrimaryMetadataField(String name, String value){
		updatePrimaryMetadataField(this.getSession().getDB(), this.trialID,
				name, value);
	}

	public static void updatePrimaryMetadataField(DB db, int trialID,
			String name, String value) {
		try {
			PreparedStatement statement = db.prepareStatement("update primary_metadata set value=? where trial=? and name=?;");
			statement.setString(1, value);
			statement.setInt(2, trialID);
			statement.setString(3, name);
			statement.execute();
			statement.close();

		} catch (SQLException e) {
			e.printStackTrace();
		}
	}

	public void addToPrimaryMetadataField(String name, String value) {
		addToPrimaryMetadataField(this.getSession().getDB(), this.trialID,
				name, value);
	}

	public static void addToPrimaryMetadataField(DB db, int trialID,
			String name, String value) {
		try {

			PreparedStatement statement = db
					.prepareStatement("INSERT INTO "
							+ db.getSchemaPrefix()
							+ "primary_metadata (trial, name, value) VALUES (?, ?, ?);");

			statement.setInt(1, trialID);
			statement.setString(2, name);
			statement.setString(3, value);
			statement.execute();
			statement.close();

		} catch (SQLException e) {
			e.printStackTrace();
		}
	}

	public void removeFromPrimaryMetadataField(String name) {
		removeFromPrimaryMetadataField(this.getSession().getDB(), this.trialID,
				name);
	}

	public static void removeFromPrimaryMetadataField(DB db, int trialID,
			String name) {
		try {

			PreparedStatement statement = db.prepareStatement("DELETE FROM "
					+ db.getSchemaPrefix()
					+ "primary_metadata WHERE trial=? AND name=?;");

			statement.setInt(1, trialID);
			statement.setString(2, name);
			statement.execute();
			statement.close();

		} catch (SQLException e) {
			e.printStackTrace();
		}
	}

	public static void updateFields(DB db,int trialID, String field,String value){
//	int node_count = dataSource.getMaxNode();
//	int contexts_per_node = dataSource.getMaxContextPerNode();
//	int threads_per_context = dataSource.getMaxThreadsPerContext();
//	int datasource_id = dataSource.getFileType();
//	int total_threads = dataSource.getNumThreads();

//	String sql = "UPDATE "
//			+ db.getSchemaPrefix()
//			+ "trial (name, data_source,  node_count, contexts_per_node, threads_per_context, total_threads)"
//			+ "VALUES (?,?,?,?,?,?" + ") ";
	String sql = "update trial set "+field+"=? where ID="+trialID;
	PreparedStatement statement;
	try {
		statement = db.prepareStatement(sql);
		statement.setString(1, value);
//		statement.setString(1, name);
//
//		statement.setInt(2, datasource_id);
//
//		statement.setInt(3, node_count);
//		statement.setInt(4, contexts_per_node);
//		statement.setInt(5, threads_per_context);
//		statement.setInt(6, total_threads);

		statement.executeUpdate();
		statement.close();
	} catch (SQLException e) {
		e.printStackTrace();
	}

	}

	public void loadXMLMetadata(DB db, Map<Integer, Function> ieMap) {
		loadMetadata(db, ieMap);
	}

	public void loadMetadata(DB db) {
		Map<Integer, Function> ieMap = new HashMap<Integer, Function>();
		loadMetadata(db, ieMap);
	}

	// Shouldn't this method override loadXMLMetadata?
	public void loadMetadata(DB db, Map<Integer, Function> ieMap) {
		StringBuffer iHateThis = new StringBuffer();
		iHateThis
				.append("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>");
		iHateThis
				.append("<tau:metadata xmlns:tau=\"http://www.cs.uoregon.edu/research/tau\">");
		iHateThis.append("<tau:CommonProfileAttributes>");
		try {
			PreparedStatement statement = db
					.prepareStatement("SELECT name, value FROM "
							+ db.getSchemaPrefix()
							+ "primary_metadata WHERE trial = ?");
			statement.setInt(1, this.trialID);
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				String name = results.getString(1);
				String value = results.getString(2);
				this.metaData.put(name, value);
				iHateThis.append("<tau:attribute><tau:name>");
				iHateThis.append(name);
				iHateThis.append("</tau:name><tau:value>");
				iHateThis.append(value);
				iHateThis.append("</tau:value></tau:attribute>");
			}
			results.close();
			iHateThis.append("</tau:CommonProfileAttributes>");
			// TODO: need to get the secondary metadata and either populate
			// uncommonMetaData, or
			// something similar.

			int node = -1;
			int context = -1;
			int thread = -1;
			boolean inThread = false;

			statement = db
					.prepareStatement("SELECT sm.name, sm.value, t.node_rank, t.context_rank, t.thread_rank, timer_callpath, iteration_start, time_start, parent FROM "
							+ db.getSchemaPrefix()
							+ "secondary_metadata sm left outer join "
							+ db.getSchemaPrefix()
							+ "thread t on sm.thread = t.id left outer join "
							+ db.getSchemaPrefix()
							+ "time_range tr on sm.time_range = tr.id WHERE sm.trial = ? order by t.node_rank, t.context_rank, t.thread_rank");
			statement.setInt(1, this.trialID);
			results = statement.executeQuery();
			edu.uoregon.tau.perfdmf.Thread currentThread = null;
			while (results.next() != false) {
				if (node != results.getInt(3) || context != results.getInt(4)
						|| thread != results.getInt(5)) {
					node = results.getInt(3);
					context = results.getInt(4);
					thread = results.getInt(5);
					if (this.getDataSource() != null) {
						currentThread = this.getDataSource().getThread(node,
								context, thread);
					}
					if (inThread) {
						iHateThis.append("</tau:ProfileAttributes>");
					}
					iHateThis.append("<tau:ProfileAttributes context=\""
							+ context + "\" node=\"" + node + "\" thread=\""
							+ thread + "\">");
					inThread = true;
				}
				MetaDataKey key = this.uncommonMetaData.new MetaDataKey(
						results.getString(1));
				Function f = ieMap.get(results.getInt(6));
				if (f == null) {
					key.timer_context = "";
				} else {
					key.timer_context = f.getName();
				}
				key.call_number = results.getInt(7);
				key.timestamp = results.getLong(8);
				String value = results.getString(2);
				// put this value in the trial's uncommon metadata map
				this.uncommonMetaData.put(key, value);
				// put this value in the thread's metadata map
				if (currentThread != null) {
					currentThread.getMetaData().put(key, value);
				}
				iHateThis.append("<tau:attribute><tau:name>");
				// for now, we need to build a super long string. Ugh.
				String tmpName = key.timer_context + " : " + key.call_number
						+ " : " + key.timestamp + " : " + key.name;
				iHateThis.append(tmpName);
				// iHateThis.append(key.name);
				iHateThis.append("</tau:name><tau:timer_context>");
				iHateThis.append(key.timer_context);
				iHateThis.append("</tau:timer_context><tau:call_number>");
				iHateThis.append(key.call_number);
				iHateThis.append("</tau:call_number><tau:timestamp>");
				iHateThis.append(key.timestamp);
				iHateThis.append("</tau:timestamp><tau:value>");
				iHateThis.append(value);
				iHateThis.append("</tau:value></tau:attribute>");
			}
			results.close();
			if (inThread) {
				iHateThis.append("</tau:ProfileAttributes>");
			}
			iHateThis.append("</tau:metadata>");
			this.setField(XML_METADATA, iHateThis.toString());
			this.setXmlMetaDataLoaded(true);
		} catch (SQLException e) {
			System.out
					.println("An error occurred loading metadata for trial object from TAUdb database.");
			e.printStackTrace();
		}
		return;
	}



}
