/**
 * 
 */
package edu.uoregon.tau.perfdmf.taudb;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author khuck
 *
 */
public class TAUdbTrial {

	public static final String[] TRIAL_COLUMNS =  {"name", "data_source", "node_count", "contexts_per_node", "threads_per_context", "total_threads"};
	private TAUdbSession session = null;
	private int id = 0; // database ID, zero indicates not set
	private String name = null;
	private TAUdbDataSource dataSource = null;
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

	public TAUdbTrial () {
		super();
	}
	
	public TAUdbTrial (TAUdbSession session, int id, String name, TAUdbDataSource dataSource, int nodeCount, int contextsPerNode, int threadsPerContext, int totalThreads) {
		this.session = session;
		this.id = id;
		this.name = name;
		this.dataSource = dataSource;
		this.nodeCount = nodeCount;
		this.contextsPerNode = contextsPerNode;
		this.threadsPerContext = threadsPerContext;
		this.totalThreads = totalThreads;
		// always do this now?
		this.loadPrimaryMetadata();
		this.timerGroups = new HashSet<TAUdbTimerGroup>();
	}
	
	public static TAUdbTrial getTrial (TAUdbSession session, int id, boolean complete) {
		TAUdbTrial trial = null;
		String query = "select id, name, data_source, node_count, contexts_per_node, threads_per_context, total_threads from trial ";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				String name = results.getString(2);
				Integer sourceID = results.getInt(3);
				TAUdbDataSource source = session.getDataSources().get(sourceID);
				int nodeCount = results.getInt(4);
				int contextsPerNode = results.getInt(5);
				int threadsPerContext = results.getInt(6);
				int totalThreads = results.getInt(7);
				trial = new TAUdbTrial (session, id, name, source, nodeCount, contextsPerNode, threadsPerContext, totalThreads);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		if (trial != null & complete) {
			trial.loadEverything();
		}
		return trial;
	}
	
	private void loadEverything() {
		this.timers = TAUdbTimer.getTimers(session, this);
		this.timerCallpaths = TAUdbTimerCallpath.getTimerCallpaths(session, this);
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
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1,this.id);
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				String key = results.getString(1);
				String value = results.getString(2);
				this.primaryMetadata.put(key, value);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * @return the id
	 */
	public int getId() {
		return id;
	}

	/**
	 * @param id the id to set
	 */
	public void setId(int id) {
		this.id = id;
	}

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @param name the name to set
	 */
	public void setName(String name) {
		this.name = name;
	}

	/**
	 * @return the dataSource
	 */
	public TAUdbDataSource getDataSource() {
		return dataSource;
	}

	/**
	 * @param dataSource the dataSource to set
	 */
	public void setDataSource(TAUdbDataSource dataSource) {
		this.dataSource = dataSource;
	}

	/**
	 * @return the nodeCount
	 */
	public int getNodeCount() {
		return nodeCount;
	}

	/**
	 * @param nodeCount the nodeCount to set
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
	 * @param contextsPerNode the contextsPerNode to set
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
	 * @param threadsPerContext the threadsPerContext to set
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
	 * @param totalThreads the totalThreads to set
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
	 * @param timers the timers to set
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
	 * @param timerGroups the timerGroups to set
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
	 * @param timerCallpaths the timerCallpaths to set
	 */
	public void setTimerCallpaths(Map<Integer, TAUdbTimerCallpath> timerCallpaths) {
		this.timerCallpaths = timerCallpaths;
	}

	/**
	 * @return the threads
	 */
	public Map<Integer, TAUdbThread> getThreads() {
		return threads;
	}

	/**
	 * @param threads the threads to set
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
	 * @param derivedThreads the derivedThreads to set
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
	 * @param timerCalls the timerCalls to set
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
	 * @param session the session to set
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
	 * @param primaryMetadata the primaryMetadata to set
	 */
	public void setPrimaryMetadata(Map<String, String> primaryMetadata) {
		this.primaryMetadata = primaryMetadata;
	}

	/**
	 * @return the metrics
	 */
	public Map<Integer, TAUdbMetric> getMetrics() {
		return metrics;
	}

	/**
	 * @param metrics the metrics to set
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
	 * @param timerValues the timerValues to set
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
	 * @param counters the counters to set
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
	 * @param counterValue the counterValue to set
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
	 * @param secondaryMetadata the secondaryMetadata to set
	 */
	public void setSecondaryMetadata(
			Map<Integer, TAUdbSecondaryMetadata> secondaryMetadata) {
		this.secondaryMetadata = secondaryMetadata;
	}

	public String toString() {
		StringBuilder b = new StringBuilder();
		b.append("id = " + id + ", ");
		b.append("name = " + name + ", ");
		b.append("dataSource = " + dataSource + ", ");
		b.append("nodeCount = " + nodeCount + ", ");
		b.append("contextsPerNode = " + contextsPerNode + ", ");
		b.append("threadsPerContext = " + threadsPerContext + ", ");
		b.append("totalThreads = " + totalThreads + "\n");
		for (String key : primaryMetadata.keySet()) {
			b.append(key + ": " + primaryMetadata.get(key) + "\n");
		}
		return b.toString();
	}
	
	public static Map<Integer, TAUdbTrial> getTrials(TAUdbSession session) {
		Map<Integer, TAUdbTrial> trials = new HashMap<Integer, TAUdbTrial>();
		String query = "select id, name, data_source, node_count, contexts_per_node, threads_per_context, total_threads from trial ";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				String name = results.getString(2);
				Integer sourceID = results.getInt(3);
				TAUdbDataSource source = session.getDataSources().get(sourceID);
				int nodeCount = results.getInt(4);
				int contextsPerNode = results.getInt(5);
				int threadsPerContext = results.getInt(6);
				int totalThreads = results.getInt(7);
				TAUdbTrial trial = new TAUdbTrial (session, id, name, source, nodeCount, contextsPerNode, threadsPerContext, totalThreads);
				trials.put(id, trial);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
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

}
