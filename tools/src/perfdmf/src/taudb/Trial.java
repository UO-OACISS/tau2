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
public class Trial {

	private Session session = null;
	private int id = 0; // database ID, zero indicates not set
	private String name = null;
	private DataSource dataSource = null;
	private int nodeCount = 0;
	private int contextsPerNode = 0;
	private int threadsPerContext = 0;
	private int totalThreads = 0;
	private Map<String, String> primaryMetadata;
	private Map<Integer, SecondaryMetadata> secondaryMetadata;
	private Map<Integer, Timer> timers;
	private Set<TimerGroup> timerGroups;
	private Map<Integer, TimerCallpath> timerCallpaths;
	private Map<Integer, Thread> threads;
	private Map<Integer, TimerCall> timerCalls;
	private List<TimerValue> timerValues;
	private Map<Integer, Thread> derivedThreads;
	private Map<Integer, Metric> metrics;
	private Map<Integer, Counter> counters;
	private List<CounterValue> counterValues;

	public Trial () {
		super();
	}
	
	public Trial (Session session, int id, String name, DataSource dataSource, int nodeCount, int contextsPerNode, int threadsPerContext, int totalThreads) {
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
		this.timerGroups = new HashSet<TimerGroup>();
	}
	
	public static Trial getTrial (Session session, int id, boolean complete) {
		Trial trial = null;
		String query = "select id, name, data_source, node_count, contexts_per_node, threads_per_context, total_threads from trial ";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				String name = results.getString(2);
				Integer sourceID = results.getInt(3);
				DataSource source = session.getDataSources().get(sourceID);
				int nodeCount = results.getInt(4);
				int contextsPerNode = results.getInt(5);
				int threadsPerContext = results.getInt(6);
				int totalThreads = results.getInt(7);
				trial = new Trial (session, id, name, source, nodeCount, contextsPerNode, threadsPerContext, totalThreads);
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
		this.timers = Timer.getTimers(session, this);
		this.timerCallpaths = TimerCallpath.getTimerCallpaths(session, this);
		this.threads = Thread.getThreads(session, this, false);
		this.derivedThreads = Thread.getThreads(session, this, true);
		this.timerCalls = TimerCall.getTimerCalls(session, this);
		this.metrics = Metric.getMetrics(session, this);
		this.timerValues = TimerValue.getTimerValues(session, this);
		this.counters = Counter.getCounters(session, this);
		this.counterValues = CounterValue.getCounterValues(session, this);
	}
	
	public void addTimerGroup(TimerGroup group) {
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
	public DataSource getDataSource() {
		return dataSource;
	}

	/**
	 * @param dataSource the dataSource to set
	 */
	public void setDataSource(DataSource dataSource) {
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
	public Map<Integer, Timer> getTimers() {
		return timers;
	}

	/**
	 * @param timers the timers to set
	 */
	public void setTimers(Map<Integer, Timer> timers) {
		this.timers = timers;
	}

	/**
	 * @return the timerGroups
	 */
	public Set<TimerGroup> getTimerGroups() {
		return timerGroups;
	}

	/**
	 * @param timerGroups the timerGroups to set
	 */
	public void setTimerGroups(Set<TimerGroup> timerGroups) {
		this.timerGroups = timerGroups;
	}

	/**
	 * @return the timerCallpaths
	 */
	public Map<Integer, TimerCallpath> getTimerCallpaths() {
		return timerCallpaths;
	}

	/**
	 * @param timerCallpaths the timerCallpaths to set
	 */
	public void setTimerCallpaths(Map<Integer, TimerCallpath> timerCallpaths) {
		this.timerCallpaths = timerCallpaths;
	}

	/**
	 * @return the threads
	 */
	public Map<Integer, Thread> getThreads() {
		return threads;
	}

	/**
	 * @param threads the threads to set
	 */
	public void setThreads(Map<Integer, Thread> threads) {
		this.threads = threads;
	}

	/**
	 * @return the derivedThreads
	 */
	public Map<Integer, Thread> getDerivedThreads() {
		return derivedThreads;
	}

	/**
	 * @param derivedThreads the derivedThreads to set
	 */
	public void setDerivedThreads(Map<Integer, Thread> derivedThreads) {
		this.derivedThreads = derivedThreads;
	}

	/**
	 * @return the timerCalls
	 */
	public Map<Integer, TimerCall> getTimerCalls() {
		return timerCalls;
	}

	/**
	 * @param timerCalls the timerCalls to set
	 */
	public void setTimerCalls(Map<Integer, TimerCall> timerCalls) {
		this.timerCalls = timerCalls;
	}

	/**
	 * @return the session
	 */
	public Session getSession() {
		return session;
	}

	/**
	 * @param session the session to set
	 */
	public void setSession(Session session) {
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
	public Map<Integer, Metric> getMetrics() {
		return metrics;
	}

	/**
	 * @param metrics the metrics to set
	 */
	public void setMetrics(Map<Integer, Metric> metrics) {
		this.metrics = metrics;
	}

	/**
	 * @return the timerValues
	 */
	public List<TimerValue> getTimerValues() {
		return timerValues;
	}

	/**
	 * @param timerValues the timerValues to set
	 */
	public void setTimerValues(List<TimerValue> timerValues) {
		this.timerValues = timerValues;
	}

	/**
	 * @return the counters
	 */
	public Map<Integer, Counter> getCounters() {
		return counters;
	}

	/**
	 * @param counters the counters to set
	 */
	public void setCounters(Map<Integer, Counter> counters) {
		this.counters = counters;
	}

	/**
	 * @return the counterValue
	 */
	public List<CounterValue> getCounterValues() {
		return counterValues;
	}

	/**
	 * @param counterValue the counterValue to set
	 */
	public void setCounterValues(List<CounterValue> counterValues) {
		this.counterValues = counterValues;
	}

	/**
	 * @return the secondaryMetadata
	 */
	public Map<Integer, SecondaryMetadata> getSecondaryMetadata() {
		return secondaryMetadata;
	}

	/**
	 * @param secondaryMetadata the secondaryMetadata to set
	 */
	public void setSecondaryMetadata(
			Map<Integer, SecondaryMetadata> secondaryMetadata) {
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
	
	public static Map<Integer, Trial> getTrials(Session session) {
		Map<Integer, Trial> trials = new HashMap<Integer, Trial>();
		String query = "select id, name, data_source, node_count, contexts_per_node, threads_per_context, total_threads from trial ";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				String name = results.getString(2);
				Integer sourceID = results.getInt(3);
				DataSource source = session.getDataSources().get(sourceID);
				int nodeCount = results.getInt(4);
				int contextsPerNode = results.getInt(5);
				int threadsPerContext = results.getInt(6);
				int totalThreads = results.getInt(7);
				Trial trial = new Trial (session, id, name, source, nodeCount, contextsPerNode, threadsPerContext, totalThreads);
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
		Session session = new Session("callpath", false);
		Map<Integer, Trial> trials = Trial.getTrials(session);
		for (Integer id : trials.keySet()) {
			Trial trial = trials.get(id);
			trial.loadEverything();
			System.out.println(trial.toString());
			System.out.println(trial.getDataSource().toString());
		}
		session.close();
	}

}
